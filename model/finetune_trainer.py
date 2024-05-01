#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import transformers
from transformers import (
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    freeze_persona_embeds,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)

from model_bart import makeMultiTurnChatbot
from trainer_seq2seq import Seq2SeqTrainer
import faulthandler

faulthandler.enable()

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dictg
    - output_dir: where to save the metrics
    """
    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))

    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    check_output_dir(training_args)

    # Set do_eval
    training_args.do_eval = False

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer, model = makeMultiTurnChatbot(model_name=model_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    # Freeze persona memory embedding
    print('\nFreeze persona encoder ...\n')
    freeze_persona_embeds(model.model)
    # for (name, param) in model.model.named_parameters():
    #     print(name, ': ', param.requires_grad)

    dataset_class = Seq2SeqDataset

    # Get datasets
    train_dataset = (
        dataset_class(
            tokenizer,
            type_path="train",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path="val",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        # if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        if training_args.do_eval
        else None
    )
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path="test",
            data_dir=data_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
        )
        if training_args.do_predict
        else None
    )

    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.task, tokenizer) if training_args.predict_with_generate else None
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(tokenizer, data_args, training_args.tpu_num_cores),
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
    )

    all_metrics = {}

    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(   # transformer.Trainer.train()
            resume_from_checkpoint=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(training_args.output_dir)

    elif training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="val", max_length=data_args.val_max_target_length, num_beams=data_args.eval_beams
        )
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

        if trainer.is_world_process_zero():
            save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    elif training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(
            test_dataset=test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams,
        )
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_output.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
                )

                # special tokens 처리
                # "<user>", "<agent>", "<agent_persona>", "<user_persona>", "<empty>", "<no_ref>" => skip
                # "agent", "user", "@이름@", "@브랜드명@", "@홈페이지@", "@영화@", "@날짜@" => skip하지 않음
                try:
                    with open(f"{model_args.model_name_or_path}/special_tokens_map.json", "r", encoding="utf8") as f:
                        special_tokens_dict = eval(f.readline())
                except:
                    special_tokens_dict = {
                        "bos_token": "</s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>",
                        "additional_special_tokens": [
                            "<user>", "<agent>", "<agent_persona>", "<user_persona>", "<empty>", "<no_ref>",
                            "agent", "user", "@이름@", "@브랜드명@", "@홈페이지@", "@영화@", "@날짜@"
                        ],
                    }
                special_tokens_list = list(special_tokens_dict.values())
                special_tokens = special_tokens_list[:-1]   # additional_special_tokens 제외
                for token in special_tokens_list[-1]:   # additional_special_tokens 처리
                    if token == "agent" or token == "user" or token[0] == "@":
                        continue
                    special_tokens.append(token)

                test_preds_skip = []
                for pred in test_preds:
                    for token in special_tokens:
                        pred = pred.replace(token, "")
                    # test_preds_skip.append(clean_up_tokenization(pred))
                    test_preds_skip.append(pred)
                test_preds = test_preds_skip

                test_preds = lmap(str.strip, test_preds)
                write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

                # Distinct 지표 측정
                from paddlenlp.metrics import Distinct
                n_size = 2
                distinct = Distinct(n_size)
                for pred in test_preds:
                    tokens = tokenizer.tokenize(pred)
                    if len(tokens) < n_size:
                        continue
                    distinct.add_inst(tokens)
                metrics[f"test_distinct_{n_size}"] = round(distinct.score(), 4)

                # BERTScore 지표 측정
                from KoBERTScore import BERTScore
                model_name = "monologg/koelectra-base-v2-discriminator"
                bertscore = BERTScore(model_name_or_path=model_name, best_layer=4)
                with open(f'{data_args.data_dir}/test.target', 'r', encoding='utf8') as f:
                    test_targets = f.readlines()
                test_targets = test_targets[:len(test_preds)]
                for line_num, target in enumerate(test_targets):
                    target = target.replace('\n', '').split('</summary>')[-1]
                    test_targets[line_num] = target
                bert_scores = bertscore(test_targets, test_preds, batch_size=128)
                metrics[f"test_bertscore"] = round(np.mean(bert_scores), 4)

            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

