#   Copyright (c) 2020.  CSIRO Australia.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of
#  the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
#  THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

import argparse
import ast
import json
import logging
import math
import os
import statistics
import time
from argparse import Namespace
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Optional

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from OpenKI import logger, logging_formatter
from OpenKI.Constants import MODEL_ARGS_GROUP, DATA_SOURCES, TRAIN_DATA_HANDLERS, RELATION_SCORERS, EVAL_DATA_HANDLERS, \
    EVALUATORS, BEST_MODEL_LABEL, OPTIMIZERS, FINAL_MODEL_LABEL, NEGATIVE_SAMPLE_STRATEGIES, EVALUATOR_NAMES, \
    TEXT_ENCODINGS, WORD_EMBEDDINGS, TEXT_ENCODING_AGGREGATIONS, STATIC_STATE_RE, IGNORE_OPENKI_EMBEDS, \
    ENTITY_WORD_EMBEDS, PREDICATE_WORD_EMBEDS, FC_TANH_THEN_SUM, CONCAT_THEN_FC_TANH, CONCAT_THEN_FC_RELU, \
    FASTTEXT_EMBEDDING, EMBEDS_CACHE_FILES, NO_EMBEDDING, update_state_dict_names, \
    ENTITY_WORD_DELIMITERS, BERT_EMBEDDING, RANDOM_EMBEDDING, DEGENERATE_MODEL_MARKER, DATA_VARIANTS, \
    CACHE_FILE_VARIANT_SUFFIXES, NYT_FB_ENT_SUFFIX, CACHE_FILE_VARIANTS, STATE_DICT_SCORER_KEY_RE, OPTIMIZER_PARAMS
from OpenKI.LossFunctions import PairwiseRankingLoss
from OpenKI.OpenKI_Data import OpenKIMemoryIndexedTrainDataLoader
from OpenKI.RelationScorers import NeighbouringRelationsEntityEncoder, EntityNeighbourhoodScorer, QueryRelationScorer, \
    DualAttentionRelationScorer, EModelEntityEncoder, MultipleRelationScorer, average_aggregator, ConcatFcCombiner, \
    RelationEncoder, SumCombiner, average_aggregator_normed
from OpenKI.TextEncoders import CachedEmbeddings
from OpenKI.UtilityFunctions import update_args, diff_args


def main_loop(args, action_groups):
    start_time = time.time()

    def save_args(report_text="", report=True, extra_label=""):
        nonlocal last_tensorboard_args
        with open(output_folder / (args.file_name_base + extra_label + "args.txt"), "w") as f_args_:
            json.dump(vars(args), f_args_, indent=4)
        if report:
            logger.info(f"saved args file {args.file_name_base}{extra_label}args.txt {report_text}")
        if tensorboard is not None:
            args_delta = {dif[0]: dif[2] if dif[3] != "removed" else "removed"
                          for dif in diff_args(last_tensorboard_args, args)}
            if 'eval_next' in args_delta:
                del args_delta['eval_next']
            if args_delta:
                tensorboard.add_text("args", re.sub(r'(^|\n)', r'\1    ', json.dumps(args_delta, indent=4)),
                                     global_step=args.epoch)  # indent 4 spaces for verbatim formatting in tensorboard
            last_tensorboard_args = deepcopy(args)

    main_scorer = None
    tensorboard = None

    if args.load_model is not None:
        name_match = None
        if args.load_model_newname_regex is not None:
            name_match = re.match(args.load_model_newname_regex, args.load_model)
        if name_match is not None:
            name_groups = name_match.groups()
            name_start = name_groups[0]
            name_end = name_groups[-1] if len(name_groups) > 1 else ""
            if name_end is None:
                name_end = ""
            file_name_base = name_start + name_end + "_"
        else:
            file_name_base = f"{args.load_model}_"
        if args.jobid is not None:
            file_name_base += f"{args.jobid}_"
    else:
        if args.jobid is None:
            args.jobid = f"OKI-{datetime.now()}".replace(' ', '_')
        file_name_base = f"{args.label}_{args.jobid}_"
    while Path(file_name_base + "args.txt").exists():
        file_name_base += "~"

    args.file_name_base = file_name_base
    output_folder = Path(args.output_folder)
    (output_folder / args.file_name_base).parent.mkdir(parents=True, exist_ok=True)

    logger_file_handler = logging.FileHandler(output_folder / f"{args.file_name_base}.log")
    logger_file_handler.setLevel(logging.INFO)
    logger_file_handler.setFormatter(logging_formatter)
    logger.addHandler(logger_file_handler)
    logger.info(f"--epochs is {args.epochs} before merging args.")
    logger.info("------------------------------ New Run ---------------------------------")
    logger.info(f"logger to file {args.file_name_base}.log")

    model_file_name = ""
    if args.load_model is not None:
        if args.run_to_load:
            model_file_name = f"{args.load_model}model_{args.run_to_load}.pt"

        # Load and update the program arguments
        new_args = args
        try:
            with open(output_folder / (new_args.load_model + new_args.run_to_load + "args.txt")) as f_args:
                args = Namespace(**json.load(f_args))
            logger.info(f"loaded run specific args for {new_args.run_to_load}")
        except FileNotFoundError:
            logger.warning(f'{(new_args.load_model + new_args.run_to_load + "args.txt")} not found! Trying without run')
            with open(output_folder / (new_args.load_model+"args.txt")) as f_args:
                args = Namespace(**json.load(f_args))
            logger.info(f"loaded generic run args {(new_args.load_model+'args.txt')}")
        if new_args.load_model != args.file_name_base:
            logger.warning(f"file name base mismatch: '{new_args.load_model}' passed, '{args.file_name_base}' found in "
                           f"args file!")
        update_args(args, new_args, action_groups, exclude=(MODEL_ARGS_GROUP,), silent=("load_model", "run_to_load"),
                    force=new_args.force_default_args + ["train", "test", "validate", "print_args_only", "run_to_load"])
        args.file_name_base = new_args.file_name_base

        # # for backward compatibility to add newly included program arguments
        if getattr(args, "print_args_only", None) is None:
            args.print_args_only = False
        if getattr(args, "last_epoch_loss", None) is None:
            args.last_epoch_loss = None
        if getattr(args, "epoch", None) is None:
            args.epoch = 0
        if getattr(args, "tensorboard_dir", None) is None:
            tensorboard_dir = Path("runs") / new_args.run_to_load  # likely candidate for previous folder
            args.tensorboard_dir = str(tensorboard_dir)
        else:
            tensorboard_dir = Path(args.tensorboard_dir)
        if getattr(args, "output_folder", None) is None:
            args.output_folder = "output"
        if getattr(args, "embed_dim_pairs", None) is None:
            args.embed_dim_pairs = new_args.embed_dim_pairs

        if getattr(args, "data_folder", None) is None:
            if getattr(args, "nyt_folder", None) is not None:
                dataset_folders = (args.nyt_folder,
                                   getattr(args, 'reverb_folder', None),
                                   getattr(args, 'nyt_folder', None))
                assert getattr(args, "reverb_folder", None) is None, \
                    f"Which folder, nyt, reverb or data_folder? " \
                    f"{' or '.join(dataset_folders)}"
                assert DATA_SOURCES[1] == "nyt", "DATA_SOURCES list has been changed! Second element is not 'nyt'!"
                args.data_folder = args.nyt_folder
                args.data_source = DATA_SOURCES[1]  # this should be nyt...
                args.nyt_folder = None
            elif getattr(args, "reverb_folder", None) is not None:
                assert DATA_SOURCES[0] == "reverb", "DATA_SOURCES list has been changed! First element is not 'reverb'!"
                args.data_folder = args.reverb_folder
                args.data_source = DATA_SOURCES[0]  # this should be reverb...
                args.reverb_folder = None
        elif getattr(args, "nyt_folder", None) is not None or getattr(args, "reverb_folder", None) is not None:
            old_folders = ' or '.join(folder for folder in (getattr(args, 'nyt_folder', None),
                                                            getattr(args, 'reverb_folder', None)) if folder is not None)
            logger.warning(f"Overriding old folder ({old_folders}) with specified data-folder {args.data_folder}")
        if getattr(args, "eval_with", None) is None:
            args.eval_with = []
        if getattr(args, "text_encodings", None) is None:
            args.text_encodings = None
        elif "ENE-entity-word-embeds" in args.text_encodings:
            i = args.text_encodings.index("ENE-entity-word-embeds")
            args.text_encodings[i] = f"ENE-entity-word-embeds,concat-then-FC-relu"
        elif type(args.text_encodings) is list:
            for te_i, te in enumerate(args.text_encodings):
                if type(te) is not list:
                    break
                if te[0] == "ENE-entity-word-embeds":
                    args.text_encodings[te_i][0] = ENTITY_WORD_EMBEDS
        if getattr(args, "eval_next", None) is None:
            args.eval_next = True
        if getattr(args, "embeds_on_cpu", None) is None:
            args.embeds_on_cpu = None
        if getattr(args, "single_gpu", None) is None:
            args.single_gpu = True  # before we had the option, it was always on single gpu
        if getattr(args, "text_embeds_static", None) is None:
            logger.info(f"text_embeds_static was not present, setting to None!")
            args.text_embeds_static = None  # it looks like early versions did not have static text embeddings!
        if getattr(args, "no_learned_relation_embeds", None) is None:
            args.no_learned_relation_embeds = False
        if getattr(args, "sqrt_normed_neighbour_ag", None) is None:
            args.sqrt_normed_neighbour_ag = False
        if getattr(args, "check-nans", None) is None:
            args.check_nans = False
        if getattr(args, "layer_norm_on_fc", None) is None:
            args.layer_norm_on_fc = False
        if getattr(args, "no_activation_on_fc", None) is None:
            args.no_activation_on_fc = False
        if getattr(args, "degenerate_epochs", None) is None:
            args.degenerate_epochs = []
            args.first_degenerate_epoch = None
        if getattr(args, "schedule_to_epoch", None) is None:
            args.schedule_to_epoch = None
        if getattr(args, "lr_final", None) is None:
            args.lr_final = None
        if getattr(args, "batch_size_final", None) is None:
            args.batch_size_final = None
        # if getattr(args, "batch_size_by_lr_schedule", None) is None:
        #     args.batch_size_by_lr_schedule = False
        if getattr(args, "predicate_text_flag", None) is None:
            args.predicate_text_flag = False
        if getattr(args, "opt_params", None) is None:
            args.opt_params = None
        if getattr(args, "stop_on_no_best_for", None) is None:
            # noinspection PyTypeHints
            args.stop_on_no_best_for: Optional[int] = None
        if getattr(args, "with_entailments_to", None) is None:
            args.with_entailments_to = None
        if getattr(args, "untyped_entailments", None) is None:
            args.untyped_entailments = False

        output_folder = Path(args.output_folder)
        torch_device = torch.device(args.device)
        if torch_device.index is None:
            torch_device = torch.device(args.device, torch.cuda.current_device())
        last_tensorboard_args = deepcopy(args)
    else:
        if not args.no_cuda:
            torch_device = torch.device("cuda", torch.cuda.current_device())
        else:
            torch_device = torch.device("cpu")
        args.device = str(torch_device)
        args.last_epoch_loss = None
        args.epoch = 0
        tensorboard_dir = Path("runs") / file_name_base
        args.tensorboard_dir = str(tensorboard_dir)
        args.eval_next = True
        args.degenerate_epochs = []
        args.first_degenerate_epoch = None
        last_tensorboard_args = argparse.Namespace()  # empty for first run so all args get dumped to tensorboard

    if args.detect_anomaly:
        args.check_nans = True

    assert getattr(args, 'data_folder', None) is not None, f"No data folder specified!"

    # setup model statistic for dev scoring and args entry for score values
    if args.eval_with is None or len(args.eval_with) == 0:
        args.eval_with = ("AUC_PR" if args.data_source in ("nyt", "nyt_ccg") else "MAP_ranking_pairs",)
    if args.eval_with[0] == "AUC_PR":
        best_models_by_map = False
        if getattr(args, "max_AUC", None) is None:
            args.max_AUC = 0.
    elif args.eval_with[0].startswith("MAP"):
        best_models_by_map = True
        if getattr(args, "max_MAP", None) is None:
            args.max_MAP = 0.
    else:
        raise NotImplementedError(f"Unable to determine best model statistic from '{args.eval_with[0]}'")

    # Extract entailment threshold and whether tye're untyped from label string (or None if not present)
    # ...dev-NC_0.20_untyped
    if "use_entailments" not in args.data_variants:
        args.news_crawl_entailments = None
        args.with_entailments_to = None
        args.untyped_entailments = None
        logger.info("Not using entailments")
    else:
        entailment_data_folder_re = re.compile(r".*(NC_)?(0.\d+)(_untyped)?$")
        entailment_re_match = entailment_data_folder_re.match(args.data_folder)
        assert entailment_re_match is not None, \
            f"--data-variants 'use_entailments' set but the data source doesn't have them!\n{args.data_folder}\n" \
            f"Use a data folder ending with something like '0.30' or 'NC_0.50' or '0.01_untyped'"
        args.news_crawl_entailments = entailment_re_match.groups()[0] is not None
        args.with_entailments_to = float(entailment_re_match.groups()[1])
        args.untyped_entailments = entailment_re_match.groups()[2] is not None
        logger.info(f"Using {'NC' if args.news_crawl_entailments else 'NS'}"
                     f"{' untyped' if args.untyped_entailments else ''} entailments to {args.with_entailments_to}")

    max_dev_epoch = max(getattr(args, "max_AUC_epoch", 0), getattr(args, "max_MAP_epoch", 0))
    epochs_since_best = args.epoch - max_dev_epoch

    def check_we_can_train():
        if args.train and (args.epoch > args.epochs or (args.epoch >= args.epochs and not args.eval_next)):
            logger.warning(f"All epochs already completed in previous run! Ignoring --train!")
        elif args.train and args.stop_on_no_best_for is not None and epochs_since_best >= args.stop_on_no_best_for \
                and not args.eval_next:
            logger.warning(f"No dev improvement for {epochs_since_best} (more than {args.stop_on_no_best_for}), "
                           f"not training!")
        else:
            return True  # none of the "training done" conditions are fulfilled, go ahead with training
        return False     # at least one "training done" condition is fulfilled, don't do training!

    # check if we've something to do and abort if we don't
    if not (args.train or args.test or args.validate):
        logger.warning("neither --train, --validate nor --test are set, nothing will happen, aborting!")
        return
    elif not (args.test or args.validate):
        if not check_we_can_train():
            # To avoid wasting resources setting up data readers etc... when there is nothing to do, abort now
            return

    # # Set/save the random seeds for torch and numpy
    # if getattr(args, "random_seed", None) is None:
    #     args.random_seed = torch.seed()
    # else:
    #     torch.manual_seed(args.random_seed)
    # np.random.seed(args.random_seed % 2**32)

    # logger.info(file_name"{datetime.now()}: Starting with jobid '{args.jobid}' and label '{args.label}'")
    logger.info(f"Starting with jobid '{args.jobid}' and label '{args.label}' with files '{args.file_name_base}")
    logger.info(json.dumps(vars(args), indent=4))
    if args.print_args_only:
        exit(0)

    # Helper functions for evaluation
    def load_eval_data_readers(split, message, eval_names=None):
        if eval_names is None:
            eval_names = args.eval_with
        logger.info(f"Loading {message} data for eval with {eval_names}...")
        return {
            name: EVAL_DATA_HANDLERS[args.data_source][EVALUATORS[name][0]](
                args.data_folder, split, device=torch_device, variants=args.data_variants,
                enatilments_to=args.with_entailments_to, untyped_entailments=args.untyped_entailments)
            # top_relations=train_data.top_relations)
            for name in eval_names
        }

    def load_evaluators(data_readers, message):
        if main_scorer is None:
            raise ValueError(f"At least one of --train and --load-model-file must be used, else there is no model!")
        if not args.no_cuda:
            main_scorer.cuda()
        num_eval_relations = next(iter(data_readers.values())).num_kb_relations  # same for all data_readers
        if args.eval_batch_size is not None:
            eval_batch_sizes = [(name, data_reader, int(args.eval_batch_size))
                                for name, data_reader in data_readers.items()]
        else:
            eval_batch_sizes = [(name, data_reader, int(args.batch_size * args.negative_rate / num_eval_relations))
                                for name, data_reader in data_readers.items()]
            #  / data_reader.expected_neighbour_len ... the one from batch size could be *2 (at least for NYT_NA)
        logger.info(f"batch sizes set to {', '.join(f'{name}: {size}' for name, _, size in eval_batch_sizes)}")
        return {
            name: EVALUATORS[name][1](main_scorer, data_reader, batch_size=batch_size_, cuda=not args.no_cuda,
                                      device=args.device, only_seen_neighbours=not args.eval_unseen_neighbours,
                                      label=f"{message} Evaluation", t_board=tensorboard)
            for name, data_reader, batch_size_ in eval_batch_sizes
        }

    def do_evaluation(test_evaluators, message):
        logger.info(f"starting {message} evaluation with {args.eval_with}")
        for name, score in ((name, evaluator.evaluate(epoch)) for name, evaluator in test_evaluators.items()):
            logger.info(f"{message} evaluation complete at Epoch {epoch} with {name} = {score}")
            if tensorboard is not None:
                tensorboard.add_text(f"{message} Evaluation", f"{name} = {score}", global_step=epoch)

    # Load training data
    logger.info(f"setup {args.data_source} train data reader")
    train_data_class = TRAIN_DATA_HANDLERS[args.data_source][args.loss_by_pair_ranking]
    train_data = train_data_class(args.data_folder, "train", torch_device, eval_top_n_rels=args.num_top_rels,
                                  use_dev_preds=args.train_with_dev_preds,
                                  openie_as_pos_samples=args.openIE_as_pos_samples,
                                  variants=args.data_variants,
                                  enatilments_to=args.with_entailments_to,
                                  untyped_entailments=args.untyped_entailments)
    del train_data_class

    num_relations = train_data.num_relations
    logger.info("setup scorers")
    text_encoder_types = None
    if args.text_encodings is not None:
        # TODO: set up text encodings for e-model also... I guess EModelEntityEncoder needs adjustment
        if type(args.text_encodings[0]) is str:  # if it's from loaded args, it'll be a list already
            args.text_encodings = [type_spec.split(',') for type_spec in args.text_encodings]
        text_encoder_types = {
            type_spec[0]: type_spec[1:] for type_spec in args.text_encodings
        }  # a TEXT_ENCODINGS entry: zero or more TEXT_ENCODING_AGGREGATIONS entries
        assert all(spec in TEXT_ENCODINGS for spec in text_encoder_types), \
            f"{[spec for spec in text_encoder_types if spec not in TEXT_ENCODINGS]} not a recognised text encoding " \
            f"string (from {text_encoder_types})."
        assert all(all(ag in TEXT_ENCODING_AGGREGATIONS for ag in ag_l) for ag_l in text_encoder_types.values())

    # NOTE: each type of text encoder (entity, predicate, contextual) has a single encoder instance for the whole
    #       model currently. Role specific attributes are embodied in an aggregator object. The idea is that the text
    #       encoder is eg. global (and probably static) word embeds or BERT representations, with role specific
    #       adjustment done in the aggregator (eg: with an FC layer or transformer, the latter not yet implemented).
    text_encoders: Optional[Dict[str, CachedEmbeddings], Dict[None, None]] = {None: None}
    text_encoder_aggregators: Optional[(str, nn.Module), (None, None)] = {None: None}

    def check_text_encoder(encoder_name: str, fc_out_dim: int):
        encoder_key_ = f"{encoder_name}-{fc_out_dim}"
        if encoder_key_ not in text_encoders:
            extra_encoder_params = {}
            if args.word_embeddings == FASTTEXT_EMBEDDING:
                delimiter = ENTITY_WORD_DELIMITERS[args.data_source] if encoder_name == ENTITY_WORD_EMBEDS else ' '
                extra_encoder_params.update({
                    "embedding_file": Path(args.word_embed_file),
                    "word_delimiter": delimiter
                })
            elif args.word_embeddings == BERT_EMBEDDING:
                extra_encoder_params.update({
                    "bert_pipeline": args.bert_pipeline,
                    "fine_tune": args.fine_tune_bert,
                    "cls_only": args.bert_use_cls
                    # TODO: (spans) add option to pass entity spans info, but... see TextEncoders.py l142
                })
            elif args.word_embeddings == RANDOM_EMBEDDING:
                extra_encoder_params.update({
                    "embed_dim": args.random_embed_dim
                })

            embeds_on_cpu = args.embeds_on_cpu is not None and encoder_name in args.embeds_on_cpu
            embeds_static = args.text_embeds_static is not None and encoder_name in args.text_embeds_static
            variant_suffix = ""
            if any(variant in CACHE_FILE_VARIANTS for variant in args.data_variants):
                variant_suffix = ""
                if "entity-text-with-descriptions" in args.data_variants and encoder_name == ENTITY_WORD_EMBEDS:
                    variant_suffix = CACHE_FILE_VARIANT_SUFFIXES["entity-text-with-descriptions"]
                if "pred-text-with-descriptions" in args.data_variants:
                    raise NotImplementedError("no text descriptions of predicates/relations implemented yet")
                if "nyt-text-entities" not in args.data_variants and args.data_source == "nyt":
                    # nyt-text-entities means don't use the FB names. Else we do the new default, use FB
                    variant_suffix += NYT_FB_ENT_SUFFIX
            elif args.data_source == "nyt":
                variant_suffix = NYT_FB_ENT_SUFFIX  # this is the new default for nyt
            cache_file_name = EMBEDS_CACHE_FILES[encoder_name][args.word_embeddings] + variant_suffix + ".pt"
            type_boundaries = None
            if args.predicate_text_flag and encoder_name == PREDICATE_WORD_EMBEDS:
                type_boundaries = (train_data.first_predicate_index,)

            text_encoders[encoder_key_] = TEXT_ENCODINGS[encoder_name][args.word_embeddings](
                cache_file_location=Path(args.data_folder) / cache_file_name,
                dataset=train_data, fc_out_dim=fc_out_dim, fc_layer_norm=args.layer_norm_on_fc,
                embeds_on_cpu=embeds_on_cpu, requires_grad=not embeds_static, no_activation=args.no_activation_on_fc,
                text_type_boundaries=type_boundaries, **extra_encoder_params
            )

    def build_text_encoding_aggregator(encoder_name, embed_dim):
        nonlocal text_encoder_types
        # if text_encoders.get(encoder_name, None) is None:
        #     text_encoders[encoder_name] = None  # ensure it has an entry, even if we're not using it
        if args.text_encodings is not None:
            aggregator_specs = text_encoder_types.get(encoder_name, None)  # relation word embeds
            if aggregator_specs is not None:
                assert len(aggregator_specs) == 1
                aggregator_spec = aggregator_specs[0]
                if aggregator_spec.startswith("FC-tanh"):  # for now only tanh implemented - TextEncoders ~l98
                    fc_out_dim = embed_dim  # the text encoder has the FC layer
                else:
                    assert aggregator_spec.startswith("concat-then-FC") or aggregator_spec == IGNORE_OPENKI_EMBEDS
                    fc_out_dim = None  # the FC layer is in the aggregator, not the text encoder
                encoder_key_ = f"{encoder_name}-{fc_out_dim}"
                if args.word_embeddings in tuple(TEXT_ENCODINGS[encoder_name].keys()):
                    check_text_encoder(encoder_name, fc_out_dim)  # creates the text encoder (if not done before!)
                else:
                    text_encoders[encoder_key_] = None
                    if args.word_embeddings != NO_EMBEDDING:
                        raise NotImplementedError(f"only {','.join(TEXT_ENCODINGS[encoder_name].keys())} embeddings "
                                                  f"implemented at this stage, not {args.word_embeddings}")
                # if args.no_learned_relation_embeds:
                #     return IGNORE_OPENKI_EMBEDS
                if aggregator_spec.startswith("concat-then-FC"):  # eg: "concat-then-FC-relu"
                    assert text_encoders[encoder_key_] is not None, "cannot use concat aggregator " \
                                                                                      "without word embeddings!"
                    if args.no_activation_on_fc:
                        agg_activation = None
                    else:
                        is_relu = aggregator_spec.endswith("-relu")
                        assert is_relu or aggregator_spec.endswith("-tanh"), \
                            "only relu and tanh activations implemented"
                        agg_activation = nn.ReLU() if is_relu else nn.Tanh()
                        # agg_activation = None
                    if args.no_learned_relation_embeds:
                        raise ValueError(f"use '-sum' text representation version instead of {aggregator_spec} when no "
                                         f"relation embeds are learned.")
                    te_aggregator = ConcatFcCombiner(0 if args.no_learned_relation_embeds else args.embed_dim,
                                                     text_encoders[encoder_key_].out_embed_dim, args.embed_dim,
                                                     activation=agg_activation, layer_norm=args.layer_norm_on_fc)
                    text_encoder_aggregators[encoder_key_] = te_aggregator
                    return te_aggregator, encoder_key_
                elif aggregator_spec.endswith("then-sum") and not args.no_learned_relation_embeds:
                    # lambda_sum = lambda x, y: x + y
                    te_aggregator = SumCombiner()
                    text_encoder_aggregators[encoder_key_] = te_aggregator
                    return te_aggregator, encoder_key_
                else:
                    text_encoder_aggregators[encoder_key_] = IGNORE_OPENKI_EMBEDS
                    return IGNORE_OPENKI_EMBEDS, encoder_key_
            else:
                return None, None  # seems we have only one of an entity_encoder and a relation_encoder
        else:
            return None, None

    def build_relation_encoder(embed_dim, num_embeds=None, which_texts=PREDICATE_WORD_EMBEDS, attention_encoding=0,
                               force_relation_encoder=False, s_or_o=None):
        if num_embeds is None:
            num_embeds = num_relations
        relation_embed_aggregator, encoder_key_ = build_text_encoding_aggregator(which_texts, embed_dim)
        if relation_embed_aggregator == IGNORE_OPENKI_EMBEDS or args.no_learned_relation_embeds:
            assert relation_embed_aggregator is not None, f"Either learned relation embeds or a relation text encoder" \
                                                          f"are required!."
            relation_embeddings = None
        else:
            relation_embeddings = nn.Embedding(num_embeds, embed_dim, padding_idx=0)

        if relation_embed_aggregator is not None or force_relation_encoder:
            # TODO: do we want to enable dropout for the RelationEncoder?
            relation_embeddings = RelationEncoder(relation_embedding=relation_embeddings, embed_dim=embed_dim,
                                                  predicate_encoder=text_encoders[encoder_key_],
                                                  predicate_encoding_aggregator=relation_embed_aggregator)
        return relation_embeddings

    subject_encoder = object_encoder = None
    if RELATION_SCORERS[0] in args.relation_scorers or \
            RELATION_SCORERS[2] in args.relation_scorers:
        # Models that need an ENE (Entity Neighbourhood Encoder):
        # entity neighbourhood scorer or dual attention scorer or ene triple tensor scorer
        # TODO: entity embeds cache file should depend on the embedding!
        subject_entity_text_aggregator, encoder_key = build_text_encoding_aggregator(ENTITY_WORD_EMBEDS, args.embed_dim)
        object_entity_text_aggregator, encoder_key = build_text_encoding_aggregator(ENTITY_WORD_EMBEDS, args.embed_dim)

        subject_embeddings = build_relation_encoder(args.embed_dim, force_relation_encoder=False, s_or_o=0)
        object_embeddings = build_relation_encoder(args.embed_dim, force_relation_encoder=False, s_or_o=1)
        # TODO: why did I want force_relation_encoder?? It allows ENE encoders to deal with extra indices for texts with
        #       relation indices, but as far as I can see, they only ever get entity pair indices...

        entity_encoder = text_encoders.get(encoder_key, None)  # f"{ENTITY_WORD_EMBEDS}-{args.embed_dim}"

        if args.sqrt_normed_neighbour_ag:
            # TODO: all runs between 3/8 and 26/9 10pm have this negated! ie: --sqrd... => un-normalised aggregator!!
            aggregator = average_aggregator_normed
        else:
            aggregator = average_aggregator
        subject_encoder = NeighbouringRelationsEntityEncoder(data_loader=train_data, subject_or_object="subject",
                                                             relation_encoder=subject_embeddings,
                                                             aggregator=aggregator,
                                                             max_nbhd_preds=args.max_nbhd_predicates,
                                                             entity_encoding_aggregator=subject_entity_text_aggregator,
                                                             entity_encoder=entity_encoder,
                                                             encoder_dropout=args.entity_emb_dropout)
        object_encoder = NeighbouringRelationsEntityEncoder(data_loader=train_data, subject_or_object="object",
                                                            relation_encoder=object_embeddings,
                                                            aggregator=aggregator,
                                                            max_nbhd_preds=args.max_nbhd_predicates,
                                                            entity_encoding_aggregator=object_entity_text_aggregator,
                                                            entity_encoder=entity_encoder,
                                                            encoder_dropout=args.entity_emb_dropout)

    scorers = []
    initial_scorer_weights = []
    unweighted_scorers = False
    for scorer in args.relation_scorers:
        scorer_index = -1
        if scorer in RELATION_SCORERS:
            scorer_index = RELATION_SCORERS.index(scorer)
        if scorer_index == 0:  # entity neighbourhood scoring
            scorers.append(EntityNeighbourhoodScorer(subject_encoder))
            scorers.append(EntityNeighbourhoodScorer(object_encoder))
            initial_scorer_weights.extend((None, None))
        elif scorer_index == 1:  # query attention scoring
            so_query_embeddings = build_relation_encoder(args.embed_dim_pairs)
            scorers.append(QueryRelationScorer(train_data, so_query_embeddings, args.max_so_predicates))
            initial_scorer_weights.append(None)
        elif scorer_index == 2:  # dual attention scoring
            so_dual_embeddings = build_relation_encoder(args.embed_dim_pairs)
            scorers.append(DualAttentionRelationScorer(train_data, so_dual_embeddings, subject_encoder, object_encoder,
                                                       args.max_so_predicates))
            initial_scorer_weights.append(None)
        elif scorer_index == 3:  # e-model
            if args.text_encodings is not None and any(enc[0] != PREDICATE_WORD_EMBEDS for enc in args.text_encodings):
                logger.warning("entity text encoding for e-model could easily be but isn't implemented!")
            subject_e_relation_encoder = build_relation_encoder(args.embed_dim, s_or_o=0)
            object_e_relation_encoder = build_relation_encoder(args.embed_dim, s_or_o=1)
            subject_e_entity_enc_aggregator, encoder_key = build_text_encoding_aggregator(ENTITY_WORD_EMBEDS,
                                                                                          args.embed_dim)
            object_e_entity_enc_aggregator, encoder_key = build_text_encoding_aggregator(ENTITY_WORD_EMBEDS,
                                                                                         args.embed_dim)
            subject_e_encoder = EModelEntityEncoder(train_data, "subject", embed_dim=args.embed_dim,
                                                    relation_encoder=subject_e_relation_encoder,
                                                    entity_encoder=text_encoders[encoder_key],
                                                    entity_encoding_aggregator=subject_e_entity_enc_aggregator)
            object_e_encoder = EModelEntityEncoder(train_data, "object", embed_dim=args.embed_dim,
                                                   relation_encoder=object_e_relation_encoder,
                                                   entity_encoder=text_encoders[encoder_key],
                                                   entity_encoding_aggregator=object_e_entity_enc_aggregator)
            scorers.append(EntityNeighbourhoodScorer(subject_e_encoder))
            scorers.append(EntityNeighbourhoodScorer(object_e_encoder))
            initial_scorer_weights.extend((None, None))  # this is redundnat, but matches the pattern elsewhere
            unweighted_scorers = True
        else:
            raise NotImplementedError(f"relation scoring with {scorer} has not been implemented! Use one of "
                                      f"{RELATION_SCORERS}")
    # weighted sum of sigmoids of scores
    main_scorer = MultipleRelationScorer(*scorers, weight_scores=not unweighted_scorers,
                                         normalise_weights=args.normalise_weights,
                                         leaky_relu=args.score_weights_leaky_relu,
                                         initial_weights=initial_scorer_weights,
                                         data_loader=train_data)
    multi_gpu = False
    if not args.no_cuda and not args.single_gpu and torch.cuda.device_count() > 1:
        main_scorer = nn.DataParallel(main_scorer)
        multi_gpu = True

    if args.load_model is not None:
        # Load the main model
        state = torch.load(output_folder / model_file_name)

        def state_to_main_key(state_key):
            # for loading single gpu models into multi-gpu run and vice versa
            if multi_gpu and not multi_gpu_state:
                return "module." + state_key
            elif not multi_gpu and multi_gpu_state:
                return state_key[len("module."):]
            else:
                return state_key

        if 'static_state_dict_entries' in state:  # always true for recent models, here for backward compatibility
            multi_gpu_state = next(iter(state['state_dict'].keys())).startswith("module.")

            main_scorer_sd = main_scorer.state_dict()
            for k in state['static_state_dict_entries']:
                # these are static word embeds, and should already have been loaded
                state['state_dict'][k] = main_scorer_sd[state_to_main_key(k)]

            # check if all scorers are present: copy any new ones to the state dict from main_scorer
            # we assume new ones are appended
            scorer_sd_keys = [set(scorer.state_dict().keys()) for scorer in scorers]
            state_sd_keys = defaultdict(set)
            sd_prefix = None
            for sd_key in state['state_dict'].keys():
                match = STATE_DICT_SCORER_KEY_RE.match(sd_key)
                if match is not None:
                    prefix, sc_idx, sc_module = match.groups()
                    if sd_prefix is None:
                        sd_prefix = prefix
                    elif prefix != sd_prefix:
                        logger.warning(f"multiple state dict prefixes! {sd_prefix} and {prefix}")
                    state_sd_keys[int(sc_idx)].add(sc_module)
            for i, scorer in enumerate(scorers):
                scorer_sd = scorer.state_dict()
                sd_keys = set(scorer_sd.keys())
                if i < len(state_sd_keys):
                    assert sd_keys == state_sd_keys[i], f"scorer mismatch at index {i}: {sd_keys} doesn't match " \
                                                        f"{state_sd_keys[i]}"
                else:
                    for key in sd_keys:
                        state["state_dict"][f"{sd_prefix}scorers.{i}.{key}"] = scorer_sd[key]
                    mix_params = main_scorer.module.scorer_param_lists if sd_prefix else main_scorer.scorer_param_lists
                    for param in mix_params:
                        new_param = torch.empty_like(main_scorer_sd[param])
                        new_param[:len(state["state_dict"][param])] = state["state_dict"][param]
                        new_param[i] = main_scorer_sd[param][i]
                        state["state_dict"][param] = new_param
            del main_scorer_sd

            state_dict = update_state_dict_names(state['state_dict'])
            state_dict = {state_to_main_key(k): v for k, v in state_dict.items()}
            main_scorer.load_state_dict(state_dict)  # (state['state_dict'])  #
        else:  # for backward compatibility...
            multi_gpu_state = next(iter(state.keys())).startswith("module.")
            state = {state_to_main_key(k): v for k, v in state.items()}
            main_scorer.load_state_dict(update_state_dict_names(state))  # (state)  #
        del state
        logger.info(f"loaded model file from {output_folder / model_file_name}")

    if not args.no_cuda:
        main_scorer.cuda()

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    tensorboard = SummaryWriter(log_dir=tensorboard_dir)
    logger.info(f"Setup tensorboard writer to {tensorboard_dir}")
    save_args("after setting up model")

    epoch = None

    if args.train and check_we_can_train():
        eval_times, train_times, epoch_times = [], [], []
        max_eval_time, max_train_time, max_epoch_time = 0., 0., 0.
        if getattr(args, "max_eval_time", None) is not None:
            max_eval_time = args.max_eval_time
        model_saved = False
        burn_in_epochs = 2

        def format_hours(hours: float):
            return f"{math.floor(hours)}h {(hours % 1) * 60:.1f}m ({hours:.3f} hours)"

        def save_model(extra_label, save_epoch=None, map_score=None, map_name="MAP", log_label="",
                       save_labelled_args=False):
            nonlocal model_saved
            if map_score is not None:
                extra_label += f"_{map_name}_{map_score}_at_{save_epoch}"
            file_name = args.file_name_base + f"model_{extra_label}.pt"
            opt_file_name = args.file_name_base + f"optimiser_{extra_label}.pt"
            # TODO: remove entity word embeds (and relation text embeds) from state dict --- also needed when loading..
            state_ = main_scorer.state_dict()
            static_state_dict_entries = [key_ for key_ in state_.keys() if STATIC_STATE_RE.match(key_)]
            for key_ in static_state_dict_entries:
                del state_[key_]
            torch.save({"state_dict": state_, "static_state_dict_entries": static_state_dict_entries},
                       output_folder / file_name)
            del state_

            # torch.save(main_scorer.state_dict(), output_folder / file_name)
            opt_state_ = optimizer.state_dict()
            # TODO: save lr scheduler if we do it in a more flexible/complex way
            # if lr_scheduler is None:
            #     opt_state_ = {"optimizer": opt_state_, "scheduler": lr_scheduler.state_dict()}
            torch.save(opt_state_, output_folder / opt_file_name)
            del opt_state_

            if extra_label == FINAL_MODEL_LABEL:
                model_saved = True
            if save_labelled_args:
                save_args(f"as {extra_label}", report=True, extra_label=extra_label)
            tensorboard.add_text(f"saved {' '.join(extra_label.split('_')[:2])} model file", file_name,
                                 global_step=save_epoch)
            logger.info(f"model saved by {log_label} at {output_folder / file_name}")

        def evaluate_and_report(report_epoch):
            nonlocal max_eval_time, max_dev_epoch, epochs_since_best
            # if scores is not None:
            #     dev_evaluator.add_scores_and_labels(scores, labels)
            # TODO: report MAP with and without train data...
            before_time = time.time()
            score = dev_evaluator.evaluate(report_epoch)
            best_report = ''
            log_best = False
            if best_models_by_map:
                score_name = "MAP"
                if score > args.max_MAP:
                    max_map = float(score)
                    args.max_MAP_epoch = report_epoch
                    if report_epoch > burn_in_epochs:
                        args.max_MAP = max_map
                    max_dev_epoch = report_epoch
                    epochs_since_best = 0
                    log_best = True
                else:
                    epochs_since_best = report_epoch - max_dev_epoch
            else:
                score_name = "AUC"
                score, pr = score
                if score > args.max_AUC:
                    max_auc = float(score)
                    args.max_AUC_epoch = report_epoch
                    if report_epoch > burn_in_epochs:
                        args.max_AUC = max_auc
                    max_dev_epoch = report_epoch
                    epochs_since_best = 0
                    log_best = True
                else:
                    epochs_since_best = report_epoch - max_dev_epoch
                if pr == DEGENERATE_MODEL_MARKER:  # signifies a degenerate model
                    args.degenerate_epochs.append(report_epoch)
                    degenerate_message = ""
                    if args.first_degenerate_epoch is None:  # not set yet
                        args.first_degenerate_epoch = report_epoch
                        degenerate_message = " first"
                    logger.info(f"...{degenerate_message} degenerate model at epoch {report_epoch}")
                    log_best = True

            eval_time = (time.time() - before_time) / 60. / 60.
            eval_times.append(eval_time)  # hours since start of evaluation
            max_eval_time = max(eval_time, max_eval_time)

            if eval_time == max_eval_time or log_best:
                args.max_eval_time = eval_time
            args.eval_next = False
            save_args("after evaluation: eval_next unset, score stuff set")

            if log_best:
                best_report = " BEST"
                logger.info(f"New best model with {score_name} {score} at epoch {report_epoch}! Saving model!")
                save_model(BEST_MODEL_LABEL, report_epoch, map_score=score, map_name=score_name, log_label=best_report,
                           save_labelled_args=True)
                if tensorboard is not None:
                    # value should be one of int, float, str, bool, or torch.Tensor
                    tensorboard.add_scalar(f"{score_name}_BEST", score, global_step=report_epoch)

            if t_epoch is not None:
                t_epoch.set_postfix(**{score_name: f"{score:.6e}{best_report}",
                                       "loss": f"{epoch_loss:.6e}" if epoch_loss is not None else ""})
            if tensorboard is not None:
                if epoch_loss is not None:
                    tensorboard.add_scalar("loss", epoch_loss, global_step=report_epoch)
                tensorboard.add_scalar(score_name, score, global_step=report_epoch)

                # Add scorer mixing parameters to tensorboard
                if getattr(main_scorer, "module", None) is None:
                    root_module = main_scorer
                else:
                    root_module = main_scorer.module  # main scorer is wrapped in DataParallel for multiple gpus
                mix_params_ = {}
                for p_name in ("temperatures", "thresholds", "weights"):
                    parameter = getattr(root_module, p_name, None)
                    if parameter is None:
                        logger.warning(f"Could'nt find mix parameter {p_name} for tensorboard!")
                    else:
                        for i_, val_ in enumerate(parameter):
                            mix_params_[f"{p_name} {i_}"] = val_
                if mix_params_:
                    tensorboard.add_scalars("joint_measure_parameters", mix_params_, global_step=report_epoch)
                    logger.info(f"joint measure params {dict((k_, float(v)) for k_,v in mix_params_.items())} sent to "
                                f"tensorboard")
            else:
                logger.warning("tensorboard is None in evaluate_and_report!?")
            main_scorer.train()
            return f"{score_name} {score}"

        logger.info("setting up evaluator")
        # if getattr(train_data, "top_relations", None) is None:  # for backward compatibility - this should be set
        #     if getattr(args, "num_top_rels", None) is None:  # old version of eval_top_n_rels
        #         if getattr(args, "eval_top_n_rels", None) is None:  # for backward compatibility - this should be set
        #             logger.waring(f"eval_top_n_rels missing from args! {json.dumps(vars(args))}")
        #             args.num_top_rels = 50
        #         else:
        #             args.num_top_rels = args.eval_top_n_rels
        #     logger.warning(f"train model has no top_relations! Setting it with {args.num_top_rels} relations.")
        #     train_data.set_top_relations(args.num_top_rels)
        if args.load_model is None:
            tensorboard.add_text("Job ID", args.jobid, global_step=args.epoch)
            tensorboard.add_text("Label", args.label, global_step=args.epoch)
            tensorboard.add_text("file_name_base", args.file_name_base, global_step=args.epoch)
            # NOTE: args are handled in save_args() via last_tensorboard_args
            # tensorboard.add_text("args", re.sub(r'(^|\n)', r'\1    ', json.dumps(vars(args), indent=4)))

        dev_data = load_eval_data_readers('dev', "Validation", (args.eval_with[0],))
        dev_evaluator = load_evaluators(dev_data, "Validation")[args.eval_with[0]]

        del dev_data

        opt_params = OPTIMIZER_PARAMS[args.optimizer]
        if args.opt_params is not None:
            for opt, val in map(lambda s: s.split(':'), args.opt_params.split(',')):
                try:
                    val = ast.literal_eval(val)
                except SyntaxError:
                    pass
                opt_params[opt] = val
        if opt_params:
            logger.info(f"Creating {args.optimizer} optimiser with params {opt_params}")
        # TODO: setup a filter for low lr parameters and make another optimizer for it (or is there a better way?)
        #       https://pytorch.org/docs/1.4.0/optim.html#per-parameter-options --- provide a list of kwargs dicts
        optimizer = OPTIMIZERS[args.optimizer](filter(lambda p: p.requires_grad, main_scorer.parameters()), lr=args.lr,
                                               **opt_params)

        # TODO: for delayed triple scoring, setup second optimizer and be sure triple scoring isn't in this one...
        if args.load_model is not None:
            # TODO: for so_ene_tensor_scorer loaded later, we may have two optimizers...
            if args.run_to_load:
                optimizer_file_name = output_folder / f"{args.load_model}optimiser_{args.run_to_load}.pt"
            else:  # for backward compatibility with no "run-to-load" and no "_" after "model"
                optimizer_file_name = f"{args.load_model}optimiser.pt"
            try:
                opt_state_dict = torch.load(optimizer_file_name)
                if set(opt_state_dict.keys()) == {"optimizer", "scheduler"}:
                    optimizer.load_state_dict(opt_state_dict["optimizer"])
                    logger.warning("Scheduler saved in optimiser file, but we build one anew anyway!")
                else:
                    optimizer.load_state_dict(opt_state_dict)
                logger.info(f"loaded optimizer state from {optimizer_file_name}")
            except FileNotFoundError:
                logger.warning(f"optimizer file not found! Creating new optimizer! {optimizer_file_name}")
            except OSError as e:
                logger.warning(f"optimizer file OSError, creating new optimizer! {e}")
            except ValueError:
                logger.critical(f"optimizer param mismatch with loaded state_dict! Resetting optimizer!")

        epoch = int(args.epoch)
        lr_scheduler = None
        if (args.lr_final or args.batch_size_final) and args.schedule_to_epoch is None:
            args.schedule_to_epoch = args.epochs
        if args.lr_final is not None:
            initial_epoch = 0  # superfluous variable to help understand what's going on in equation below

            def lr_lambda(this_epoch):
                return 1. + (min(this_epoch, args.schedule_to_epoch) - initial_epoch) \
                       / (args.schedule_to_epoch - initial_epoch) \
                       * (args.lr_final/args.lr - 1)
            # noinspection PyTypeChecker
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            # TODO: save lr scheduler if we do it in a more flexible/complex way

        save_model(FINAL_MODEL_LABEL, epoch, log_label="before inference")

        loss_function = PairwiseRankingLoss(args.loss_margin)
        if not args.no_cuda:
            loss_function.cuda()

        partial_epoch = None
        try:
            elapsed_time = (time.time() - start_time) / 60. / 60.  # hours since start of execution
            logger.info(f"starting training loop on {args.device} after {format_hours(elapsed_time)}...")
            epoch_loss = args.last_epoch_loss  # None for first epoch, last recorded epoch loss for loaded models
            t_epoch = None  # a dummy value to avoid referencing before assignment
            store_hparams = False
            args_max_train_time = getattr(args, "max_train_time", None)
            if args_max_train_time is not None:
                max_train_time = args_max_train_time
            # TODO: we had ` or args_max_train_time is None` for running initial eval. Why??
            if args.eval_next and not getattr(args, "skip_initial_eval", False):
                # eval is up next or no epochs completed yet
                score_string = evaluate_and_report(epoch)
            else:
                skip_msg = ' and '.join(s for s, flag in (('eval_next', args.eval_next),
                                                          ('max train time is None', args_max_train_time is None),
                                                          ('skip initial eval option', args.skip_initial_eval))
                                        if flag)
                args.skip_initial_eval = False
                logger.info(f"skipping initial evaluation with {skip_msg}")
                score_string = "eval skipped"
            main_scorer.train()
            elapsed_time = (time.time() - start_time) / 60. / 60.  # hours since start of execution
            if args.stop_on_no_best_for is not None and epochs_since_best >= args.stop_on_no_best_for:
                save_args(f"{epochs_since_best} epochs no dev improvement "
                          f"(more than {args.stop_on_no_best_for})")
                logger.info(f"quitting before epoch due to {epochs_since_best} epochs no dev improvement "
                            f"(more than {args.stop_on_no_best_for})")
                store_hparams = True
            elif args.max_inference_hours is not None and \
                    elapsed_time + max_train_time + 2/60 > args.max_inference_hours and \
                    len(eval_times) > 0:
                args.eval_next = False  # to be sure we go straight to train next time
                save_args("quitting after long initial eval, eval_next unset")
                logger.info(f"quitting before epoch due to long initial execution (dev evaluation?), "
                            f"elapsed time {format_hours(elapsed_time)} + previous max train time "
                            f"{format_hours(max_train_time)} + {format_hours(2./60)}")
                store_hparams = False
            else:
                with tqdm(range(epoch+1, args.epochs+1), desc=f"Epoch") as t_epoch:
                    logger.info(f"starting from epoch {epoch} with loss {epoch_loss} and {score_string}")
                    for epoch in t_epoch:
                        # if not do_training:
                        #     break  # terrible hack to avoid indenting the training loop (with icky git ramifications!)
                        epoch_start_time = time.time()
                        logger.info(f"Epoch {epoch} after {format_hours(elapsed_time)}")

                        # TODO: Bracewell doesn't seem to have magma installed, which seems necessary for the norms...
                        #       https://github.com/pytorch/pytorch#install-dependencies
                        #       Add LAPACK support for the GPU if needed
                        #       conda install -c pytorch magma-cuda102  #
                        #       or [ magma-cuda101 | magma-cuda100 | magma-cuda92 ] depending on your cuda version
                        # for te_name, te in text_encoders.items():
                        #     if te is not None and te.FC is not None:
                        #         logger.info(f"reporting text encoder FC norms for {te_name}")
                        #         tensorboard.add_scalar(f"{te_name}_FC_matrix_norm", te.FC.weight.norm(),
                        #                                global_step=epoch)
                        #         tensorboard.add_scalar(f"{te_name}_FC_nuclear_norm", te.FC.weight.norm(p='nuc'),
                        #                                global_step=epoch)
                        #         tensorboard.add_scalar(f"{te_name}_bias_norm", te.FC.bias.norm(),
                        #                                global_step=epoch)
                        #     te_ag = text_encoder_aggregators.get(te_name, None)
                        #     if te_ag is not None and te_ag.FC is not None:
                        #         logger.info(f"reporting text encoder aggregator FC norms for {te_name}")
                        #         tensorboard.add_scalar(f"{te_name}_aggregator_FC_matrix_norm", te_ag.FC.weight.norm(),
                        #                                global_step=epoch)
                        #         tensorboard.add_scalar(f"{te_name}_aggregator_FC_nuclear_norm",
                        #                                te_ag.FC.weight.norm(p='nuc'), global_step=epoch)
                        #         tensorboard.add_scalar(f"{te_name}_aggregator_bias_norm", te_ag.FC.bias.norm(),
                        #                                global_step=epoch)
                        # if sum((te is not None and
                        #         (te.FC is not None or text_encoder_aggregators.get(te_name, None) is not None))
                        #        for te_name, te in text_encoders.items()) == 0:
                        #     logger.info(f"no text encoders to report norms for out of "
                        #                 f"{sum((te is not None) for te in text_encoders.values())}")

                        epoch_loss = 0
                        # we use epoch - 1 to get results of previous epoch (and before initial epoch)
                        model_saved = False
                        # TODO: setup 2nd optimizer here for so_ene_tensor_scorer starting later. we also need to save
                        #       it in save model, zero its grad... Maybe create a 2-opyimizer class and switch to that..
                        #       Dont forget to run only optim2 on tensor_scorer for a few epochs.
                        # TODO: for scheduled batch size, 2 options:
                        #       - use an lr scheduler (eg: 0.05 to 0.0005 over 100 epochs) and batch size 128 - 4096
                        #       - make my own eg: lambda epoch: init_bs + max_epoch/epoch*(max_bs - init_bs)

                        batch_size = args.batch_size
                        # if lr_scheduler is not None and args.batch_size_by_lr_schedule:
                        #     current_lr = lr_scheduler.get_last_lr()[0]
                        #     if args.batch_size_final is None:
                        #         batch_size = int(args.batch_size * (args.lr / current_lr))
                        #     else:
                        #         batch_size = int(args.batch_size_final * (args.lr_final / current_lr))
                        #     logger.info(f"Batch size set to {batch_size} with learning rate {current_lr}")
                        # elif ...
                        if args.batch_size_final is not None:
                            batch_size = math.floor(args.batch_size + (args.batch_size_final - args.batch_size) \
                                         * (min(epoch, args.schedule_to_epoch) - 1) \
                                         / (args.schedule_to_epoch - 1))  # these zeros are the initial scheduling epoch
                            logger.info(f"Batch size set to {batch_size} with constant lr {args.lr}")

                        logger.info("about to run batches...")
                        last_batch_loss = None
                        args.eval_next = False
                        # TODO: Yufei: pre-compute all batches before epoch loop!
                        #       More CPUs??
                        #       Need to deal with GPU/CPU sync for neighbour lists...
                        # TODO: For batch size scheduling, we need to adjust args.batch_size here...
                        #       Ideally, we might use the rule of thumb batch / lr ~= const ("don't decaty the lr...")
                        #       My hypothesis is that 1-cycle lr where const is adjusted is ideal
                        with tqdm(OpenKIMemoryIndexedTrainDataLoader(
                                train_data, batch_size=batch_size, shuffle=True,
                                negative_sample_rate=args.negative_rate,
                                negative_strategy=args.negative_strategy), desc="Data") as t_batch:
                            for batch_num, (positive_triples, negative_triple_lists, positive_indices) \
                                    in enumerate(t_batch):
                                partial_epoch = (batch_num + 1) / len(t_batch)

                                if not args.no_cuda:
                                    # TODO: if embeds are not on GPU, we want both cuda and cpu versions of indices!
                                    #       (currently gpu indices are passed back to cpu)
                                    positive_triples = positive_triples.cuda()
                                    negative_triple_lists = negative_triple_lists.cuda()
                                main_scorer.zero_grad()
                                positive_scores = main_scorer(positive_triples.unsqueeze(1), (batch_num, "positive"))
                                # positive_triples.unsqueeze(1) : add a dummy dim to be similar to negative triple lists
                                negative_scores = main_scorer(negative_triple_lists, (batch_num, "negative"))
                                loss = loss_function(positive_scores.expand_as(negative_scores), negative_scores)

                                # TODO: this probably blocks on gpu...
                                if args.check_nans and torch.any(torch.isnan(loss)):
                                    raise ValueError(f"Nan loss {loss} at epoch {epoch}!! Quitting!!")

                                loss.backward()
                                if args.check_nans and any(torch.any(torch.isnan(p.grad))
                                                           for p in main_scorer.parameters() if p.grad is not None):
                                    raise ValueError(f"Nan grad in a parameter at epoch {epoch}!! Quitting!!")

                                optimizer.step()

                                if last_batch_loss is not None:
                                    # get loss from previous batch to hopefully allow batch selection etc.. to be
                                    # synchronous with gpu operations... didn't seem to help though...
                                    epoch_loss += last_batch_loss.item()
                                last_batch_loss = loss

                                # tensorboard.add_histogram("subject_embedding_gradients_positive",
                                #                           subject_embeddings.weight.grad[positive_triples[:, 2]],
                                #                           global_step=epoch)
                                # tensorboard.add_histogram("object_embedding_gradients_positive",
                                #                           object_embeddings.weight.grad[positive_triples[:, 2]],
                                #                           global_step=epoch)
                                # # tensorboard.add_histogram("so_embedding_gradients_positive",
                                # #                           so_embeddings.weight.grad[positive_triples[:, 2]],
                                # #                           global_step=epoch)
                                # tensorboard.add_histogram("subject_embedding_gradients_negative",
                                #                        subject_embeddings.weight.grad[negative_triple_lists[:, :, 2]],
                                #                           global_step=epoch)
                                # tensorboard.add_histogram("object_embedding_gradients_negative",
                                #                         object_embeddings.weight.grad[negative_triple_lists[:, :, 2]],
                                #                           global_step=epoch)
                                # # tensorboard.add_histogram("so_embedding_gradients_negative",
                                # #                           so_embeddings.weight.grad[negative_triple_lists[:, :, 2]],
                                # #                           global_step=epoch)

                                elapsed_time = (time.time() - start_time) / 60. / 60.  # hours since program start
                                if args.max_inference_hours is not None \
                                        and elapsed_time + 5 / 60 > args.max_inference_hours:
                                    train_time = (time.time() - epoch_start_time) / 60. / 60.  # in hours
                                    prev_partial_epoch = getattr(args, "partial_epoch", 0.)
                                    logger.info(f"Epoch {epoch  - 1 + prev_partial_epoch:.2f} batch {batch_num} of "
                                                f"{len(t_batch)} ({partial_epoch*100:.1f}%) quitting with partial loss "
                                                f"{epoch_loss}, training for {format_hours(train_time)} after elapsed "
                                                f"time {format_hours(elapsed_time)} + {format_hours(5/60)}")
                                    break
                                partial_epoch = None
                        # EPOCH COMPLETED
                        if lr_scheduler is not None:
                            # noinspection PyArgumentList
                            lr_scheduler.step()
                            # noinspection PyUnresolvedReferences
                            logger.info(f"lr step to {lr_scheduler.get_last_lr()[0]}")
                        epoch_loss += last_batch_loss.item()  # catch up on the final batch loss
                        logger.info("batch loop done: saving model and updated args")
                        save_model(FINAL_MODEL_LABEL, epoch, log_label=f"epoch {epoch} end")
                        model_saved = True
                        args.epoch = epoch
                        # epoch_loss /= args.num_relations  # do we need to scale the batch loss??

                        args.last_epoch_loss = epoch_loss
                        elapsed_time = (time.time() - start_time) / 60. / 60.  # hours since program start
                        train_time = (time.time() - epoch_start_time) / 60. / 60.  # hours since start of epoch
                        max_train_time = max(max_train_time, train_time)
                        args.max_train_time = max_train_time

                        if partial_epoch is None:  # this means the batch loop exited normally
                            args.eval_next = True
                            save_args("after batches completed (eval_next set)")
                        else:  # this means the batch loop was broken by exception or break stmt
                            args.last_epoch_loss /= partial_epoch
                            save_args("in epoch completed: reset last_epoch_loss with partial epoch")
                            store_hparams = False
                            break

                        train_times.append(train_time)
                        if args.max_inference_hours is not None \
                                and elapsed_time + max_eval_time + 2/60 > args.max_inference_hours:
                            logger.info(f"no time left for epoch eval after {train_time} in training this epoch.")
                            logger.info(f"epoch {epoch} done with loss {epoch_loss} --- ending after elapsed time "
                                        f"{format_hours(elapsed_time)} + max eval time {format_hours(max_eval_time)} "
                                        f"+ {format_hours(2./60)}")
                            store_hparams = False
                            break
                        score_string = evaluate_and_report(epoch)
                        tensorboard.flush()
                        # recalculate elapsed and epoch times after evaluation
                        elapsed_time = (time.time() - start_time) / 60. / 60.  # hours since start of inference
                        epoch_time = (time.time() - epoch_start_time) / 60. / 60.  # hours since start of epoch
                        max_epoch_time = max(max_epoch_time, epoch_time)
                        epoch_times.append(epoch_time)
                        logger.info(f"epoch {epoch} done with loss {epoch_loss} and {score_string} "
                                    f"after {format_hours(elapsed_time)}")
                        if args.stop_on_no_best_for is not None and epochs_since_best >= args.stop_on_no_best_for \
                                and not args.eval_next:
                            logger.warning(f"--- ending after no dev improvement for {epochs_since_best}")
                            store_hparams = True
                            break
                        if args.max_inference_hours is not None \
                                and elapsed_time + max_train_time + 5/60 > args.max_inference_hours:
                            logger.info(f"--- ending after elapsed time {format_hours(elapsed_time)} + max train time "
                                        f"{format_hours(max_train_time)} + {format_hours(2./60)}")
                            store_hparams = False
                            break
                        store_hparams = True

        finally:
            if partial_epoch is not None:
                if not hasattr(args, "partial_epoch"):
                    args.partial_epoch = 0.
                args.partial_epoch += partial_epoch
                if args.partial_epoch >= 1:
                    args.partial_epoch -= 1
                    args.eval_next = True
                    logger.info(f"partial epochs {args.partial_epoch:.2f} made up an epoch, evaluating next!")
                else:
                    epoch -= 1  # because it'll be stored in args.epoch, which is for completed
                    #             epochs, whereas this epoch isn't completed yet...
                    args.eval_next = False  # well, we should wait till a whole epoch is done...
                    logger.info(f"partial epochs {args.partial_epoch:.2f} less than an epoch, not evaluating next!")
                logger.info(f"epoch {epoch} done with loss {epoch_loss} --- partial epoch "
                            f"after {format_hours(elapsed_time)}")
                args.epoch = epoch
                save_args(f"partial epoch in finally: set partial_epoch, epoch and eval_next")
            if not model_saved:
                logger.info("saving model in finally clause...")
                save_model(FINAL_MODEL_LABEL, epoch, log_label="finally clause")
                # probably not necessary, but just to be safe (and if args.epoch is changed above)
                save_args(f"in finally just in case...")
            if args.dont_log_exceptions:
                if not sys.exc_info() == (None, None, None):
                    raise
                else:
                    print(f"sys.exc_info() is {sys.exc_info()}, seems there was no exception...")
                logger.info("finishing without saving model nor calculating last MAP value!")
            else:
                logger.info(f"training all done with {args.file_name_base} after {format_hours(elapsed_time)}")
                # score_string = evaluate_and_report(epoch)  # final results
                # logger.info(f"final evaluation: {score_string}")
                tensorboard.flush()
                if epoch >= args.epochs and not args.eval_next or args.force_store_hparams or store_hparams:
                    logger.info(f"Storing hprams to tensorboard at epoch {epoch} of {args.epochs}")
                    hparams = {}
                    for k, v in vars(args).items():
                        if k in ("max_MAP", "dont_log_exceptions", "train", "validate", "test", "map_on_train"):
                            continue
                        if type(v) in (int, float, str, bool, torch.Tensor):
                            hparams[k] = v
                        else:
                            hparams[k] = str(v)
                    if best_models_by_map:
                        tensorboard.add_hparams(hparams, {"MAP_BEST": args.max_MAP})
                    else:
                        tensorboard.add_hparams(hparams, {"AUC_BEST": args.max_AUC})
                else:
                    logger.info(f"Not storing hprams to tensorboard at epoch {epoch} of {args.epochs} and "
                                f"{'' if args.eval_next else 'not'} evaluating next.")
            # elapsed_time = (time.time() - start_time) / 60 / 60  # hours since start of inference
            # logger.info(f"training all done with {args.file_name_base} after {format_hours(elapsed_time)}")
            if len(epoch_times):
                logger.info(f"epoch times: min {format_hours(min(epoch_times))}, max {format_hours(max(epoch_times))}, "
                            f"mean {format_hours(statistics.mean(epoch_times))}, "
                            f"median {format_hours(statistics.median(epoch_times))}, "
                            f"stdev {format_hours(statistics.stdev(epoch_times)) if len(epoch_times) > 1 else 0}")
            if len(train_times):
                logger.info(f"train times: min {format_hours(min(train_times))}, max {format_hours(max(train_times))}, "
                            f"mean {format_hours(statistics.mean(train_times))}, "
                            f"median {format_hours(statistics.median(train_times))}, "
                            f"stdev {format_hours(statistics.stdev(train_times)) if len(train_times) > 1 else 0}")
            if len(eval_times):
                logger.info(f"eval  times: min {format_hours(min(eval_times))}, max {format_hours(max_eval_time)}, "
                            f"mean {format_hours(statistics.mean(eval_times))}, "
                            f"median {format_hours(statistics.median(eval_times))}, "
                            f"stdev {format_hours(statistics.stdev(eval_times)) if len(eval_times) > 1 else 0}")

    # VALIDATE AND TEST ONLY OPTIONS
    def evaluate_on(split, message):
        logger.info(f"Starting {message} final evaluation")
        data_readers = load_eval_data_readers(split, message)
        test_evaluators = load_evaluators(data_readers, message)
        do_evaluation(test_evaluators, message)

    if epoch is None:
        epoch = args.epoch
    if args.test:
        evaluate_on("test", "Test")

    if args.validate:
        if not args.train:
            evaluate_on("dev", "Validation")
        else:
            logger.warning("Evaluation already done during training - ignoring --evaluate!")

    if args.map_on_train:
        evaluate_on("train", "Train")


if __name__ == '__main__':
    def process_args():
        parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

        # GENERAL ARGUMENTS
        group = parser.add_argument_group("General arguments", "Arguments to control general behaviour.")
        group.add_argument("--jobid", default=None, 
                           help="Job specific id for this run. Current time used if not provided.")
        group.add_argument("--max-inference-hours", default=None, type=float,
                           help="Inference exits after this many hours (default None for no limit).")
        group.add_argument("--label", default="un-labelled", 
                           help="Label for this configuration (shared by other jobs).")
        group.add_argument("--output-folder", default="output",
                           help="Folder in which to store model and log files (default 'output')")
        group.add_argument("--train", action="store_true", help="whether to train the model.")
        group.add_argument("--no-cuda", action="store_true",  help="to run on CPU only")
        group.add_argument("--single-gpu", action="store_true", help="restrict to one gpu (else use all available. "
                                                                     "--no-cuda overrides this setting.")
        group.add_argument("--detect-anomaly", action="store_true",
                           help="Run with `torch.autograd.set_detect_anomaly(True)` (also checks for nans)")
        group.add_argument("--check-nans", action="store_true",
                           help="Check for nan scores and gradients each batch.")
        group.add_argument("--load-model", default=None,
                           help="File name {base} to load. Pytorch model file {base}model_{final|BEST...}.pt and "
                                "program arguments from {base}args.txt. " f"Any other '{MODEL_ARGS_GROUP}' arguments "
                                f"except --run-to-load are ignored (unless included in --force-default-args).")
        group.add_argument("--run-to-load", default=FINAL_MODEL_LABEL,
                           help="Load model file '{base}model_{RUN_TO_LOAD}.pt' instead of '{base}model_" 
                                f"{FINAL_MODEL_LABEL}.pt")
        group.add_argument("--load-model-newname-regex", default=None,
                           help="regex whose match is used to construct a new output file name. The first"
                                "and last groups are concatenated and the jobid appended.")
        group.add_argument("--force-default-args", nargs='*', default=[],
                           help="When loading a model, options listed here will be reset to default values. 'train', "
                                "'test', 'validate', and 'print_args_only' are always forced.")
        group.add_argument("--print-args-only", action="store_true", help="Print resolved program arguments and exit.")
        group.add_argument("--dont-log-exceptions", action="store_true", 
                           help="don't send uncaught exceptions  to the logger")
        group.add_argument("--force-store-hparams", action="store_true",
                           help="Store hparams to tensorboard, even if we end before --epochs are completed.")
        # group.add_argument("--random-seed", default=None, type=int,
        #                    help="Random seed for pytorch and numpy (default None selects the seed randomly)")

        # MODEL STRUCTURE AND DATA SOURCE
        # DATA AND IT'S VARIANTS
        group = parser.add_argument_group(MODEL_ARGS_GROUP, "Options for model structure and data source.")
        group.add_argument("--data-folder", default=None,
                           help="Folder containing training, dev and test data.")
        group.add_argument("--data-source", default=DATA_SOURCES[0], choices=DATA_SOURCES,  # default is reverb
                           help=f"Data source ({' or '.join(DATA_SOURCES)}: reverb data supplied by OpenKI authors or"
                                f" Riedel 2010 nyt data processed via generate_openki.py.")
        group.add_argument("--data-variants", default=[], nargs="*", choices=DATA_VARIANTS,
                           help=f"Variants passed to the data reader (reverb data accepts {'or'.join(DATA_VARIANTS)})")

        # THE BASE MODEL
        group.add_argument("--entity-nbhood-scoring", dest="relation_scorers", action="append_const",
                           const=RELATION_SCORERS[0], help=f"Use entity neighbourhood scoring.")
        group.add_argument("--entity-pair-query-scoring", dest="relation_scorers", action="append_const",
                           const=RELATION_SCORERS[1], help=f"Use relations between same entity pairs scoring.")
        group.add_argument("--entity-pair-dual-scoring", dest="relation_scorers", action="append_const",
                           const=RELATION_SCORERS[2], help=f"Use relations between same entity pairs scoring.")
        group.add_argument("--e-model-scoring", dest="relation_scorers", action="append_const",
                           const=RELATION_SCORERS[3], help="Use E-model scoring (Universal schema [Riedel 2013])")
        group.add_argument("--sqrt-normed-neighbour-ag", action="store_true",
                           help="Divide by sqrt(n) and scale by sqrt(dim) when aggregating neighbours (results in unit "
                                "expected norm of average).")

        # ADDING TEXT ENCODINGS
        group.add_argument("--entity-word-embeds-fc-tanh-sum", dest="text_encodings", action="append_const",
                           const=f"{ENTITY_WORD_EMBEDS},{FC_TANH_THEN_SUM}",
                           help=f"Sum averaged entity word embeds (passed through an FC layer) "
                                f"with aggregated vectors.")
        group.add_argument("--entity-word-embeds-concat-fc-tanh", dest="text_encodings", action="append_const",
                           const=f"{ENTITY_WORD_EMBEDS},{CONCAT_THEN_FC_TANH}",
                           help=f"Concat averaged entity word embeds with aggregated vectors, then pass through an "
                                f"FC layer with tanh. This is NOT recommended, it failed abysmally!")
        group.add_argument("--entity-word-embeds-concat-fc-relu", dest="text_encodings", action="append_const",
                           const=f"{ENTITY_WORD_EMBEDS},{CONCAT_THEN_FC_RELU}",
                           help=f"Concat averaged entity word embeds with aggregated vectors, then pass through an "
                                f"FC layer with relu.")
        group.add_argument("--predicate-word-embeds-concat-fc-tanh", dest="text_encodings", action="append_const",
                           const=f"{PREDICATE_WORD_EMBEDS},{CONCAT_THEN_FC_TANH}",
                           help=f"Concat averaged predicate word embeds with learned predicate vectors, then pass "
                                f"through an FC layer with tanh.")
        group.add_argument("--predicate-word-embeds-fc-tanh-sum", dest="text_encodings", action="append_const",
                           const=f"{PREDICATE_WORD_EMBEDS},{FC_TANH_THEN_SUM}",
                           help=f"Sum averaged predicate word embeds (passed through an FC layer) with learned "
                                f"predicate vectors.")
        group.add_argument("--word-embeddings", choices=WORD_EMBEDDINGS, default=NO_EMBEDDING,
                           help=f"Type of word embeddings:  {', '.join(WORD_EMBEDDINGS)}; default {NO_EMBEDDING}.")
        group.add_argument("--word-embed-file", default=None,
                           help="Path to file containing pretrained word embeddings.")
        group.add_argument("--layer-norm-on-fc", action="store_true",
                           help="Apply layer normalisation on the FC projection layer from entity/text embeddings.")
        group.add_argument("--no-activation-on-fc", action="store_true",
                           help="No activation function on FC projection layer from entity/text embeddings.")
        group.add_argument("--pred-static-embeds", dest="text_embeds_static", action="append_const",
                           const=PREDICATE_WORD_EMBEDS)
        group.add_argument("--entity-static-embeds", dest="text_embeds_static", action="append_const",
                           const=ENTITY_WORD_EMBEDS)
        group.add_argument("--pred-static-embeds-on-cpu", dest="embeds_on_cpu", action="append_const",
                           const=PREDICATE_WORD_EMBEDS)
        group.add_argument("--entity-static-embeds-on-cpu", dest="embeds_on_cpu", action="append_const",
                           const=ENTITY_WORD_EMBEDS)
        group.add_argument("--bert-pipeline", default="feature-extraction", help="name of huggingface pipeline")
        group.add_argument("--fine-tune-bert", action="store_true", help="fine_tune BERT (default: don't fine_tune")
        group.add_argument("--bert-use-cls", action="store_true", help="use only the BERT CLS token representation")
        group.add_argument("--predicate-text-flag", action="store_true",
                           help="Text encoder receives a single added dimension containing 0 for KB relations, 1 for "
                                "text predicates. Entity text encoding is not effected.")
        group.add_argument("--random-embed-dim", default=None, type=int, help="dimension of random predicate text-like "
                                                                              "embeds")

        # MODEL PARAMETERS
        # group.add_argument("--num-relations", default=182657, help="number of predicates and relations (default "
        #                                                            "182658, as used in OpenKI reverb data).")
        group.add_argument("--max-nbhd-predicates", default=16, type=int, 
                           help="maximum number of relations/predicates sampled for subject/object "
                                "neighbourhood during training (all are used during inference).")
        group.add_argument("--max-so-predicates", default=8, type=int, 
                           help="maximum number of relations/predciates sampled for entity pair "
                                "neighbourhood during training (all are used during inference).")
        group.add_argument("--embed-dim", default=12, type=int,
                           help="Embedding dimension for single entity neighbour embeddings.")
        group.add_argument("--embed-dim-pairs", default=24, type=int,
                           help="Embedding dimension for entity pair neighbour embeddings.")
        group.add_argument("--no-learned-relation-embeds", action="store_true",
                           help="Use only word based relation representations, no relation specific embeddings.")
        group.add_argument("--normalise-weights", action="store_true",
                           help="With multiple scorers, constrain the sum of relative weights of scorers.")
        group.add_argument("--score-weights-leaky-relu", action="store_true",
                           help="Use a leaky ReLu for scorer weights (prevents saturation when they go to zero).")

        # TRAINING OPTIONS
        group = parser.add_argument_group("Training Options", "Options to control model training.")
        group.add_argument("--negative-rate", default=16, type=int, help="number of negative samples per positive.")
        group.add_argument("--negative-strategy", default=NEGATIVE_SAMPLE_STRATEGIES[0],
                           help=f"one of {NEGATIVE_SAMPLE_STRATEGIES}.")
        group.add_argument("--loss-margin", default=1.0, type=float, help="margin for pairwise ranking loss.")
        group.add_argument("--optimizer", default="Adam", help=f"optimiser to use, one of {OPTIMIZERS}.")
        group.add_argument("--opt-params", default=None,
                           help=f"Extra parameters to use with the optimiser. Comma separated list of 'opt:val'.")
        group.add_argument("--lr", default=5e-3, type=float, help="(initial) learning rate (default 0.005).")
        group.add_argument("--lr-final", default=None, type=float,
                           help="Set this for linear learning rate schedule from --lr to this value from current "
                                "epoch to --epochs.")  # lr=0.16, bs=4096 has same ratio as .005, 128
        group.add_argument("--batch-size", default=128, type=int,
                           help="(initial) number of positive samples per batch.")
        group.add_argument("--batch-size-final", default=None, type=int,
                           help="Set this for linear batch size schedule from --batch-size to this value from current "
                                "epoch to  --epochs.")
        # group.add_argument("--batch-size-by-lr-schedule", action="store_true",
        #                    help="Set batch size according to learning rate schedlue: batch_size = initial_batch_size "
        #                         "* (initial_learning_rate / current_learning_rate). If --batch-size-final is provided"
        #                         ", it is used as the target maximim and --batch-size is ignored.")
        # Keeping the lr/bs ratio constant does no annielding! NOT desirable...
        group.add_argument("--schedule-to-epoch", default=None, type=int,
                           help="Batch size / lr scheduling until this epoch. Defaults to --epochs.")
        group.add_argument("--epochs", default=None, type=int, help="number of epochs of training.")
        group.add_argument("--stop-on-no-best-for", type=int, default=None,
                           help="Stop after no dev improvement for this many epochs (default: None, no early stopping)")
        group.add_argument("--entity-emb-dropout", default=None, type=float,
                           help="Dropout rate on learned entity representations when combining with word embeddings "
                                "etc... (default 0.5, -1 to remove dropout)")
        group.add_argument("--train-with-dev-preds", action="store_true", 
                           help="Include OpenIE predicates from dev and test data in training triples. Neighbouring "
                                "relations lists and KB predicates are removed first.")
        # group.add_argument("--openIE-as-neg-samples", action="store_true",
        #                    help="Include OpenIE predicates in -ve samples (otherwise use only KB relations).")
        group.add_argument("--openIE-as-pos-samples", action="store_true", 
                           help="Include OpenIE predicates in +ve samples (otherwise use only KB relations).")
        group.add_argument("--loss-by-pair-ranking", action="store_true",
                           help="Neg samples with same relation but different arg pairs.")

        # EVALUATION OPTIONS
        group = parser.add_argument_group("Evaluation Options", "Options to control model evaluation. Default "
                                          "evaluators are map-ranking-pairs for reverb (used in the OpenKI paper), "
                                          "auc-pr for nyt data. The first mentioned is reported during training and "
                                          "best models are saved.")
        group.add_argument("--map-ranking-pairs-eval", dest="eval_with", action="append_const",
                           const=EVALUATOR_NAMES[0],
                           help="Evaluate with MAP by ranking pairs (default for reverb data)")
        group.add_argument("--map-ranking-relations-eval", dest="eval_with", action="append_const",
                           const=EVALUATOR_NAMES[1], help="Evaluate with MAP by ranking relations (overrides default)")
        group.add_argument("--auc-pr-eval", dest="eval_with", action="append_const",
                           const=EVALUATOR_NAMES[2], help="Evaluate with AUC-PR (default for NYT data)")
        group.add_argument("--test", action="store_true", help="Evaluate the model on test data.")
        group.add_argument("--validate", action="store_true",
                           help="Evaluate the model on dev data. (NOTE: this is done anyway with --train!).")
        group.add_argument("--eval-batch-size", default=None, type=int,
                           help="Min batch size for evaluation. A batch size based on training memory requirements will"
                                " be chosen if this is low.")
        group.add_argument("--map-on-train", action="store_true", help="Calculate MAP on train data.")
        group.add_argument("--num-top-rels", default=50, type=int, 
                           help="Use only top (this number) of KB relations in evaluation (default 50, use 10 for "
                                "smaller data).")
        group.add_argument("--eval-unseen-neighbours", action="store_true",
                           help="Include s- and o- neighbour relations not seen in training data (faster but lower "
                                "eval scores).")
        group.add_argument("--skip-initial-eval", action="store_true",
                           help="Do not perform initial evaluation when starting a training run.")

        return parser.parse_args(), parser

    if 'CUDA_LAUNCH_BLOCKING' in os.environ:
        print(f"CUDA_LAUNCH_BLOCKING set to '{os.environ['CUDA_LAUNCH_BLOCKING']}'")
    main_args, parser_ = process_args()

    if main_args.dont_log_exceptions:
        # noinspection PyProtectedMember
        main_loop(main_args, parser_._action_groups)
    else:
        # noinspection PyBroadException
        try:
            # noinspection PyProtectedMember
            main_loop(main_args, parser_._action_groups)
        except Exception:
            logging.exception("Uncaught Exception...")
