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
import re
from pathlib import Path

import torch.optim as optim
import radam
import adabelief_pytorch

# The following need to be declared before the following imports...
DEGENERATE_MODEL_MARKER = -1
IGNORE_OPENKI_EMBEDS = None
NO_TEXT = None  # Marker for when a corresponding text to a relation/predicate is not available

from OpenKI.Evaluation import OpenKIMapEvaluatorRankingPairs, OpenKIMapEvaluatorRankingRelations, OpenKiAucPrEvaluator
from OpenKI.TextEncoders import FastTextRelationEmbedding, FastTextEntityEmbedding, \
    BertEntityTextEmbedding, BertRelationTextEmbedding, FastTextRelationPairEmbedding, BertRelationPairTextEmbedding, \
    RandomEntityTextEmbedding, RandomRelationTextEmbedding
from OpenKI.NYT_2010_Data import OpenKINYTTrainDataReaderNegPairs, \
    OpenKiNYTEvalPerPairDataReader, OpenKINYTEvalPerRelationDataReader, OpenKINYTTrainDataReader
from OpenKI.Reverb_Data import OpenKIReverbTrainDataReader, OpenKIReverbTrainDataReaderNegPairs, \
    OpenKiReverbEvalPerPairDataReader, OpenKIReverbEvalPerRelationDataReader
from OpenKI.RelationScorers import attention_aggregator, max_pool_weighted_aggregator, weighted_average_aggregator

TRAIN_DATA_HANDLERS = {  # [ by_pair_ranking F, T ]
    "reverb": [OpenKIReverbTrainDataReader, OpenKIReverbTrainDataReaderNegPairs],
    "nyt": [OpenKINYTTrainDataReader, OpenKINYTTrainDataReaderNegPairs]
}
EVAL_DATA_HANDLERS = {  # [ by_pair_ranking F, T ]
    "reverb": [OpenKIReverbEvalPerRelationDataReader, OpenKiReverbEvalPerPairDataReader],
    "nyt": [OpenKINYTEvalPerRelationDataReader, OpenKiNYTEvalPerPairDataReader]
}
ENTITY_WORD_DELIMITERS = {
    "reverb": " ",
    "nyt": " "
}
DATA_SOURCES = list(TRAIN_DATA_HANDLERS.keys())

EVALUATORS = {
    "MAP_ranking_pairs": (True, OpenKIMapEvaluatorRankingPairs),
    "MAP_ranking_rels": (False, OpenKIMapEvaluatorRankingRelations),
    "AUC_PR": (True, OpenKiAucPrEvaluator)  # AUC_PR doesn't care about pair ranking or not...
}
EVALUATOR_NAMES = list(EVALUATORS.keys())

RELATION_SCORERS = (
    "entity-neighbourhood",      # 0
    "entity-pair-query",         # 1
    "entity-pair-dual-attn",     # 2
    "e-model",                   # 3
)

# types of text encodings
ENTITY_WORD_EMBEDS = "entity-word-embeds"
PREDICATE_WORD_EMBEDS = "predicate-word-embeds"
PREDICATE_PAIR_WORD_EMBEDS = "predicate-pair-word-embeds"

# word embeddings
NO_EMBEDDING = "none"
FASTTEXT_EMBEDDING = "fasttext"
BERT_EMBEDDING = "bert"
RANDOM_EMBEDDING = "random"

WORD_EMBEDDINGS = (
    NO_EMBEDDING,
    FASTTEXT_EMBEDDING,
    BERT_EMBEDDING,
    RANDOM_EMBEDDING
)

RELATION_FT_EMBEDS_CACHE_FILE = "relation_text_ft_embeds"
RELATION_PAIR_FT_EMBEDS_CACHE_FILE = "relation_pair_text_ft_embeds"
RELATION_BERT_MEAN_EMBEDS_CACHE_FILE = "relation_text_bert_mean_embeds"
RELATION_PAIR_BERT_MEAN_EMBEDS_CACHE_FILE = "relation_pair_text_bert_mean_embeds"
RELATION_RANDOM_EMBEDS_CACHE_FILE = "relation_text_random_embeds"
ENTITY_FT_EMBEDS_CACHE_FILE = "entity_ft_text_embeds"
ENTITY_BERT_MEAN_EMBEDS_CACHE_FILE = "entity_text_bert_mean_embeds"
ENTITY_RANDOM_EMBEDS_CACHE_FILE = "entity_text_random_embeds"

EMBEDS_CACHE_FILES = {
    ENTITY_WORD_EMBEDS: {
        FASTTEXT_EMBEDDING: ENTITY_FT_EMBEDS_CACHE_FILE,
        BERT_EMBEDDING: ENTITY_BERT_MEAN_EMBEDS_CACHE_FILE,
        RANDOM_EMBEDDING: ENTITY_RANDOM_EMBEDS_CACHE_FILE
    },
    PREDICATE_WORD_EMBEDS: {
        FASTTEXT_EMBEDDING: RELATION_FT_EMBEDS_CACHE_FILE,
        BERT_EMBEDDING: RELATION_BERT_MEAN_EMBEDS_CACHE_FILE,
        RANDOM_EMBEDDING: RELATION_RANDOM_EMBEDS_CACHE_FILE
    },
    PREDICATE_PAIR_WORD_EMBEDS: {
        FASTTEXT_EMBEDDING: RELATION_PAIR_FT_EMBEDS_CACHE_FILE,
        BERT_EMBEDDING: RELATION_PAIR_BERT_MEAN_EMBEDS_CACHE_FILE
    }
}

# variants on treatment of data that can be listed after --data-variants on the command line
DATA_VARIANTS = ("entity-text-with-descriptions", "pred-text-with-descriptions", "nyt-text-entities",
                 "ignore_test_NA", "no_test_pairs_in_train", "deduplicate_kb_rel_triples",
                 "deduplicate_predicate_neighbours", "deduplicate_predicates_neighbours_log")
# TODO: setup deduplication in NYT original data
# TODO: TEXT ONLY: see above on deduplication for NYT
CACHE_FILE_VARIANT_SUFFIXES = {
    "entity-text-with-descriptions": "_w_desc",
    "pred-text-with-descriptions": "_w_desc",
    "nyt-text-entities": ""
}
CACHE_FILE_VARIANTS = set(CACHE_FILE_VARIANT_SUFFIXES.keys())
NYT_FB_ENT_SUFFIX = "_fb"

# embedding classes
TEXT_ENCODINGS = {
    ENTITY_WORD_EMBEDS: {
        FASTTEXT_EMBEDDING: FastTextEntityEmbedding,
        BERT_EMBEDDING: BertEntityTextEmbedding,
        RANDOM_EMBEDDING: RandomEntityTextEmbedding},
    PREDICATE_WORD_EMBEDS: {
        FASTTEXT_EMBEDDING: FastTextRelationEmbedding,
        BERT_EMBEDDING: BertRelationTextEmbedding,
        RANDOM_EMBEDDING: RandomRelationTextEmbedding},
    PREDICATE_PAIR_WORD_EMBEDS: {
        FASTTEXT_EMBEDDING: FastTextRelationPairEmbedding,
        BERT_EMBEDDING: BertRelationPairTextEmbedding}
}

# aggregation configurations for text encoding + learned embeddings
FC_TANH_THEN_SUM = "FC-tanh-then-sum"
CONCAT_THEN_FC_TANH = "concat-then-FC-tanh"
CONCAT_THEN_FC_RELU = "concat-then-FC-relu"
IGNORE_OPENKI_EMBEDS = "ignore"
TEXT_ENCODING_AGGREGATIONS = (
    FC_TANH_THEN_SUM,
    CONCAT_THEN_FC_TANH,
    CONCAT_THEN_FC_RELU,
    IGNORE_OPENKI_EMBEDS
)

RELATION_PLACEHOLDERS = {"<MASK_R>", "<PLACEHOLDER_R>"}
ENTITY_PLACEHOLDERS = {"<MASK_E>"}

NEGATIVE_SAMPLE_STRATEGIES = (
    "uniform",
)

OPTIMIZERS = {
    "Adam": optim.Adam,
    "RAdam": radam.RAdam,
    "SGD": optim.SGD,
    "AdaBelief": adabelief_pytorch.AdaBelief
}
OPTIMIZER_PARAMS = {
    "Adam": dict(),
    "RAdam": dict(),
    "SGD": dict(),
    "AdaBelief": {
        "eps": 1e-16,
        "betas": (0.9, 0.999),
        "weight_decouple": True,
        "rectify": True
    }
}

MODEL_ARGS_GROUP = "Model arguments"

FINAL_MODEL_LABEL = "final"
BEST_MODEL_LABEL = "BEST"

STATIC_STATE_RE = re.compile(r"(module\.)?scorers\.\d\.("
                             r"entityEncoder\.entity|"
                             r"entityEncoder\.relation_encoder\.predicate|"
                             r"relation_embedding\.predicate"
                             r")_encoder\.weight")
# 'scorers.0.entityEncoder.relation_encoder.predicate_encoder.weight'
# 'scorers.0.relation_embedding.predicate_encoder.weight'

STATE_DICT_NAME_CHANGES = (  # for backward compatability when we change a varaible name
    (r"(module\.)?scorers.(\d+).entityEncoder.relation_embedding.weight",
     r"\1scorers.\2.entityEncoder.relation_encoder.weight"),
)
STATE_DICT_SCORER_KEY_RE = re.compile(r"(.*)scorers\.(\d+)\.(.*)")


def update_state_dict_names(state_dict):
    for find_expr, repl_expr in STATE_DICT_NAME_CHANGES:
        changed_items = []
        for k, v in state_dict.items():
            new_k = re.sub(find_expr, repl_expr, k)
            if new_k != k:
                changed_items.append((k, new_k, v))
        for k, new_k, v in changed_items:
            state_dict[new_k] = v
            del state_dict[k]
    return state_dict
