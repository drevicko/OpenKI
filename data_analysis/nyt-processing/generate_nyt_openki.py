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

#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
#  and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#
#
import argparse
import gzip
import json
import math
import os
import random
import csv
from ast import literal_eval
from typing import List

from xopen import xopen
from collections import Counter, defaultdict
from pathlib import Path
import re
import subprocess
import logging
import ast
from tqdm import tqdm
from openki_pred_heuristic import extract_window


parser = argparse.ArgumentParser(
    description="Generate files for ingestion to OpenKI models. Input files are tsv versions of original protobuf files"
                " containing extracted NYT data, as provided by Riedel et.al. Use 'bp_to_tsv.py' to convert to tsv.")
# General program control
parser.add_argument("--log-level", default="INFO")
parser.add_argument("--report-stats-only", action="store_true", help="")
parser.add_argument("--test-location-only", action="store_true", help="")

# Source of predicates (or sentence proxies for them)
parser.add_argument("--openki-predicate-heuristic", action="store_true",
                    help="Use the predicate heuristic (described in the paper as 'sentences'!): A window from 3 words "
                         "before to 2 words after the entity spans, with entities replaced by <subj> and <obj> tokens.")
parser.add_argument("--dev-fraction", type=float, default=0.05,
                    help="Proportion of train data to reserve for validation. The default is 5%, as used by OpenKI "
                         "authors (private communication).")

args = parser.parse_args()

logging.basicConfig(level=args.log_level)


def wc_count(filename, unzip=None):
    if unzip or unzip is None and filename.endswith('.gz'):
        unzip_process = subprocess.Popen(['unpigz', '--to-stdout', filename], stdout=subprocess.PIPE)
        wc_process = subprocess.Popen(['wc', '-l'], stdin=unzip_process.stdout, stdout=subprocess.PIPE)
        unzip_process.stdout.close()  # enable write error in unpig if ssh dies
        out, err = wc_process.communicate()
    else:
        out, err = subprocess.Popen(['wc', '-l', filename],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
    try:
        return int(out.partition(b' ')[0])
    except ValueError:
        logging.critical(err)
        logging.critical(filename)
        logging.critical(out)
        raise


def get_plain_types(types):
    if types[0].endswith("_1") or types[0].endswith("_2"):
        plain_types = (types[0][:-2], types[1][:-2])
    else:
        plain_types = types
    return plain_types


report_stats_only = args.report_stats_only

# HARD-CODED VARIABLES TO CHOOSE HOW TO CONSTRUCT DATA
use_NA_in_test = False  # whether to include entity pairs without a detected kb relation in test data.
#                         This has a big impact on evaluation scores, as these negative examples dominate the test data.
#                         One could argue that a good proportion of these pairs SHOULD have a kb relation, but it is not
#                         there because the kb is incomplete.
#                         Previous work excluded these, hence we exclude also.
use_new_test_relations = False  # include kb_manual/testNewRelations.pb (all pairs have entities not linked to FreeBase)
#                                 this provides a deep pool of pairs with no KB relation (see above). Note that
#                                 evaluation scores with these two options turned on are VERY low!.
if use_new_test_relations:  # NA are used in test data with this option (overrides use_NA_in_test)
    assert use_NA_in_test

underscores_in_entites = False  # Replace spaces in entity names with underscores. This means entity names are not
#                                 properly recognised by the text encoders. Best to leave as False!
entities_in_predicate_proxy = False  # use (sentence, e1_span, e2_span) as predicate proxies (else just <sentence>)
#                                      NOTE: better not to do this, as all 'predicates' then appear in just one entity
#                                      neighbour list!


# CONSISTENCY CHECKS AND SETTING UP VARIABLES

# Data Locations
data_location = Path("/media/ian/Data2/riedel/data-nyt-2010/")
data_version_folder_wo_threshold = ""
data_version_folder_base = ""

if use_new_test_relations:
    data_version_folder = "openki_data_new_NA"
elif use_NA_in_test:
    data_version_folder = "openki_data_NA"
else:
    data_version_folder = "openki_data"
if args.openki_predicate_heuristic:
    data_version_folder += "-pred_heuristic"
if underscores_in_entites:
    data_version_folder = data_version_folder + "-ents_with_underscore"
# if use_MTB_markers:  # on second thoughts, this would dramatically in
#     data_version_folder += "-MTB_markers"
elif entities_in_predicate_proxy:
    data_version_folder += "-entitity_spans_in_preds"
if args.test_rels_in_nb:
    data_version_folder += "~dev-in-nbs~"

assert args.dev_fraction < 1
dev_frac_xtra = f"-{str(args.dev_fraction * 100).replace('.', '_').strip('0').strip('_')[:5]}_pc_dev"
data_version_folder += dev_frac_xtra
data_version_folder_wo_threshold += dev_frac_xtra
data_version_folder_base += dev_frac_xtra

if args.test_location_only:
    print(data_location / data_version_folder)
    print(f"Exists: {(data_location / data_version_folder).exists()}")
    exit(0)
(data_location / data_version_folder).mkdir(parents=True, exist_ok=True)

# Openki Data File Names
entities_file = "entities.json"   # [eid: {"id": eid, "type": str, "s_neighbours": [relid...],
#                                          "o_neighbours": [relid...],"names": [str...]}, ...]
types_file = "types.csv"
entity_names_file = "e2id.txt"
entity_neighbours_file = "e_neighbours.json"
relations_file = "relations.csv"  # rel. index, relation name; id, 0,1 ~ dummy relations; relations first
relation_types_file = "relation_types.csv"  # rel. index, relation name, types1, types2, ...
train_triples_file = "train_data.json"   # [{like reverb but no s/o_neighbours}, ...]
dev_triples_file = "dev_data.json"       # [{like reverb but no s/o_neighbours}, ...]
test_triples_file = "test_data.json"     # [{like reverb but no s/o_neighbours}, ...]


# input files
train_locations = [
    "heldout_relations/train",
    "kb_manual/train"
]
test_locations = [
    "heldout_relations/test",
]
tsv_file_names = [
    "Negative.tsv",
    "Positive.tsv"
]
new_relations_tsv = "kb_manual/testNewRelations.tsv"
test_file = test_locations[0] + tsv_file_names[1]
train_files = [line + n for line in train_locations for n in tsv_file_names] + \
              [line + tsv_file_names[0] for line in test_locations]
all_files = [line + n for line in train_locations + test_locations for n in tsv_file_names]
rels_file_name_base = None  # dummy value to pacify code formatter

if use_new_test_relations:
    all_files.append(new_relations_tsv)

# Regex for the subset of FB relations used in Riedel's 2010 NYT data.
reduced_relation_re = re.compile(r"location|person|people|business")
pred_fix_re = re.compile(r'\((NEG__)?[0-9a-zA-Z.à'"'"'-]+__\(')  # for mistakes in early extraction from CCG pipeline


def generate_data_files():
    num_skipped_preds = 0
    include_texts = False  # WIP to collate source texts and entity spans for more sophisticated ways of using eg. BERT
    #                        This is not complete, so do not set this to True!

    train_triples = {}  # keys: e_pairs; vals: {"preds":[relId...], "rels":[relId...]}
    dev_triples = {}  # keys: e_pairs; vals: {"preds":[relId...], "rels":[relId...]}
    test_triples = {}   # keys: e_pairs; vals: {"preds":[], "rels":[relId...]}
    entities_d = {}  # holds information on entities, keyed by entity fb id
    entities_=[]  # convenience list of entity fb ids
    # keys: fbEId; vals: {"id": eId, "type_id": int, "names": Counter(e_name), "fb_id": fbEId
    #                     "s_neighbours": [(pred_id, sent_id), ...],
    #                     "o_neighbours": [(pred_id, sent_id), ...]
    #                     "s_neighbour_spans": [(span[0], span[1]), ...],
    #                     "o_neighbour_spans": [(span[0], span[1]), ...]

    entity_mask = "<MASK_E>"
    entities_d[entity_mask] = {"id": len(entities_d), "fb_id": entity_mask, "type_id": 0, "names": {}}
    if include_texts:
        entities_d[entity_mask]["s_neighbour_spans"] = []
        entities_d[entity_mask]["o_neighbour_spans"] = []
    entities_d[entity_mask]["s_neighbours"] = []
    entities_d[entity_mask]["o_neighbours"] = []
    types: List[str] = [None]    # index: type_id, str(type)
    types_d = {None: 0}  # keys: str(type), vals: type_id
    kb_relations_d = {}  # keys: fbRelStr; values: kb_rel_id
    kb_relations = []    # fbRelStr's (index = kb_rel_id)
    predicates_d = {}   # {pred:  {"id": int, "types": Counter(), "subjects": Counter(), "objects": Counter()} }
    predicates = []     # predStr's (index = pred_rel_id)

    def get_predicate_id_w_updated(pred_key, static=False):
        pred_record = predicates_d.get(pred_key, None)
        # add predicate to predicates and predicates_d
        if pred_record is None:
            if static:
                raise ValueError(f"Predicate key {pred_key} not present in predicates dict!")
            pred_id_ = len(predicates_d)
            predicates_d[pred_key] = {"id": pred_id_, "types": Counter(), "subjects": Counter(), "objects": Counter()}
            predicates.append(pred_key)
            updated = True
        else:
            pred_id_ = pred_record["id"]
            updated = False
        return pred_id_, updated

    def get_predicate_id(pred_key, static=False):
        return get_predicate_id_w_updated(pred_key, static)[0]

    def get_type_id(a_type, static=False):
        type_id = types_d.get(a_type, None)
        if type_id is None:
            if static:
                raise ValueError(f"Type {a_type} not present in types list!")
            type_id = len(types)
            types.append(a_type)
            types_d[a_type] = type_id
        return type_id

    sentences_d = {"": {"id": 0, "entity_pairs": [], "spans": [], "in_test_data": None}}
    sentences = [""]  # This is used for KB relations (which have no associated sentence) as a dummy sentence
    for file_num, f_in in enumerate(all_files):
        # NOTE: the below only functions for NYT data without CCG. With ccg based data, is_test is set per row in the
        #       input files
        if use_NA_in_test:
            is_test_file = "test" in f_in
            # this includes testNegative and kb_manual/testNewRelations.tsv if use_new_test_relations is set
        else:
            is_test_file = f_in.endswith("testPositive.tsv")  # testNegative files don't have kb relations
        reported_multiple_types = False
        with open(data_location / f_in) as fin:
            fin = csv.reader(fin, delimiter='\t')
            num_used_triles = 0
            already_warned_on_no_entailment = False
            for row in tqdm(fin, desc=(data_location / f_in).stem, total=wc_count(str(data_location / f_in))):
                # if line_filter(row):
                #     triple_id_2_used.append(None)
                #     continue
                num_used_triles += 1

                # Unpack the  CCG relsEE / 2010 NYT file line (they differ a little)
                score = None
                fbid1, fbid2, e1, e2, e1_spans, e2_spans, type1, type2, relation, sentence = row
                relations = [relation]
                sentence = sentence.replace('\n', ' ')
                # TODO: Dongxu predicate proxy can be used here. We should also store the sentences as an option
                #       for text processing (eg: BERT of the entity span when calculating the predicate neighbour
                #       representation).
                if args.openki_predicate_heuristic:
                    predicate = extract_window(sentence, e1, e2)
                    if predicate is None:
                        num_skipped_preds += 1
                        if num_skipped_preds < 5:
                            logging.warning(f"Predicate heuristic failed! Using whole sentence!:\n{e1} and {e2} "
                                            f"in:\n{sentence}")
                        continue
                else:
                    predicate = sentence
                # pred_w_index = None
                p_span = None
                # article_id = None
                # Now determine test set status is_test to 0=train, 1=dev, 2=test
                if is_test_file:
                    is_test = 2
                else:
                    # ... decide if this is going to be train=0 or dev=1...
                    if random.random() < args.dev_fraction:
                        is_test = 1
                    else:
                        is_test = 0

                e1_spans = literal_eval(e1_spans)
                e2_spans = literal_eval(e2_spans)  # lists of 2-tuples, usually just one tuple

                ent_info = [entities_d.get(fbid1, None), entities_d.get(fbid2, None)]
                fbids = (fbid1, fbid2)
                these_types = (type1, type2)
                type_ids = (get_type_id(type1), get_type_id(type2))
                names = (e1, e2)

                # Update entity info with name, fb_id and type
                for s_or_o in (0, 1):
                    if ent_info[s_or_o] is None:
                        ent_id = len(entities_d)
                        entities_.append(fbids[s_or_o])
                        ent_info[s_or_o] = {"id": ent_id, "fb_id": fbids[s_or_o], "type_id": type_ids[s_or_o],
                                            "names": Counter()}
                        if include_texts:
                            ent_info[s_or_o]["s_neighbour_spans"] = []
                            ent_info[s_or_o]["o_neighbour_spans"] = []
                        ent_info[s_or_o]["s_neighbours"] = []
                        ent_info[s_or_o]["o_neighbours"] = []
                        ent_info[s_or_o]["types"] = Counter()
                        entities_d[fbids[s_or_o]] = ent_info[s_or_o]
                    else:
                        assert fbids[s_or_o] == ent_info[s_or_o]["fb_id"], f"{fbids[s_or_o]} != " \
                                                                           f"{ent_info[s_or_o]['fb_id']}"
                        if type_ids[s_or_o] != ent_info[s_or_o]["type_id"]:
                            # update type_id to most frequent type
                            if ent_info[s_or_o]["types"][types[ent_info[s_or_o]["type_id"]]] \
                                    < ent_info[s_or_o]["types"][these_types[s_or_o]] + 1:
                                ent_info[s_or_o]["type_id"] = type_ids[s_or_o]
                    # assert ent_info[s_or_o]["type"] == types[s_or_o], f"non-matching entity types for  " \
                    #                                                  f"{names[s_or_o]}({fbids[s_or_o]}, "\
                    #                                                  f"{ent_info[s_or_o]['fb_id']}), new type  " \
                    #                                                  f"{types[s_or_o]},old type "
                    #                                                  f"{ent_info[s_or_o]['type']}"
                    ent_info[s_or_o]["names"].update((names[s_or_o],))
                    ent_info[s_or_o]["types"].update((these_types[s_or_o],))

                # Update list of relations and get relation ids
                kb_rel_ids = []
                for relation in relations:
                    # get id (and add relation to kb_relations_d if not present)
                    if relation != "NA":
                        kb_rel_record = kb_relations_d.get(relation, None)
                        if kb_rel_record is None:
                            kb_rel_id = len(kb_relations)
                            kb_relations_d[relation] = {"id": kb_rel_id, "types": Counter()}
                            kb_relations_d[relation]["id"] = kb_rel_id
                            kb_relations.append(relation)
                        else:
                            kb_rel_id = kb_rel_record["id"]
                        kb_relations_d[relation]["types"].update((type_ids,))
                    else:
                        kb_rel_id = None
                    kb_rel_ids.append(kb_rel_id)

                # Update list of predicates and get predicate id
                predicate_key = (predicate, e1_spans, e2_spans) if entities_in_predicate_proxy else predicate
                pred_id = get_predicate_id(predicate_key)
                predicates_d[predicate_key]["types"].update((type_ids,))

                # update per predicate lists of entities as subjects/objects
                for s_or_o, subj_obj_key in enumerate(("subjects", "objects")):
                    predicates_d[predicate_key][subj_obj_key].update((fbids[s_or_o],))

                # Add triple info to train and test data, including so neighbour lists
                e_ids = tuple(ent_info[s_or_o]["id"] for s_or_o in (0, 1))
                train_triple = train_triples.get(e_ids, None)
                # NOTE: excluding dev/test pairs from train data is done in NYT_2010_Data.py
                if train_triple is None:  # all predicates go in train, so always add a train triple
                    train_triple = {"preds": [], "rels": [], "s_spans": [], "o_spans": [], "p_spans": [],
                                    "sentences": [], "is_test": [], "type_ids": type_ids}
                    train_triples[e_ids] = train_triple
                else:
                    if train_triple["type_ids"] != type_ids:
                        msg = f"triple with multiple type pairs: {train_triple['type_ids']}, {type_ids}"
                        if not reported_multiple_types:
                            logging.warning(msg)
                            reported_multiple_types = True

                # Store kb relation ids in train/dev/test triple data as appropriate
                if is_test > 0:
                    if is_test == 2:
                        these_dev_test_triples = test_triples
                    elif is_test == 1:
                        these_dev_test_triples = dev_triples
                    else:
                        raise ValueError(f"Got an illegal value for is_test: {is_test}. Should be one of ""{0, 1, 2}!")
                    triple = these_dev_test_triples.get(e_ids, None)
                    if triple is None:  # only add dev/test triples when is_test is set
                        triple = {"preds": [], "rels": [], "s_spans": [], "o_spans": [], "p_spans": [],
                                  "sentences": [], "is_test": [], "type_ids": type_ids}
                        these_dev_test_triples[e_ids] = triple
                    for kb_rel_id in kb_rel_ids:
                        if kb_rel_id is not None:
                            triple["rels"].append(kb_rel_id)
                            # Q: how do I / do I include predicates in dev/test data??
                            # A: the data loader class does this if the args flag --train-with-dev-preds is set
                        elif not use_NA_in_test:
                            raise ValueError(f"Missing KB relation in {f_in} at entity ids {e_ids}")
                elif is_test == 0:  # only kb relations not in dev/test go in train data
                    for kb_rel_id in kb_rel_ids:
                        if kb_rel_id is not None:
                            train_triple["rels"].append(kb_rel_id)
                else:
                    raise ValueError(f"Got an illegal value for is_test: {is_test}. Should be one of ""{0, 1, 2}!")
                train_triple["preds"].append(pred_id)
                # train_triple["sentences"].append(sentence_id)
                train_triple["s_spans"].append(e1_spans)
                train_triple["o_spans"].append(e2_spans)
                train_triple["p_spans"].append(p_span)
                train_triple["is_test"].append(is_test)


    num_kb_rels = len(kb_relations_d)
    num_predicates = len(predicates_d)
    num_useless_predicates = sum(1 for pred_info in predicates_d.values()
                                 if len(pred_info["subjects"]) <= 1 and len(pred_info["objects"]) <= 1)
    logging.info(f"{num_useless_predicates} single pair predicates out of {num_predicates}.")
    logging.info(f"{num_skipped_preds} sentences failed to produce predicates by heuristic")
    logging.info(f"{len(train_triples)} train, {len(dev_triples)} dev and {len(test_triples)} test triples; "
                 f"{len(entities_d)} entities.")

    def update_relation_id(old_id_, is_predicate):
        if is_predicate:
            return 2 + old_id_ + num_kb_rels
        else:
            return 2 + old_id_

    def update_neighbour_tuple(rel_id_tuple):
        # update relation/pred ids in s/o-neighbour lists with #relations and #dummy entries
        old_id_, sent_id = rel_id_tuple
        new_id_ = update_relation_id(old_id_, sent_id != 0)  # sentence 0 is empty and used for KB relations
        # NOTE: if sent_id is None, it is still recognised as a predicate (as opposed to KB relation)
        if include_texts:
            return new_id_, sent_id
        else:
            return new_id_

    # # Choose dev set triples
    # # NOTE: train_triples is indexed by entity pair.
    # if use_NA_in_test:
    #     train_rel_indices = [i for i, triple in enumerate(train_triples.values())]
    # else:
    #     train_rel_indices = [i for i, triple in enumerate(train_triples.values()) if len(triple["rels"]) > 0]
    #
    # dev_indices = set(random.sample(train_rel_indices, math.floor(len(train_rel_indices) * devFraction)))
    # # 30% to 10% of train data

    # Now that we have all kb relations, we can finalise combined relation/predicate indexes
    for ent_info in entities_d.values():
        for neighbour_key in ("s_neighbours", "o_neighbours"):
            ent_info[neighbour_key] = list(map(update_neighbour_tuple, ent_info[neighbour_key]))

    if report_stats_only:
        return

    # OUTPUT FILES FOR OPENKI
    # # Write sentences file
    # with open(data_location / data_version_folder / sentences_file, 'w') as f_sentences:
    #     f_sentences = csv.writer(f_sentences, delimiter='\t')
    #     for sentence_id, sentence in enumerate(sentences):
    #         sentence_info = sentences_d[sentence]
    #         f_sentences.writerow((sentence_id, sentence, sentence_info["in_test_data"]))

    def dump_openki_triples_json(triples_to_dump, file_name, inclusion_test=None, unique_rels=False):
        if unique_rels:
            compress_rels = lambda rels: list(set(rels))
        else:
            compress_rels = lambda rels: rels
        # # NOTE: NYT_2010_Data.py applies an option to main.py for this. No need to deduplicate here.
        # if unique_preds:
        #     compress_preds = lambda preds: list(set(preds))
        # else:
        #     compress_preds = lambda preds: preds

        # the dict keys here are in keeping with those used in OpenKI (for greater consistency)
        triple_generator = (
            {
                "input_s": e_pair[0],
                "input_o": e_pair[1],
                "input_p": [update_relation_id(predicate_id, True) for predicate_id in triple_["preds"]],
                "label": compress_rels([update_relation_id(kb_rel_id_, False) for kb_rel_id_ in triple_["rels"]]),
                "spans_p": triple_["p_spans"],
                "sentences": triple_["sentences"]
            } for e_pair, triple_ in triples_to_dump.items()
        )
        if inclusion_test is not None:
            triples_to_dump = [val for i, val in enumerate(triple_generator) if inclusion_test(i)]
        else:
            triples_to_dump = [val for val in triple_generator]
        json.dump(triples_to_dump, file_name, indent=4)

    logging.info(f"Saving files to {data_location / data_version_folder}")
    with open(data_location / data_version_folder / train_triples_file, 'w') as f_train, \
            open(data_location / data_version_folder / dev_triples_file, 'w') as f_dev:
        # if use_NYT_CCG_extractions_v0:
        dump_openki_triples_json(train_triples, f_train)
        dump_openki_triples_json(dev_triples, f_dev, unique_rels=True)

    with open(data_location / data_version_folder / test_triples_file, 'w') as f_test:
        dump_openki_triples_json(test_triples, f_test, unique_rels=True)

    with open(data_location / data_version_folder / relations_file, 'w') as f_rel, \
         open(data_location / data_version_folder / relation_types_file, 'w') as f_rel_types:
        f_rel = csv.writer(f_rel)
        f_rel_types = csv.writer(f_rel_types)
        f_rel.writerow((0, "<MASK_R>"))
        f_rel.writerow((1, "<PLACEHOLDER_R>"))
        f_rel_types.writerow((0, "<MASK_R>", {}))
        f_rel_types.writerow((1, "<PLACEHOLDER_R>", {}))
        # NOTE: any changes to the order (kb rels then predicates) and "+ 2" below need to be reflected in
        #       update_relation_id() above.
        f_rel.writerows(((i + 2, relation) for i, relation in enumerate(kb_relations)))
        f_rel_types.writerows(((i + 2, relation, dict(kb_relations_d[relation]["types"]))
                               for i, relation in enumerate(kb_relations)))
        f_rel.writerows(((i + 2 + num_kb_rels, pred) for i, pred in enumerate(predicates)))
        f_rel_types.writerows(((i + 2 + num_kb_rels, pred, dict(predicates_d[pred]["types"]))
                               for i, pred in enumerate(predicates)))

    # {"id": len(entities_d), "fb_id": Counter(), "types": Counter(),
    #  "s_neighbours": [(pred_id, sent_id, span), ...], "o_neighbours": [(pred_id, sent_id, span), ...],
    #  "names": Counter()}
    with open(data_location / data_version_folder / entities_file, 'w') as f_ent:
        dump_data = []
        for fbid in entities_:
            entity_info = entities_d.get(fbid, None)
            if entity_info is not None:
                dump_data.append(entity_info)
        json.dump(dump_data, f_ent, indent=4)

    # write out the types index
    with open(data_location / data_version_folder / types_file, "w") as f_types:
        f_types = csv.writer(f_types)
        f_types.writerows(enumerate(types))


if report_stats_only:
    logging.warning(f"report_stats_only is set! No data will be written to file!!")
logging.info(f"generating data files for original NYT data")
generate_data_files()
