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
import math
from copy import copy

import ast
from functools import partial

import torch
import csv
import json
from pathlib import Path
from collections import defaultdict, Counter

from OpenKI import logger
from OpenKI.OpenKI_Data import OpenKiGraphDataHandler, OpenKiEvalPerRelationDataHandler, \
    OpenKiEvalPerPairDataHandler, OpenKITrainDataReaderNegPairs, OpenKITrainDataReader


class OpenKiNYTDataReader(OpenKiGraphDataHandler):
    num_dummy_relations = 2  # the first 2 relations are <MASK_R> and <PLACEHOLDER_R>

    def __init__(self, nyt_folder, data_split, device, eval_top_n_rels=None, use_dev_preds=False, variants=[],
                 use_fb_entities=True, **kwargs):
        """
        Base class for data readers for OpenKI NYT data [Riedel 2010].
        :param nyt_folder: Folder containing reverb data from openKI paper
        :param data_split: Name of data split: "train", "dev" or "test"
        :param device: torch device we're using
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param use_dev_preds: Includes dev/test pairs and their OpenIE relations (KB relations and s/o-neighbours
                removed). Use for e-model, otherwise NOT RECOMMENDED!!
        """
        super().__init__(**kwargs)
        if variants is None:
            variants = []
        self.no_test_pairs_in_train = "no_test_pairs_in_train" in variants
        # self.no_test_pairs_in_train_old = "no_test_pairs_in_train" in variants
        self.deduplicate_kb_relation_triples = "deduplicate_kb_rel_triples" in variants
        #   De-duplicating lists of kb relations for each pair results in them being less frequently sampled, both
        # as positive and negative samples.The data files have each relation repeated as many times as there are
        # predicates for the entity pair. This is equivalent to one repetition for each source text containing the
        # entity pair.
        self.deduplicate_predicate_neighbours = "deduplicate_predicate_neighbours" in variants
        self.deduplicate_predicates_neighbours_log = "deduplicate_predicates_neighbours_log" in variants
        self.deduplicate_predicate_neighbours = self.deduplicate_predicate_neighbours or \
                                                self.deduplicate_predicates_neighbours_log
        #   Neighbour lists for entities can contain many entries from some predicates due to multple occurrences of
        # the predicate with that entity in texts. These options reduce this duplication, either entirely or on a log
        # scale (ie: n entries in an neighbour list reduced to ~log(n) ).

        self.use_entailments = "use_entailments" in variants

        folder = Path(nyt_folder)
        self.data_folder = data_split
        self.device = device
        assert data_split in ("dev", "test", "train")

        # Load entity data
        with open(folder / "entities.json") as f:
            self.entity_data = json.load(f)  # list of dicts
            assert all(e_info["id"] == i for i, e_info in enumerate(self.entity_data))
        self.entities = [None for _ in self.entity_data]
        if use_fb_entities:
            try:
                with open(folder / "e2info_fb.json") as f:
                    e2info = json.load(f)
                name_info_fields = ["name", "name_label", "name_wiki_t", "name_wiki", "name_wiki_", "alias"]
                desc_info_fields = ["desc"]

                def get_name_or_desc(entity_id, key_list, mention_span):
                    info = e2info.get(str(entity_id), None)
                    name = None
                    if info is not None:
                        for name_key in key_list:
                            name = info.get(name_key, None)
                            if name:  # this should catch empty lists from e2info_ft and None from e2info_wd
                                break
                        if type(name) is list:
                            if name:
                                name = name[0]
                    name = name or mention_span
                    if name is None:
                        return None
                    else:
                        return name.strip('" \n')

                # setup text representations of entities and relations
                # DATA_VARIANTS = ("entity-text-with-descriptions", "pred-text-with-descriptions")
                missing_entity_infos = 0
                for ent_info in self.entity_data:
                    ent_id = ent_info["id"]
                    try:
                        ent_mention = next(iter(ent_info["names"].keys()))
                    except StopIteration:
                        ent_mention = None
                    fb_ent_id = ent_info["fb_id"]
                    ent_name = get_name_or_desc(ent_id, name_info_fields, ent_mention)
                    if "entity-text-with-descriptions" in variants:
                        ent_desc = get_name_or_desc(ent_id, desc_info_fields, None)
                        if ent_name is None:
                            if ent_desc is None:
                                missing_entity_infos += 1
                            else:
                                self.entities[ent_id] = ent_desc
                        elif ent_desc is None:
                            self.entities[ent_id] = ent_name
                        else:
                            self.entities[ent_id] = f"{ent_name}. {ent_desc}"
                    elif ent_name is None:
                        missing_entity_infos += 1
                    else:
                        self.entities[ent_id] = ent_name
            except FileNotFoundError:
                logger.warning(f"FreeBase entity info file {folder / 'e2info_fb.json'} missing, "
                               f"reverting to entity names or spans in NYT data!")
                use_fb_entities = False
        if not use_fb_entities:
            for e in self.entity_data:
                try:
                    self.entities[e["id"]] = next(iter(e["names"].keys()))  # use first available name
                except StopIteration:
                    self.entities[e["id"]] = e["fb_id"]  # fallback on id if no names found

        # Load relation names
        with open(folder / "relations.csv") as f:
            relations_info = list(csv.reader(f))
            self.relations = [line[1] for line in relations_info]
            # TODO: relation types are in a separate file "relation_types.csv": relid, dict(type: count)
            # if len(relations_info[0]) == 3:
            #     self.relation_types = [ast.literal_eval(line[2]) for line in relations_info]
            # else:
            #     self.relation_types = None  # fallback for older data without types stored in relations file

        # Load main data
        with open(folder / f"{data_split}_data.json") as f:
            self.data = json.load(f)
            if self.no_test_pairs_in_train and data_split == "train":
                # Remove all dev/test entity pairs from train data
                with open(folder / f"test_data.json") as f_test:
                    exclude_pairs = set((pair_info['input_s'], pair_info['input_o']) for pair_info in json.load(f_test))
                # older models did not do the following (which is a problem). self.no_test_pairs_in_train_old
                with open(folder / f"dev_data.json") as f_dev:
                    exclude_pairs |= set((pair_info['input_s'], pair_info['input_o']) for pair_info in json.load(f_dev))
                self.data = [pair_info for pair_info in self.data
                             if (pair_info['input_s'], pair_info['input_o']) not in exclude_pairs]
            elif data_split == "train" or data_split == "dev":
                # Remove test kb-relation triples from dev and test, and dev kb-relation triples from train data
                # Collate test labels (to be removed from dev and train data)
                with open(folder / f"test_data.json") as f_test:
                    exclude_triples = {(pair_info['input_s'], pair_info['input_o']): set(pair_info['label'])
                                       for pair_info in json.load(f_test) if pair_info['label']}
                if data_split == "train":
                    # Collate dev labels (to be removed from train data only)
                    with open(folder / f"dev_data.json") as f_test:
                        for pair_info in json.load(f_test):
                            dev_labels = pair_info["label"]
                            if dev_labels:
                                pair = (pair_info['input_s'], pair_info['input_o'])
                                exclude_labels = exclude_triples.get(pair, set())
                                if not exclude_labels:
                                    exclude_triples[pair] = exclude_labels
                                exclude_labels.update(dev_labels)

                # remove excluded triples from train/dev
                for pair_info in self.data:
                    exclude_labels = exclude_triples.get((pair_info['input_s'], pair_info['input_o']), None)
                    if exclude_labels is not None:
                        labels = pair_info['label']
                        if labels:
                            pair_info['label'] = [r for r in labels if r not in exclude_labels]

        if (data_split == "test" or data_split == "dev") and "ignore_test_NA" in variants:
            logger.info(f"Building eval data without NA (no kb relation) entity pairs!")
            self.data = [triple for triple in self.data if triple["label"]]
        if data_split == "train" and use_dev_preds:
                for split in ("test", "dev"):
                    with open(folder / f"{split}_data.json") as f:
                        split_data = json.load(f)
                        for pair_info in split_data:
                            if pair_info['input_p']:
                                pair_info['label'] = []
                                # pair_info['input_s_neighbor'] = None  # these are not used anyway ...
                                # pair_info['input_o_neighbor'] = None  # the neighbour info comes from entities.json
                                self.data.append(pair_info)
                                # TODO: check that non-unique entity pairs is ok...
        self.num_relations = len(self.relations)
        self.num_entities = len(self.entity_data)

        # Find the first predicate index (all KG relation indices are less than this).
        # self.first_predicate_index = min(min(preds) for pair_info in self.data)  # assumes we instantiate the min!
        i = 0
        for i in range(len(self.relations)):
            p = self.relations[i]
            if not p.startswith("/") and p != "<MASK_R>" and p != "<PLACEHOLDER_R>":
                break
        self.first_predicate_index = i
        self.num_kb_relations = self.first_predicate_index - self.num_dummy_relations

        # Set up sentences associate with triples
        self.relation_texts = copy(self.relations)  # in NYT data, the relation representation is the sentence text

        for i, relation in enumerate(self.relations[:self.first_predicate_index]):
            self.relation_texts[i] = self.relations[i].strip('/').replace('/', '. ').replace('_', ' ')

        # TODO: TEXTS: Update the NYT pre-processing to have neighbours with two values: (pred_id, text_id)
        #       This will enable separation of predicates and source texts if we wish to use a different predicate
        #       representation than "sentence proxies".

        # A few persistent tensors to avoid allocating them each time
        self.relations_mask = torch.zeros((self.num_relations,), dtype=torch.bool, device=self.device,
                                          requires_grad=False)
        self.empty_index_tensor = torch.tensor([], dtype=torch.long, device=self.device, requires_grad=False)

        # Set up indexes for (s,o) pairs, s and o for quick access to data
        self.entity_pair_index = defaultdict(list)
        self.relations_by_e_pair = {}       # used as entity pair neighbour lists
        self.relation_lists_by_e_pair = {}  # used to check negative examples aren't positive
        self.subject_index = defaultdict(list)
        self.object_index = defaultdict(list)
        self.label_index = defaultdict(list)
        seen_so_neighbours = []
        seen_s = []
        seen_o = []
        # seen_relations = []
        # seen_predicates = []
        for data_index, pair_info in enumerate(self.data):
            # set up seen entities, so neighbour lists ('relatioon_lists_by_e_pair') and indexes back into self.data
            so = (pair_info["input_s"], pair_info["input_o"])
            s, o = so
            these_predicates = set(pair_info['input_p'])
            these_kb_relations = set(pair_info['label'])
            # seen_relations.extend(these_kb_relations)
            # seen_predicates.extend(these_predicates)
            these_relations = sorted(these_kb_relations | these_predicates)
            seen_so_neighbours.extend(these_relations)
            if so in self.entity_pair_index:  # NOTE: this can happen when dev/test preds are included!
                if len(these_relations) == 0 and len(these_predicates) == 0:
                    continue  # Q: does this ever happen?? A: not with Riedel's 2010 NYT data
                these_relations.extend(self.relation_lists_by_e_pair[so])  # add in those already found
                these_relations = sorted(set(these_relations))
            else:
                seen_s.append(s)
                seen_o.append(o)

            # save/update entity pair data (ie: triples). NOTE: there are no repeats here!!
            self.relation_lists_by_e_pair[so] = tuple(these_relations)
            # TODO: TEXTS: to decouple preds and texts, this needs to be (#these_relations, 2) tensor or (#rels, 1)
            #       tensor in relations_by_e_pair
            self.relations_by_e_pair[so] = torch.tensor(these_relations, dtype=torch.long,
                                                        requires_grad=False, device=device).unsqueeze(-1)
            # the unsqueeze(-1) adds a dummy dimension of size 1.
            # Later a size 2 version of this can contain text indices

            # update indexes back into data
            self.entity_pair_index[so].append(data_index)
            self.subject_index[s].append(data_index)
            self.object_index[o].append(data_index)
            for label in pair_info['label']:
                self.label_index[label].append(so)

        # setup deduplication of neighbour lists.
        def deduplicate(nb_list: list, log_scaled=False):
            if len(nb_list) and type(nb_list[0]) is list:
                nb_list = map(tuple, nb_list)
            if log_scaled:
                counts = Counter(nb_list)
                counts = {p_id: math.ceil(math.log2(c)) + 1 for p_id, c in counts.items()}
                return sum(([p_id] * count for p_id, count in counts.items()), [])
            return list(set(nb_list))

        # TODO: TEXTS: if we have texts decoupled from predicates, s_or_o_neighbours need to be shape (#neighbours, 2)
        if self.deduplicate_predicate_neighbours:
            deduplicate = partial(deduplicate, log_scaled=self.deduplicate_predicates_neighbours_log)
            self.s_or_o_neighbours = [
                [torch.tensor(deduplicate(e_info["s_neighbours"]), dtype=torch.long, requires_grad=False, device=device).unsqueeze(-1),
                 torch.tensor(deduplicate(e_info["o_neighbours"]), dtype=torch.long, requires_grad=False, device=device).unsqueeze(-1)]
                for e_info in self.entity_data]
        else:
            self.s_or_o_neighbours = [
                [torch.tensor(e_info["s_neighbours"], dtype=torch.long, requires_grad=False, device=device).unsqueeze(-1),
                 torch.tensor(e_info["o_neighbours"], dtype=torch.long, requires_grad=False, device=device).unsqueeze(-1)]
                for e_info in self.entity_data]

        self.max_neighbour_len = max(max(ts.shape[-1], to.shape[-1]) for ts, to in self.s_or_o_neighbours)
        self.expected_neighbour_len = sum(ts.shape[-1] + to.shape[-1] for ts, to in self.s_or_o_neighbours) \
                                      / len(self.s_or_o_neighbours) / 2

        self.entity_pair_list = tuple(self.entity_pair_index.keys())
        self.subject_index.default_factory = None
        self.object_index.default_factory = None

        self.seen_s_or_o_neighbours = torch.zeros((2, self.num_relations), dtype=torch.bool, requires_grad=False,
                                                  device=device)
        for s_neighbours, o_neighbours in self.s_or_o_neighbours:
            self.seen_s_or_o_neighbours[0, s_neighbours] = True  # create boolean mask of seen neighbours
            self.seen_s_or_o_neighbours[1, o_neighbours] = True  # create boolean mask of seen neighbours

        self.seen_so_neighbours = torch.zeros((self.num_relations,), dtype=torch.bool, requires_grad=False,
                                              device=device)
        self.seen_so_neighbours[seen_so_neighbours] = True
        seen_s = list(set(seen_s))
        seen_o = list(set(seen_o))
        self.seen_entities = torch.zeros((2, len(self.entity_data)), dtype=torch.bool,
                                         requires_grad=False, device=device)
        self.seen_entities[0, seen_s] = True
        self.seen_entities[1, seen_o] = True

        # setup lists for later use
        # A data sample:
        # {'input_o': 8406,
        #  'input_o_neighbor': [225, 7, 48],
        #  'input_p': [7136, 516, 630],
        #  'input_s': 17546,
        #  'input_s_neighbor': [156, 3, 3],
        #  'label': [3]}
        # setup triple data (these are the data points we iterate over during training)
        self.all_triple_data, self.kb_only_triple_data = [], []
        for data_index, pair_info in enumerate(self.data):
            s, o, pred_list, rel_list = (pair_info[k] for k in ('input_s', 'input_o', 'input_p', 'label'))
            pred_triples, rel_triples = [], []
            link_list = pred_list + rel_list
            for new_triples, links_to_add in ((pred_triples, pred_list), (rel_triples,  rel_list)):
                new_triples.extend(
                    ((s, o, rel),
                     self.s_or_o_neighbours[s][0],  # I believe these are not currently used...
                     self.s_or_o_neighbours[o][1],
                     link_list) for rel in links_to_add
                )

            # these are only really used in eval and test data
            self.kb_only_triple_data.extend(rel_triples)

            # these are only really used in train data
            self.all_triple_data.extend(rel_triples)
            self.all_triple_data.extend(pred_triples)
        self.top_relations = None
        self.set_top_relations(OpenKiNYTDataReader, self, eval_top_n_rels)

        # NOTE: we set self.triple_data in OpenKIReverbTrainDataReader.__init__()


class OpenKINYTTrainDataReaderNegPairs(OpenKITrainDataReaderNegPairs, OpenKiNYTDataReader):
    pass


class OpenKINYTTrainDataReader(OpenKITrainDataReader, OpenKiNYTDataReader):
    pass


class OpenKINYTEvalPerRelationDataReader(OpenKiNYTDataReader, OpenKiEvalPerRelationDataHandler):
    def __init__(self, nyt_folder, data_split, device, eval_top_n_rels=None, top_relations=None, **kwargs):
        """
        Load evaluation data and set up for OpenKI evaluation.
        :param nyt_folder: see OpenKiNYTDataReader
        :param data_split: see OpenKiNYTDataReader
        :param device: see OpenKiNYTDataReader
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param top_relations: top_relations as calculated in train data, to use here (supersedes eval_top_n_relations)
        """
        super().__init__(nyt_folder, data_split, device, eval_top_n_rels if top_relations is None else -1, **kwargs)
        if top_relations is not None:
            self.top_relations = top_relations  # these should be from the train data
        self.relation_indexed_data = tuple(
            tuple(td[0] for td in self.kb_only_triple_data if td[0][2] == i)  # td[0] is the triple
            for i in self.top_relations
        )


class OpenKiNYTEvalPerPairDataReader(OpenKiNYTDataReader, OpenKiEvalPerPairDataHandler):
    def __init__(self, nyt_folder, data_split, device, **kwargs):
        super().__init__(nyt_folder, data_split, device, **kwargs)
        top_relations = self.top_relations
        entity_pairs = []
        labels = []
        for i, item in enumerate(self.data):
            ok_labels = [rel - self.num_dummy_relations for rel in item['label'] if rel in top_relations]
            entity_pairs.append((item['input_s'], item['input_o']))
            # TODO: is it beneficial to have a tuple of 5k short tensors here?
            labels.append(torch.tensor(ok_labels, dtype=torch.long, device=device))
        self.entity_pairs = torch.tensor(entity_pairs, dtype=torch.long, device=device)
        self.labels = tuple(labels)
