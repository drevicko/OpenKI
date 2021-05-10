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

import json
import re
from collections import defaultdict
from pathlib import Path

import torch

from OpenKI import logger
from OpenKI.OpenKI_Data import OpenKiGraphDataHandler, OpenKiEvalPerRelationDataHandler, \
    OpenKiEvalPerPairDataHandler, OpenKITrainDataReaderNegPairs, OpenKITrainDataReader


class OpenKiReverbDataReader(OpenKiGraphDataHandler):
    num_dummy_relations = 2  # the first 2 relations are <MASK_R> and <PLACEHOLDER_R>

    def __init__(self, reverb_folder, data_split, device, eval_top_n_rels=None, use_dev_preds=False, variants=[],
                 enatilments_to=None, untyped_entailments=False,
                 use_freebase_names=False, *args, **kwargs):
        """
        Base class for data readers for OpenKI reverb data provided with paper.
        :param reverb_folder: Folder containing reverb data from openKI paper
        :param data_split: Name of data split: "train", "dev" or "test"
        :param device: torch device we're using
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param use_dev_preds: Includes dev/test pairs and their OpenIE relations (KB relations and s/o-neighbours
                removed). Use for e-model, otherwise NOT RECOMMENDED!!
        """
        super().__init__(*args, **kwargs)
        folder = Path(reverb_folder)
        self.data_folder = data_split
        self.device = device
        assert data_split in ("dev", "test", "train")
        with open(folder / "e2name.json") as f:
            self.e2name = json.load(f)
        # self.text_variant = "_fb" if use_freebase_names else "_wd"
        if use_freebase_names:  # NOTE: cached embeddings don't respect use_freebase_names!!
            with open(folder / "e2info_fb.json") as f:
                e2info = json.load(f)
            name_info_fields = ["name", "name_label", "name_wiki_t", "name_wiki", "name_wiki_", "alias"]
            desc_info_fields = ["desc"]
        else:
            with open(folder / "e2info_wd.json") as f:
                e2info = json.load(f)
            name_info_fields = ["name", "wikipedia"]
            desc_info_fields = ["desc"]
        with open(folder / "p2name.json") as f:
            self.p2name = json.load(f)
        with open(folder / f"{data_split}_data.json") as f:
            self.data = json.load(f)

        def get_name_or_desc(entity_id, key_list):
            info = e2info[entity_id]
            if info is None:
                return None
            name = None
            for name_key in key_list:
                name = info.get(name_key, None)
                if name:  # this should catch empty lists from e2info_ft and None from e2info_wd
                    break
            if type(name) is list:
                if name:
                    name = name[0]
                else:
                    name = None
            if name is None:
                return None
            else:
                return name.strip('" \n')

        # setup text representations of entities and relations
        # DATA_VARIANTS = ("entity-text-with-descriptions", "pred-text-with-descriptions")
        self.entities = [None for _ in self.e2name]
        missing_entitiy_infos = 0
        for ent_id, fb_ent_id in self.e2name.items():
            ent_id = int(ent_id)
            ent_name = get_name_or_desc(ent_id, name_info_fields)
            if "entity-text-with-descriptions" in variants:
                ent_desc = get_name_or_desc(ent_id, desc_info_fields)
                if ent_name is None:
                    if ent_desc is None:
                        missing_entitiy_infos += 1
                    else:
                        self.entities[ent_id] = ent_desc
                elif ent_desc is None:
                    self.entities[ent_id] = ent_name
                else:
                    self.entities[ent_id] = f"{ent_name}. {ent_desc}"
            elif ent_name is None:
                missing_entitiy_infos += 1
            else:
                self.entities[ent_id] = ent_name


        # count predicates and relations and find special predicate/relation indices
        self.num_relations = len(self.p2name)
        self.num_entities = len(self.e2name)

        # Find the first predicate index (all KG relation indices are less than this).
        # self.first_predicate_index = min(min(preds) for pair_info in self.data)  # assumes we instantiate the min!
        i = 0
        for i in range(len(self.p2name)):
            p = self.p2name[str(i)]
            if not p.startswith("<http://rdf.freebase.com") and p != "<MASK_R>" and p != "<PLACEHOLDER_R>":
                break
        self.first_predicate_index = i
        self.num_kb_relations = self.first_predicate_index - self.num_dummy_relations

        # A few persistent tensors to avoid allocating them each time
        self.relations_mask = torch.zeros((self.num_relations,), dtype=torch.bool, device=self.device,
                                          requires_grad=False)
        self.empty_index_tensor = torch.tensor([[]], dtype=torch.long, device=self.device, requires_grad=False)

        # setup relation and predicate texts
        # "251": "<http://rdf.freebase.com/ns/base.biblioness.bibs_location.state>",
        # "256": "reverb::\"is a district located in\"",

        relation_name_re = re.compile(r'<http://rdf.freebase.com/ns/(.*)>')
        predicate_name_re = re.compile(r'reverb::"(.*)"')
        self.relation_texts = [None for _ in self.p2name]
        for pred_id in range(self.num_dummy_relations, self.first_predicate_index):
            pred_name = None
        for pred_id_s, pred_name in self.p2name.items():
            pred_id = int(pred_id_s)
            if pred_id < self.num_dummy_relations:
                continue
            if pred_id < self.first_predicate_index:
                text = relation_name_re.match(pred_name).group(1)
                text = ", ".join(text.split('.'))
                text = " ".join(text.split('_'))
                self.relation_texts[pred_id] = text
            else:
                text = predicate_name_re.match(pred_name).group(1)
                self.relation_texts[pred_id] = text
        # TODO: setup relations and relation_texts for reverb data...


        # Add text predicate info from dev/test to training data
        if data_split == "train":
            self.consolidate_train_labels()
            if use_dev_preds:
                for split in ("test", "dev"):
                    with open(folder / f"{split}_data.json") as f:
                        split_data = json.load(f)
                        for pair_info in split_data:
                            if pair_info['input_p']:
                                pair_info['label'] = []  # these are the true dev/test data (ie: the KB relations)
                                # TODO: why remove s/o neighbours? Because some of them may be kb relations? And the
                                #       neighbour lists will be in the train data anyway? This means we omit neighbour
                                #       info for predicates ONLY in dev/test data...
                                #       Really, we shouldn't use dev/test pairs at all (not even for entity pair models)
                                # Answer: They want to test performance on entities never seen before. Imagine dev/test
                                #       data as a collection of single instances.
                                #       It is not clear that reverb data neighbour lists include neighbour instances
                                #       where the other neighbour is an 'unseen' one from dev/test data...
                                pair_info['input_s_neighbor'] = None
                                pair_info['input_o_neighbor'] = None
                                self.data.append(pair_info)

        # Set up indexs for (s,o) pairs, s and o for quick access to data
        self.entity_pair_index = {}
        self.relations_by_e_pair = {}
        self.relation_lists_by_e_pair = {}
        self.subject_index = defaultdict(list)
        self.object_index = defaultdict(list)
        self.label_index = defaultdict(list)
        self.s_or_o_neighbours = [[self.empty_index_tensor, self.empty_index_tensor] for _ in self.e2name]
        seen_s_neighbours = []
        seen_o_neighbours = []
        seen_so_neighbours = []
        seen_s = []
        seen_o = []
        self.max_neighbour_len = 0
        for data_index, pair_info in enumerate(self.data):
            so = (pair_info["input_s"], pair_info["input_o"])
            if so in self.entity_pair_index:  # NOTE: this doesn't actually happen in released data!
                # # other_index = self.entity_pair_index[so]
                # assert set(pair_info['input_p'] + pair_info['label']) <= self.relations_by_e_pair[so]
                # s, o = so
                # for label in pair_info['label']:
                #     assert so in self.label_index[label]
                # if pair_info['input_s_neighbor'] is not None:
                #     assert list(self.s_or_o_neighbours[s][0]) == pair_info['input_s_neighbor']
                # if pair_info['input_o_neighbor'] is not None:
                #     assert list(self.s_or_o_neighbours[o][1]) == pair_info['input_o_neighbor']
                continue
            self.entity_pair_index[so] = data_index
            these_relations = sorted(set(pair_info['input_p'] + pair_info['label']))
            self.relation_lists_by_e_pair[so] = tuple(these_relations)
            self.relations_by_e_pair[so] = torch.tensor(these_relations, dtype=torch.long,
                                                        requires_grad=False, device=device).unsqueeze(-1)
            # the unsqueeze(-1) adds a dummy dimension of size 1.
            # Later a size 2 version of this can contain text indices
            seen_so_neighbours.extend(these_relations)
            s, o = so
            seen_s.append(s)
            seen_o.append(o)
            self.subject_index[s].append(data_index)
            self.object_index[o].append(data_index)
            for label in pair_info['label']:
                self.label_index[label].append(so)
            if self.s_or_o_neighbours[s][0] is self.empty_index_tensor and pair_info['input_s_neighbor'] is not None:
                neighbors = pair_info['input_s_neighbor']
                self.s_or_o_neighbours[s][0] = torch.tensor(neighbors, dtype=torch.int64, requires_grad=False,
                                                            device=device).unsqueeze(-1)
                seen_s_neighbours.extend(pair_info['input_s_neighbor'])
                self.max_neighbour_len = max(len(neighbors), self.max_neighbour_len)
                # TODO: SeenOnlyNeighbours: only calculate 'seen_s_neighbours' when the option is selected?
            if self.s_or_o_neighbours[o][1] is self.empty_index_tensor and pair_info['input_o_neighbor'] is not None:
                neighbors = pair_info['input_o_neighbor']
                self.s_or_o_neighbours[o][1] = torch.tensor(neighbors, dtype=torch.int64, requires_grad=False,
                                                            device=device).unsqueeze(-1)
                seen_o_neighbours.extend(pair_info['input_o_neighbor'])
                self.max_neighbour_len = max(len(neighbors), self.max_neighbour_len)
                # TODO: SeenOnlyNeighbours: only calculate 'seen_s_neighbours' when the option is selected?
            # Note that the above lists are equal for all appearances of s or o in the released OpenKI data

        self.expected_neighbour_len = sum(ts.shape[-1] + to.shape[-1] for ts, to in self.s_or_o_neighbours) \
                                      / len(self.s_or_o_neighbours) / 2
        self.entity_pair_list = tuple(self.entity_pair_index.keys())
        self.subject_index.default_factory = None
        self.object_index.default_factory = None
        self.seen_s_or_o_neighbours = torch.zeros((2, self.num_relations), dtype=torch.bool, requires_grad=False,
                                                  device=device)
        self.seen_s_or_o_neighbours[0, seen_s_neighbours] = True  # create boolean mask of seen neighbours
        self.seen_s_or_o_neighbours[1, seen_o_neighbours] = True  # create boolean mask of seen neighbours

        self.seen_so_neighbours = torch.zeros((self.num_relations,), dtype=torch.bool, requires_grad=False,
                                              device=device)
        self.seen_so_neighbours[seen_so_neighbours] = True
        seen_s = list(set(seen_s))
        seen_o = list(set(seen_o))
        self.seen_entities = torch.zeros((2, len(self.e2name)), dtype=torch.bool, requires_grad=False, device=device)
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
        self.all_triple_data, self.kb_only_triple_data = [], []
        for data_index, pair_info in enumerate(self.data):
            s, o, pred_list, rel_list = (pair_info[k] for k in ('input_s', 'input_o', 'input_p', 'label'))
            pred_triples, rel_triples = [], []
            link_list = pred_list + rel_list
            for new_triples, links_to_add in ((pred_triples, pred_list), (rel_triples,  rel_list)):
                new_triples.extend(
                    ((s, o, rel),
                     self.s_or_o_neighbours[s][0],
                     self.s_or_o_neighbours[o][1],
                     link_list) for rel in links_to_add
                )

            # these are only really used in eval and test data
            self.kb_only_triple_data.extend(rel_triples)

            # these are only really used in train data
            self.all_triple_data.extend(rel_triples)
            self.all_triple_data.extend(pred_triples)
        self.set_top_relations(OpenKiReverbDataReader, self, eval_top_n_rels)
        # NOTE: we set self.triple_data in OpenKIReverbTrainDataReader.__init__()

    def consolidate_train_labels(self):
        """
        Train data has the 'label' attribute (KG relation indices between subject and object) spread over multiple
        contiguous data entries. ie: there are multiple contiguous dicts in the list self.data all the same up to
        the 'label' entry, which is a different int for each one. Here we consolidate these into a list in order to
        have the same format as the dev and test data.
        We also do some checks to make sure the assumed data format is indeed followed.
        :return: Void
        """
        assert self.data_folder == "train", f"Consolidation should only be done on train data!"
        new_train_data = []
        so_seen = set()
        so_last = None
        pair_data_last = None
        this_label_list = None
        for i, pair_data in enumerate(self.data):
            so = (pair_data["input_s"], pair_data["input_o"])
            if so == so_last:
                different_keys = list(k for k in pair_data.keys() if pair_data[k] != pair_data_last[k] and k != "label")
                if different_keys:
                    print(f"pair {so} at {i} has differing contiguous items:")
                    print(so, so_last)
                    print(different_keys)
                    raise ValueError(f"pair {so} at {i} has differing contiguous items!")
                this_label_list.append(pair_data['label'])
            else:  # so_last is None or so != so_last:
                assert so not in so_seen, f"pair {so} at {i} seen non-contiguously!"
                if so_last is not None:
                    pair_data_last['label'] = this_label_list
                    new_train_data.append(pair_data_last)
                this_label_list = [pair_data['label']]
                so_last = so
                so_seen.add(so)
                pair_data_last = pair_data
        pair_data_last['label'] = this_label_list
        new_train_data.append(pair_data_last)
        self.data = new_train_data


# TODO: refactor OpenKiReverbDataReader as passed data reader, which means all references to it's variables need to be
#  either copied to self or derefernced at each read. The following classes would then be generic
#  and moved to OpenKI_Data.py
class OpenKIReverbTrainDataReaderNegPairs(OpenKITrainDataReaderNegPairs, OpenKiReverbDataReader):
    pass


class OpenKIReverbTrainDataReader(OpenKITrainDataReader, OpenKiReverbDataReader):
    pass


class OpenKIReverbEvalPerRelationDataReader(OpenKiReverbDataReader, OpenKiEvalPerRelationDataHandler):
    def __init__(self, reverb_folder, data_split, device, eval_top_n_rels=None, top_relations=None, **kwargs):
        """
        Load evaluation data and set up for OpenKI evaluation.
        :param reverb_folder: see OpenKiReverbDataReader
        :param data_split: see OpenKiReverbDataReader
        :param device: see OpenKiReverbDataReader
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param top_relations: top_relations as calculated in train data, to use here (supersedes eval_top_n_relations)
        """
        super().__init__(reverb_folder, data_split, device, eval_top_n_rels if top_relations is None else -1, **kwargs)
        if top_relations is not None:
            self.top_relations = top_relations  # these should be from the train data
        self.relation_indexed_data = tuple(
            tuple(td[0] for td in self.kb_only_triple_data if td[0][2] == i)  # td[0] is the triple
            for i in self.top_relations
        )


class OpenKiReverbEvalPerPairDataReader(OpenKiReverbDataReader, OpenKiEvalPerPairDataHandler):
    def __init__(self, reverb_folder, data_split, device, **kwargs):
        super().__init__(reverb_folder, data_split, device, **kwargs)
        entity_pairs = []
        labels = []
        top_relations = set(map(int, self.top_relations))
        for i, item in enumerate(self.data):
            ok_labels = [rel - self.num_dummy_relations for rel in item['label'] if rel in top_relations]
            entity_pairs.append((item['input_s'], item['input_o']))
            # TODO: is it beneficial to have a tuple of 5k short tensors here?
            labels.append(torch.tensor(ok_labels, dtype=torch.long, device=device))
        self.entity_pairs = torch.tensor(entity_pairs, dtype=torch.long, device=device)
        self.labels = tuple(labels)  # labels are indices in the list of kb rels (NOT in self.relations!!)
