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
import pickle

import torch
import json
from pathlib import Path
from collections import defaultdict

from OpenKI.OpenKI_Data import OpenKiGraphDataHandler, OpenKiEvalPerRelationDataHandler, \
    OpenKiEvalPerPairDataHandler, OpenKITrainDataReaderNegPairs, OpenKITrainDataReader

DUMMY_RELATION = "<DUMMY_RELATION>"
NO_RELATION = "no_relation"
ENTITIES_FILE = "entities.pickle"
RELATIONS_FILE = "relations.pickle"
TEXTS_FILE = "texts.pickle"


class OpenKiTACREDDataReader(OpenKiGraphDataHandler):
    num_dummy_relations = 0  # I haven't set dummy relations here, as we don't use them

    def __init__(self, tacred_folder, data_split, device, eval_top_n_rels=None, use_dev_preds=False, variants=[],
                 enatilments_to=None, untyped_entailments=False,
                 **kwargs):
        """
        Base class for data readers for TACRED data.
        :param tacred_folder: Folder containing reverb data from openKI paper
        :param data_split: Name of data split: "train", "dev" or "test"
        :param device: torch device we're using
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param use_dev_preds: Includes dev/test pairs and their OpenIE relations (KB relations and s/o-neighbours
                removed). Use for e-model, otherwise NOT RECOMMENDED!!
        """
        super().__init__(**kwargs)
        folder = Path(tacred_folder)
        # self.entity_data = []
        self.device = device
        assert data_split in ("dev", "test", "train")
        self.data_folder = data_split

        # self.s_or_o_neighbours = None         # list (by entity index) of pairs (subj/obj) of neighbour tensors
        # self.relation_and_text_lists_by_e_pair = None  # lists of relation indices, indexed by pairs of entity indices
        # self.relations_by_e_pair = None       # tensor of relation indices, indexed by pairs of entity indices

        # self.entities = None                  # list of entity strings, index = entity index
        # self.relations = None                 # list of relation strings, index = relation index. "<MASK_R>" "<PLACEHOLDER_R>" at 0,1 resp.
        # self.relation_texts = None            # ... ideally, this should be a separate list of
                                              # (text_id, [entity_instances]), but we need to store neighbours as
                                              # (rel_id, txt_id, entity_instance_id)

        # read the global files for entity, relation and text indices
        # (created with extract_tacred_entities_and_relations.py)
        with open(folder / ENTITIES_FILE, "rb") as pickle_file:
            self.entities = pickle.load(pickle_file)
        with open(folder / RELATIONS_FILE, "rb") as pickle_file:
            self.relations = pickle.load(pickle_file)
        with open(folder / TEXTS_FILE, "rb") as pickle_file:
            self.relation_texts = pickle.load(pickle_file)

        self.data = {}  # raw TACRED data, indexed by tacred id
        self.relation_and_text_lists_by_e_pair = defaultdict(list)   # indexed by pairs of entity indexes,
                                                            # sorted lists of relation indexes
        self.relations_by_e_pair = {}
        self.all_triple_data = []

        o_neighbours = defaultdict(list)
        s_neighbours = defaultdict(list)
        seen_so_neighbours = set()
        seen_s_neighbours = set()
        seen_o_neighbours = set()
        seen_s = set()
        seen_o = set()

        with open(folder / f"{data_split}.json") as pickle_file:
            for tacred_info in json.load(pickle_file):
                # add raw info to tacred_id indexed dict `data`
                tacred_id = tacred_info["id"]
                if tacred_id not in self.data:
                    self.data[tacred_id] = tacred_info
                relation = tacred_info["relation"]
                relation_info = self.relations[relation]
                relation_id = relation_info["index"]

                # get the sentence
                text = ' '.join(tacred_info["token"])
                text_id = self.relation_texts[text]["index"]
                relation_and_text = (relation_id, text_id)

                # get subject/object entities as strings + types
                subj = (' '.join(tacred_info["token"][int(tacred_info["subj_start"]):int(tacred_info["subj_end"]) + 1]),
                        tacred_info["subj_type"])
                obj = (' '.join(tacred_info["token"][int(tacred_info["obj_start"]):int(tacred_info["obj_end"]) + 1]),
                       tacred_info["obj_type"])

                subj_id = self.entities[subj]["index"]
                obj_id = self.entities[obj]["index"]

                s_neighbours[subj_id].append(relation_and_text)
                o_neighbours[obj_id].append(relation_and_text)
                # NOTE: so_neighbours are stored in self.relation_and_text_lists_by_e_pair for historical reasons
                self.relation_and_text_lists_by_e_pair[(subj_id, obj_id)].append(relation_and_text)

                # TODO: check that the below makes sense...
                seen_so_neighbours.add(relation_id)
                seen_s_neighbours.add(relation_id)
                seen_o_neighbours.add(relation_id)
                seen_s.add(subj_id)
                seen_o.add(obj_id)

        self.relation_and_text_lists_by_e_pair.default_factory = None
        s_neighbours.default_factory = None
        o_neighbours.default_factory = None

        self.relation_lists_by_e_pair = {}
        for pair, rel_list in self.relation_and_text_lists_by_e_pair.items():
            rel_list.sort()
            self.relation_lists_by_e_pair[pair] = sorted(set(rel_id for rel_id, _ in rel_list))
            self.relations_by_e_pair[pair] = \
                torch.tensor(rel_list, dtype=torch.long, requires_grad=False, device=device)
        self.entity_pair_list = list(self.relation_and_text_lists_by_e_pair.keys())

        empty_tensor = torch.tensor([], dtype=torch.long, requires_grad=False, device=device)
        self.s_or_o_neighbours = [[empty_tensor, empty_tensor] for _ in self.entities]
        for s_or_o, neighbour_list in ((0, s_neighbours), (1, o_neighbours)):
            for entity, neighbours in neighbour_list.items():
                self.s_or_o_neighbours[entity][s_or_o] = \
                    torch.tensor(neighbours, dtype=torch.long, requires_grad=False, device=device)

        # if data_split == "train":  # TACRED doesn't differentiate preds and rels, so this makes no sense...
        #     if use_dev_preds:
        #         for split in ("test", "dev"):
        #             with open(folder / f"{split}_data.json") as f:
        #                 split_data = json.load(f)
        #                 for pair_info in split_data:
        #                     if pair_info['input_p']:
        #                         pair_info['label'] = []
        #                         # pair_info['input_s_neighbor'] = None
        #                         # pair_info['input_o_neighbor'] = None
        #                         self.data.append(pair_info)

        self.num_relations = len(self.relations)
        self.num_entities = len(self.entities)
        self.first_predicate_index = len(self.relations)
        self.num_kb_relations = self.first_predicate_index - self.num_dummy_relations

        # A few persistent tensors to avoid allocating them each time
        self.relations_mask = torch.zeros((self.num_relations,), dtype=torch.bool, device=self.device,
                                          requires_grad=False)
        self.empty_index_tensor = torch.tensor([], dtype=torch.long, device=self.device, requires_grad=False)

        # # Set up indexes for (s,o) pairs, s and o for quick access to data
        # seen_so_neighbours = []
        # seen_s = []
        # seen_o = []
        # # seen_relations = []
        # # seen_predicates = []
        # for data_index, pair_info in enumerate(self.data):
        #     so = (pair_info["input_s"], pair_info["input_o"])
        #     s, o = so
        #     these_predicates = set(pair_info['input_p'])
        #     these_kb_relations = set(pair_info['label'])
        #     # seen_relations.extend(these_kb_relations)
        #     # seen_predicates.extend(these_predicates)
        #     these_relations = sorted(list(these_kb_relations)+list(these_predicates))
        #     seen_so_neighbours.extend(these_relations)
        #     if so in self.entity_pair_index:  # NOTE: this doesn't actually happen in data from generate_openki.py!
        #         # these_relations = pair_info['input_p'] + pair_info['label']
        #         if len(these_relations) == 0 and len(these_predicates) == 0:
        #             continue  # TODO: does this ever happen??
        #         these_relations.extend(self.relation_and_text_lists_by_e_pair[so])  # add in those already found
        #         these_relations = sorted(set(these_relations))
        #     else:
        #         seen_s.append(s)
        #         seen_o.append(o)

        self.max_neighbour_len = max(max(ts.shape[-1], to.shape[-1]) for ts, to in self.s_or_o_neighbours)
        self.expected_neighbour_len = sum(ts.shape[-1] + to.shape[-1] for ts, to in self.s_or_o_neighbours) \
                                      / len(self.s_or_o_neighbours) / 2
        # create boolean masks of seen neighbours
        self.seen_s_or_o_neighbours = torch.zeros((2, self.num_relations), dtype=torch.bool, requires_grad=False,
                                                  device=device)
        self.seen_so_neighbours = torch.zeros((self.num_relations,), dtype=torch.bool, requires_grad=False,
                                              device=device)
        self.seen_entities = torch.zeros((2, len(self.entities)), dtype=torch.bool, requires_grad=False,
                                         device=device)

        self.seen_s_or_o_neighbours[0, list(seen_s_neighbours)] = True
        self.seen_s_or_o_neighbours[1, list(seen_o_neighbours)] = True
        self.seen_so_neighbours[list(seen_so_neighbours)] = True
        self.seen_entities[0, list(seen_s)] = True
        self.seen_entities[1, list(seen_o)] = True

        self.all_triple_data = []
        for so, relations_and_texts in self.relation_and_text_lists_by_e_pair.items():
            s, o = so
            self.all_triple_data.extend(
                ((s, o, rel, txt),
                 self.s_or_o_neighbours[s][0],
                 self.s_or_o_neighbours[o][1],
                 relations_and_texts) for rel, txt in relations_and_texts
            )

        self.kb_only_triple_data = self.all_triple_data  # for TACRED, all are kb relations
        # NOTE: we set self.triple_data in OpenKITrainDataReader.__init__() etc...

        self.top_relations = None
        # TODO: hardcode top_relations to all of them, since it's only 58 anyway??
        self.set_top_relations(OpenKiTACREDDataReader, self, eval_top_n_rels)


class OpenKITACREDTrainDataReaderNegPairs(OpenKITrainDataReaderNegPairs, OpenKiTACREDDataReader):
    pass


class OpenKITACREDTrainDataReader(OpenKITrainDataReader, OpenKiTACREDDataReader):
    pass


class OpenKITACREDEvalPerRelationDataReader(OpenKiTACREDDataReader, OpenKiEvalPerRelationDataHandler):
    def __init__(self, tacred_folder, data_split, device, eval_top_n_rels=None, top_relations=None, **kwargs):
        """
        Load evaluation data and set up for OpenKI evaluation.
        :param tacred_folder: see OpenKiTACREDDataReader
        :param data_split: see OpenKiTACREDDataReader
        :param device: see OpenKiTACREDDataReader
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param top_relations: top_relations as calculated in train data, to use here (supersedes eval_top_n_relations)
        """
        super().__init__(tacred_folder, data_split, device, eval_top_n_rels if top_relations is None else -1, **kwargs)
        if top_relations is not None:
            self.top_relations = top_relations  # these should be from the train data
        self.relation_indexed_data = tuple(
            tuple(td[0] for td in self.kb_only_triple_data if td[0][2] == i)  # td[0] is the triple
            for i in self.top_relations
        )


class OpenKiTACREDEvalPerPairDataReader(OpenKiTACREDDataReader, OpenKiEvalPerPairDataHandler):
    def __init__(self, tacred_folder, data_split, device, **kwargs):
        super().__init__(tacred_folder, data_split, device, **kwargs)
        self.entity_pairs = torch.tensor(self.entity_pair_list, dtype=torch.long, device=self.device)
        self.labels = [self.relations_by_e_pair[pair] for pair in self.entity_pair_list]
