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
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data import dataloader
from torch.utils.data import Dataset
from OpenKI import logger
from collections import Counter


class OpenKiGraphDataHandler:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_s_or_o_neighbour_copies = {}
        self.device_relations_by_e_pair_copies = {}
        # Set up dummy variables, mostly for pycharm name checking when it's unclear which subclass is being used
        if getattr(self, "data_folder", None) is None:
            self.data_folder = None
        if getattr(self, "entity_pair_list", None) is None:
            self.entity_pair_list = None
        if getattr(self, "data", None) is None:
            self.data = None  # not apparently used outside of subclasses
        if getattr(self, "device", None) is None:
            self.device = None
        if getattr(self, "data_split", None) is None:
            self.data_split = None
        if getattr(self, "entities", None) is None:
            self.entities = None
        if getattr(self, "relations", None) is None:
            self.relations = None
        if getattr(self, "relation_texts", None) is None:
            self.relation_texts = None
        if getattr(self, "relations_mask", None) is None:
            self.relations_mask = None
        # if getattr(self, "top_relations", None) is None:
        #     self.top_relations = None  # this is now a class variable, settable only by train data
        if getattr(self, "relations_by_e_pair", None) is None:
            self.relations_by_e_pair = None       # tensor of relation indices, indexed by pairs of entity indices
        if getattr(self, "relation_lists_by_e_pair", None) is None:
            self.relation_lists_by_e_pair = None  # lists of relation indices, indexed by pairs of entity indices
        if getattr(self, "relation_indexed_data", None) is None:
            self.relation_indexed_data = None     # setup in EvalPerRelationDataReader: eval data by kb relation indices
        if getattr(self, "num_dummy_relations", None) is None:
            self.num_dummy_relations = None
        if getattr(self, "first_predicate_index", None) is None:
            self.first_predicate_index = None
        if getattr(self, "num_kb_relations", None) is None:
            self.num_kb_relations = None
        if getattr(self, "num_relations", None) is None:
            self.num_relations = None
        if getattr(self, "num_entities", None) is None:
            self.num_entities = None
        if getattr(self, "seen_entities", None) is None:
            self.seen_entities = None
        if getattr(self, "seen_s_or_o_neighbours", None) is None:
            self.seen_s_or_o_neighbours = None
        if getattr(self, "seen_so_neighbours", None) is None:
            self.seen_so_neighbours = None
        if getattr(self, "s_or_o_neighbours", None) is None:
            self.s_or_o_neighbours = None
        if getattr(self, "max_neighbour_len", None) is None:
            self.max_neighbour_len = 0
        if getattr(self, "empty_index_tensor", None) is None:
            self.empty_index_tensor = None
        if getattr(self, "kb_only_triple_data", None) is None:
            self.kb_only_triple_data = None
        if getattr(self, "triple_data", None) is None:
            self.triple_data = None
        if getattr(self, "all_triple_data", None) is None:
            self.all_triple_data = None
        if getattr(self, "top_relations", None) is None:
            self.top_relations = None

    # methods for batching
    def get_s_or_o_neighbours(self, entity_indices, s_or_o, only_predicates=False, seen_s_or_o_relation_mask=None):
        # TODO: add argument 'neighbour_list=None' so we don't have to look it up if we've already got it...
        """
        returns the list of subject/object neighbours (predicates/relations) for each entity index
        :param entity_indices: iterable of m entity indices
        :param s_or_o: are they subjects (0) or objects (1)?
        :param only_predicates: If True, do not return indices for KB relations (used for evaluation)
        :param seen_s_or_o_relation_mask: Boolean mask indicating which relations have been seen as subject or object
                neighbours.
        :return: list of m shape (l,1) or (l,2) int tensors with indices of s- or o- relation neighbours for each entity
                 index. The second dimension is 2 if predicate and text indices are provided.
        """
        # when using multiple GPUs, we need to manually copy neighbour lists to the appropriate device
        if entity_indices.device != self.device:
            s_or_o_neighbours = self.device_s_or_o_neighbour_copies.get(entity_indices.device, None)
            if s_or_o_neighbours is None:
                s_or_o_neighbours = [[nb.to(entity_indices.device) for nb in s_or_o_nb]
                                     for s_or_o_nb in self.s_or_o_neighbours]  # make a copy on this gpu/device
                self.device_s_or_o_neighbour_copies[entity_indices.device] = s_or_o_neighbours
            neighbours = [s_or_o_neighbours[e][s_or_o] for e in entity_indices]
        else:
            neighbours = [self.s_or_o_neighbours[e][s_or_o] for e in entity_indices]
        if only_predicates:
            self.relations_mask.fill_(True)
            self.relations_mask[:self.first_predicate_index] = False
            if seen_s_or_o_relation_mask is not None:
                self.relations_mask[~seen_s_or_o_relation_mask] = False
            for i, (nb, e) in enumerate(zip(neighbours, entity_indices)):
                neighbours[i] = nb[self.relations_mask[nb]]  # nb bool masked with (mask index masked with nb)
        elif seen_s_or_o_relation_mask is not None:
            for i, (nb, e) in enumerate(zip(neighbours, entity_indices)):
                neighbours[i] = nb[seen_s_or_o_relation_mask[nb]]  # nb bool masked with (seen index masked with nb)
        return neighbours

    def get_so_relations(self, triples, only_predicates=False, seen_relation_mask=None):
        """
        :param triples: iterable of m triples: (e1_idx, e2_idx, relation_idx)
        :param only_predicates: If True, do not return indices for KB relations (used for evaluation)
        :param seen_relation_mask: an (#rels,) bool tensor indicating rels seen in training (used for evaluation)
        :return: list of m shape (l,) tensors with indices of other relations between each entity pair
        """
        # neighbours = []
        # for i, tr in enumerate(triples):
        #     self.relations_mask.fill_(False)
        #     self.relations_mask[self.relations_by_e_pair.get(tuple(tr[:2].tolist()), self.empty_index_tensor)] = True
        #     self.relations_mask[tr[2]] = False  # remove the query relation
        #     if seen_relation_mask is not None:
        #         mask = self.relations_mask & seen_relation_mask
        #     else:
        #         mask = self.relations_mask
        # !!! neighbours.append(torch.nonzero(mask).view(-1))   NOTE: this line gobbled cuda memory!!!
        # return neighbours

        # TODO: finish setting up so_relations for multiple gpus - need to copy relevant (lists/dicts of) tensors to
        #       the gpu that triples is in, then use the copied triples henceforth. The code below is an unfinished
        #       adaptation of similar code for get_s_or_o_neighbours() above.
        # if triples.device != self.device:
        #     neighbours = [self.relations_by_e_pair.get(tuple(tr[:2].tolist()), self.empty_index_tensor)
        #                   for tr in triples]
        #     s_or_o_neighbours = self.device_s_or_o_neighbour_copies.get(triples.device, None)
        #     if s_or_o_neighbours is None:
        #         s_or_o_neighbours = self.seen_s_or_o_neighbours.to(triples.device)
        #         self.device_s_or_o_neighbour_copies[triples.device] = s_or_o_neighbours
        # else:
        # when using multiple GPUs, we need to manually copy neighbour lists to the appropriate device
        if triples.device != self.device:
            relations_by_e_pair = self.device_relations_by_e_pair_copies.get(triples.device, None)
            if relations_by_e_pair is None:
                relations_by_e_pair = {pair: so_nb.to(triples.device)
                                       for pair, so_nb in self.relations_by_e_pair.items()}  # make a copy on this gpu/device
                self.device_relations_by_e_pair_copies[triples.device] = relations_by_e_pair
            neighbours = [relations_by_e_pair.get(tuple(tr[:2].tolist()), self.empty_index_tensor) for tr in
                          triples]
        else:
            neighbours = [self.relations_by_e_pair.get(tuple(tr[:2].tolist()), self.empty_index_tensor) for tr in
                          triples]

        self.relations_mask.fill_(True)
        if only_predicates:
            self.relations_mask[:self.first_predicate_index] = False
        if seen_relation_mask is not None:
            # TODO: we need to distinguish masking relation embeds from relation texts when they're independent
            self.relations_mask[~seen_relation_mask] = False
        for i, (nb, tr) in enumerate(zip(neighbours, triples)):
            this_tr_not_masked = self.relations_mask[tr[2]]
            if this_tr_not_masked:                  # don't include this relation in the list
                self.relations_mask[tr[2]] = False  # only do the cuda op if it changes anything
            neighbours[i] = nb[self.relations_mask[nb[:, 0]]]  # nb bool masked with (seen index masked with nb)
            if this_tr_not_masked:
                self.relations_mask[tr[2]] = True
        return neighbours

    @classmethod
    def set_top_relations(cls, base_cls, instance, top_n_rels):
        if instance.data_folder == "train":
            if top_n_rels is None:
                base_cls.top_relations = torch.arange(instance.num_dummy_relations, instance.first_predicate_index,
                                                      device=instance.device, dtype=torch.long)
                logger.info(f"loaded {instance.data_folder} with all {len(base_cls.top_relations)} top relations.")
            elif top_n_rels == -1:
                logger.info(f"loaded {instance.data_folder} without setting top_relations.")
            else:
                relation_counts = Counter(t[0][2] for t in instance.kb_only_triple_data)
                base_cls.top_relations = torch.tensor(tuple(mc[0] for mc in relation_counts.most_common(top_n_rels)),
                                                  device=instance.device, dtype=torch.long)
                logger.info(f"loaded {instance.data_folder} with {len(base_cls.top_relations)} top relations.")
                sorted_top_rel_counts = relation_counts.most_common(top_n_rels)
                top_relation = sorted_top_rel_counts[0]
                bottom_relation = sorted_top_rel_counts[-1]
                logger.info(f"Retained eval relation frequencies from {bottom_relation[0]} ({bottom_relation[1]} times) to "
                            f"{top_relation[0]} ({top_relation[1]} times)")
        instance.top_relations = base_cls.top_relations

    # def get_entity_texts(self, vocab):
    #     """
    #     Generator of (index, words) of entities
    #     TODO: or do we just iterate from zero?
    #     :return:
    #     """
    #     raise NotImplementedError()
    #
    # def get_relation_texts(self):
    #     raise NotImplementedError()


class OpenKiGraphTrainDataHandler(OpenKiGraphDataHandler, Dataset):
    # methods for returning relations between/to/from entities
    def __getitem__(self, indices):
        """
        Get triples for these indices
        :param indices: Iterable of integers from 0 to len(self.triple_data)
        :return: iterator of tuples of ( (s, o, rel) triples, s_neighbours, o_neighbours, so_relations )
             or: iterator of tuples of ( (s, o, rel, txt) quads, s_neighbours, o_neighbours, so_relations )
        """
        # self.triple_data is a list of tuples
        # TODO: SPEEDUP: can this be done with tensor indexing on gpu?
        return zip(*(self.triple_data[triple_index] for triple_index in indices))

    def __len__(self):
        return len(self.triple_data)

    def get_negative_args(self, triples, num_neg_samples, check_not_positive=True, negative_strategy=None):
        """
        Returns -ve examples for each triple. These have the same two entities but a different relation.
        :param triples: (s, o, p) triples iterable for which -ve samples are to be drawn
        :param num_neg_samples: number of negative examples to find for each positive
        :param check_not_positive: whether to check that each negative sample is not a positive one
        :param negative_strategy: one of OpenKiDataset.negative_strategies. Currenlty only "uniform"
        :return: a list of num_neg_samples negative samples for each entity pair in triples
        """
        raise NotImplementedError()


class OpenKiEvalPerRelationDataHandler(Dataset, OpenKiGraphDataHandler):
    def __getitem__(self, indices):
        return tuple(self.relation_indexed_data[i] for i in indices)

    def __len__(self):
        # return self.first_predicate_index - self.num_dummy_relations
        return len(self.relation_indexed_data)


class OpenKiEvalPerPairDataHandler(Dataset, OpenKiGraphDataHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self, "entity_pairs", None) is None:
            self.entity_pairs = None
        if getattr(self, "labels", None) is None:
            self.labels = None

    def __getitem__(self, indices):
        """
        :param indices: Entity pair indexes (each pair has a unique index) of length batch_size
        :return: tensor of pairs of entity indices (batch_size, 2), tuple of batch_size (n,) tensors of relation indexes
        """
        # TODO: SPEEDUP: can this be done with tensor indexing on gpu?
        return self.entity_pairs[indices], tuple(self.labels[i] for i in indices)
        # NOTE: self.labels should contain 0-based indices
        # return tuple(self.get_so_relations(self.entity_pair_list[i]) for i in indices)
        # self.entity_pair_list = tuple(entity_pair_index.keys())

    def __len__(self):
        return len(self.entity_pairs)


class OpenKITrainDataReaderNegPairs(OpenKiGraphTrainDataHandler):
    negative_strategies = [
        "uniform",
        "none"
    ]

    def __init__(self, data_folder, data_split, device, openie_as_pos_samples=False,
                 **kwargs):
        """
        Train data reader when using ranking pairs in the loss function (aligned with default eval). NOTE: the OpenKI
        paper does NOT do this!
        :param data_folder: Folder containing NYT data from openKI paper
        :param data_split: Name of data split: "train", "dev" or "test"
        :param device: torch device we're using
        :param eval_top_n_rels: Integer. If provided, cls.top_relations is calculated from most frequent KB relations.
        :param use_dev_preds: Includes dev/test pairs and their OpenIE relations (KB relations and s/o-neighbours
                removed). Use for e-model, otherwise NOT RECOMMENDED!!
        :param openie_as_pos_samples: OpenIE predicates included in positive samples during training (default False)

        """
        super().__init__(data_folder, data_split, device, **kwargs)
        if openie_as_pos_samples:
            self.triple_data = self.all_triple_data
        else:
            self.triple_data = self.kb_only_triple_data
        # self.max_neg_predicate = self.first_predicate_index if openie_as_neg_samples else self.num_relations
        # Nothing more to do, as OpenKiNYTDataReader sets up self.triples
        # used in OpenKiGraphTrainDataHandler.__getitem__() and .__len__()

        self.kb_neg_pairs_index = [
            torch.tensor([idx for idx, (_, _, _, so_n) in enumerate(self.triple_data) if kb_idx in so_n],
                         dtype=torch.long, requires_grad=False)
            for kb_idx in range(self.first_predicate_index)
        ]  # for each kb predicate, a 1-d tensor of indexes in self.triples

        self.triples = torch.tensor([td[0] for td in self.triple_data], dtype=torch.long,
                                    device=device, requires_grad=False)  # shape (#triples, 3)
        self.neg_pair_mask = torch.ones((self.triples.shape[0],), dtype=torch.bool,
                                        requires_grad=False, device=device)  # shape (#triples, ), set each batch

    def get_negative_args(self, triples, num_neg_samples, check_not_positive=True, negative_strategy="uniform"):
        """
        Returns -ve examples for each triple. These have the same relation but a different entity pair.
        :param triples: (s, o, p) triples iterable for which -ve samples are to be drawn
        :param num_neg_samples: number of negative examples to find for each positive
        :param check_not_positive: whether to check that each negative sample is not a positive one
        :param negative_strategy: one of OpenKiDataset.negative_strategies. Currenlty only "uniform"
        :return: a list of num_neg_samples negative samples for each entity pair in triples
        """
        assert negative_strategy in self.negative_strategies, \
            f'only negative strategies "{",".join(self.negative_strategies)}" implemented!'

        # TODO: implement single entity negative samples... need self.kb_neg_subj_index and ...obj_index (and in this
        #  case check_not_positive probably not implemented, as these indices will ensure s or o is the same).
        if negative_strategy == self.negative_strategies[0]:  # "uniform"
            # for each triple (s, o, p), get num_neg triples (s, o, q) which are not in the data
            negative_samples = []
            # for each batch item, make a list of -ve pair indexes and sample from them.
            for _, _, p in triples:
                if check_not_positive:
                    self.neg_pair_mask.fill_(True)
                    self.neg_pair_mask[self.kb_neg_pairs_index[p]] = False  # shape (#triples, )
                    neg_triples = self.triples[self.neg_pair_mask]  # shape (#neg_triples, 3)
                else:
                    neg_triples = self.triples  # shape (#triples, 3)
                perm = torch.randperm(neg_triples.size(0))  # shape (#neg_triples, )
                idx = perm[:num_neg_samples]  # shape (num_neg_samples, )
                neg_sample = neg_triples[idx, :].clone()  # shape (num_neg_samples, 3)
                neg_sample[:, 2] = p
                negative_samples.append(neg_sample)
            # TODO: we could have a (batch_size*num_neg_samples, 3) tensor and write to it each time to avoid creating
            #       tensors.
            return torch.stack(negative_samples)

        elif negative_strategy == self.negative_strategies[1]:  # "none"
            # this is used in evaluation with so pairs as queries
            return []
        else:
            raise NotImplementedError(f"Negative strategy {negative_strategy} not implemented! "
                                      f"Try one of {' '.join(self.negative_strategies)}")


class OpenKITrainDataReader(OpenKiGraphTrainDataHandler):
    negative_strategies = [
        "uniform",
        "none"
    ]

    def __init__(self, *args, openie_as_pos_samples=False, **kwargs):
        """
        Train data reader when using ranking pairs in the loss function (aligned with default eval). NOTE: the OpenKI
        paper does NOT do this!
        :param data_folder: Folder containing NYT data from openKI paper
        :param data_split: Name of data split: "train", "dev" or "test"
        :param device: torch device we're using
        :param eval_top_n_rels: Integer. If provided, self.top_relations is calculated from most frequent KB relations.
        :param use_dev_preds: Includes dev/test pairs and their OpenIE relations (KB relations and s/o-neighbours
                removed). Use for e-model, otherwise NOT RECOMMENDED!!
        :param openie_as_neg_samples: OpenIE predicates included in negative samples during training (default False)
        :param openie_as_pos_samples: OpenIE predicates included in positive samples during training (default False)
        """
        super().__init__(*args, **kwargs)
        if openie_as_pos_samples:
            self.triple_data = self.all_triple_data
        else:
            self.triple_data = self.kb_only_triple_data
        self.max_neg_predicate = self.num_relations
        # Nothing more to do, as OpenKiNYTDataReader sets up self.triples
        # used in OpenKiGraphTrainDataHandler.__getitem__() and .__len__()

    def get_negative_args(self, triples, num_neg_samples, check_not_positive=True, negative_strategy="uniform"):
        """
        Returns -ve examples for each triple. These have the same two entities but a different relation.
        :param triples: (s, o, p) triples iterable for which -ve samples are to be drawn
        :param num_neg_samples: number of negative examples to find for each positive
        :param check_not_positive: whether to check that each negative sample is not a positive one
        :param negative_strategy: one of OpenKiDataset.negative_strategies. Currenlty only "uniform"
        :return: a list of num_neg_samples negative samples for each entity pair in triples
        """
        assert negative_strategy in self.negative_strategies,  \
            f'only negative strategies "{",".join(self.negative_strategies)}" implemented!'

        # TODO: Strategies for text indices for -ve samples when swapping predicates:
        #       - simply give the -ve predicate an empty text (index 0): problematic as the model can learn to use
        #           non-zero text embeddings as negative examples (except for relations, but that it can do also)
        #       - instead of a new predicate, choose a new predicate+text instance (from those we have). But the model
        #           can learn that the entities are not mentioned in the text (maybe?)
        #       - instead of a new predicate, choose a new predicate+text instance (from those we have) AND put the
        #           current entities in place of the original entities. Still problematic AND we'd have to run ALL -ve
        #           examples through BERT!
        #       - swap out entities instead of predicates... We still have to run BERT on all -ve examples, unless we
        #           maintain a pool of precomputed examples? Run a thread generating them on another gpu?
        #       - don't use predicate texts AT ALL! The best text based models previously used just entity info...
        texts_in_triples = len(triples[0]) == 4
        assert texts_in_triples or len(triples[0]) == 3
        if negative_strategy == self.negative_strategies[0]:  # "uniform"
            # for each triple (s, o, p), get num_neg triples (s, o, q) which are not in the data
            if check_not_positive:
                # for each batch item, make a list of -ve relation indexes and sample from them.
                negative_samples = []
                for triple in triples:
                    s, o = triple[:2]
                    # this_pred = triple[2]
                    r0 = self.num_dummy_relations
                    negative_relations = []
                    for r1 in self.relation_lists_by_e_pair[(s, o)]:
                        # if texts_in_triples:  # if relation lists by e pair also contains text indices...
                        #     r1 = r1[0]
                        #     if r0 == r1:
                        #         continue  # this is when a predicate has several associated texts, we only take the 1st
                        negative_relations.extend(range(r0, r1))
                        r0 = r1 + 1
                    if r0 < self.num_relations:
                        negative_relations.extend(range(r0, self.num_relations))
                    # TODO: these need to be 3- or 4- tuples depending on our data!
                    #       Indeed, relation_lists_by_e_pair needs to have text indices in it as well, else we need some
                    #       other way of accessing associated texts to the negative predicates. This also means we can't
                    #       use entity spans with negative examples! We could alternatively sample
                    if texts_in_triples:
                        # TODO: implemented "give the -ve predicate an empty text" from the options above...
                        negative_samples.append([(s, o, p, 0) for p in random.sample(negative_relations, num_neg_samples)])
                    else:
                        negative_samples.append([(s, o, p) for p in random.sample(negative_relations, num_neg_samples)])
            else:
                if texts_in_triples:
                    # TODO: implemented "give the -ve predicate an empty text" from the options above...
                    negative_samples = [[[triple[0], triple[1], p, 0] for p in
                                         torch.randint(self.num_dummy_relations, self.max_neg_predicate,
                                                       (num_neg_samples,))]
                                        for triple in triples]
                else:
                    negative_samples = [[[triple[0], triple[1], p] for p in
                                         torch.randint(self.num_dummy_relations, self.max_neg_predicate,
                                                       (num_neg_samples,))]
                                        for triple in triples]
            return negative_samples
        elif negative_strategy == self.negative_strategies[1]:  # "none"
            # this is used in evaluation with so pairs as queries
            return []
        else:
            raise NotImplementedError(f"Negative strategy {negative_strategy} not implemented! "
                                      f"Try one of {' '.join(self.negative_strategies)}")


# Now for the DataLoaders that use the above data handlers
class OpenKIMemoryIndexedTrainDataLoader(DataLoader):
    def __init__(self, dataset: OpenKiGraphTrainDataHandler, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=dataloader.default_collate, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None, negative_sample_rate=1, negative_strategy=None,
                 check_not_positive=True):
        """
        A torch.utils.data.DataLoader that keeps data and it's indices in memory (useful if the data is not too large
        in memory).
        :param negative_sample_rate: number of negative samples per positive sample
        :param negative_strategy: a string passed to the dataset's negative sample method
        :param check_not_positive: bool, whether to check if negative samples are indeed not positive
        """
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory,
                         drop_last, timeout, worker_init_fn)
        # self.dataset = dataset
        assert num_workers == 0, "This data loader uses in-memory indexing and needs only one worker"
        # TODO: how hard is it to use the pytorch threaded batch sampler here?
        self.sample_iter = iter(self.batch_sampler)
        self.negative_sample_rate = negative_sample_rate
        self.negative_strategy = negative_strategy
        self.check_not_positive = check_not_positive

        # # batch caches to streamline data access
        # self.triples, self.s_neighbours, self.o_neighbours, self.so_rels = None, None, None, None

    def __iter__(self):
        return self

    def __next__(self):
        """Based on __next__() from torch.utils.data.dataloader._DataLoaderIter, but uses array indexing to obtain data
        and implements pulling negative samples from the datasest (presuming it implements it!)
        :returns tuple of tensors ([positive_e1_e2_p_triples], [[negative_e1_e2_p_triples],])
        """
        indices = next(self.sample_iter)  # list of integers
        # self.triples, self.s_neighbours, self.o_neighbours, self.so_rels = self.dataset[indices]  # these are tuples
        # # ([s,o,p]...), (tensor...),     (tensor...),       (tensor...)
        triples, _, _, _ = self.dataset[indices]  # these are tuples
        # NOTE: the _, _, _ here are actually s_neighbours, o_neighbours, so_neighbours (as predicate indices)

        # indices = np.array(indices)
        neg_batch = self.dataset.get_negative_args(triples, self.negative_sample_rate,
                                                   self.check_not_positive, self.negative_strategy)
        if type(neg_batch) is not torch.Tensor:
            neg_batch = torch.tensor(neg_batch)
        return torch.tensor(triples), neg_batch, indices


class OpenKIMemoryIndexedEvalPerRelationDataLoader(DataLoader):
    def __init__(self, dataset: OpenKiEvalPerRelationDataHandler, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None,
                 num_workers=0, collate_fn=dataloader.default_collate, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        """
        A torch.utils.data.DataLoader that keeps data and it's indices in memory (useful if the data is not too large
        in memory).
        This is used in evaluation with top n predicates as queries in MAP calculation
        :param negative_sample_rate: number of negative samples per positive sample
        """
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, drop_last,
                         timeout, worker_init_fn)
        assert batch_size == 1, "Only one relation at a time implemented: set batch_size to one"
        # self.dataset = dataset
        assert num_workers == 0, "This data loader uses in-memory indexing and needs only one worker"
        self.sample_iter = iter(self.batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        """
        For OpenKI eval data, we iterate over the top n KB relations, yielding a single tensor of triples for each
        KB relation.
        Based on __next__() from torch.utils.data.dataloader._DataLoaderIter, but uses array indexing to obtain data
        and implements pulling negative samples from the datasest (presuming it implements it!)
        :returns tensor [positive_e1_e2_p_triples] of shape (n, 3) where n is the number of entity pairs
        """
        indices = next(self.sample_iter)  # list of integers
        # TODO: to cope with batches of >1 relations, we need to deal with different lengths per relation here
        return (torch.tensor(triple_list) for triple_list in self.dataset[indices])


class OpenKIMemoryIndexedEvalPerPairDataLoader(DataLoader):
    def __init__(self, dataset: OpenKiEvalPerPairDataHandler, batch_size, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=dataloader.default_collate, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        """
        A torch.utils.data.DataLoader that keeps data and it's indices in memory (useful if the data is not too large
        in memory).
        This is used in evaluation with top n predicates as queries in MAP calculation
        :param negative_sample_rate: number of negative samples per positive sample
        """
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, drop_last,
                         timeout, worker_init_fn)
        assert num_workers == 0, "This data loader uses in-memory indexing and needs only one worker"
        self.sample_iter = iter(self.batch_sampler)

    def __iter__(self):
        return self

    def __next__(self):
        """
        For OpenKI eval data, we iterate over the top n KB relations, yielding a single tensor of triples for each
        KB relation.
        Based on __next__() from torch.utils.data.dataloader._DataLoaderIter, but uses array indexing to obtain data
        and implements pulling negative samples from the datasest (presuming it implements it!)
        :returns tensor [positive_e1_e2_p_triples] of shape (n, 3) where n is the number of entity pairs
        """
        indices = next(self.sample_iter)  # list of integers
        # TODO: to cope with batches of >1 relations, we need to deal with different lengths per relation here
        return self.dataset[indices]
