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
from collections import Counter
from itertools import zip_longest
from typing import Union

from torchnet.meter import mAPMeter
from tqdm import tqdm
from OpenKI import logger
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

from OpenKI.Constants import DEGENERATE_MODEL_MARKER
from OpenKI.NYT_2010_Data import OpenKiNYTDataReader
from OpenKI.OpenKI_Data import OpenKiGraphDataHandler, OpenKIMemoryIndexedEvalPerPairDataLoader, \
    OpenKiEvalPerRelationDataHandler, OpenKIMemoryIndexedEvalPerRelationDataLoader, OpenKiEvalPerPairDataHandler


class OpenKIEvaluator:
    def __init__(self, scorer: nn.Module, eval_data: Union[OpenKiEvalPerPairDataHandler,OpenKiGraphDataHandler],
                 batch_size, cuda, device, only_seen_neighbours, other_evaluator: 'OpenKIMapEvaluator' = None,
                 label="Evaluation"):
        self.scorer = scorer
        self.eval_data = eval_data
        self.batch_size = batch_size
        self.cuda = cuda
        self.device = device
        self.only_seen_neighbours = only_seen_neighbours
        self.label = label

    def evaluate(self, epoch=None):
        raise NotImplementedError()


class OpenKiAucPrEvaluator(OpenKIEvaluator):
    def __init__(self, scorer: nn.Module,
                 eval_data: Union[OpenKIMemoryIndexedEvalPerPairDataLoader,OpenKiGraphDataHandler], batch_size, cuda,
                 device, only_seen_neighbours, other_evaluator: 'OpenKIMapEvaluator' = None, label="Evaluation",
                 t_board: 'SummaryWriter' = None):
        super().__init__(scorer, eval_data, batch_size, cuda, device, only_seen_neighbours, other_evaluator, label)

        if other_evaluator is not None:
            raise NotImplementedError("Multiple evaluators for AUC-PR evaluation not implemented!")
        self.other_evaluator = None
        self.t_board = t_board
        self.data_split = eval_data.data_folder

        self.entity_pair_indices = {pair: i for i, pair in enumerate(eval_data.entity_pair_list)}
        # TODO: data has entity_pair_list, a tuple of entitiy pairs: use it!
        self.num_pairs = len(self.entity_pair_indices)

        # scores_size = eval_data.num_dummy_relations + eval_data.num_kb_relations  # ... is the same as ... (next line)
        # scores_size = eval_data.first_predicate_index  # NOTE: we keep redundant indexes 0 and 1 ... NO!
        # TODO: evaluate on top 50 relations... Oh, but NYT 2010 has only 47 relations anyway, so no difference!!
        #       -

        num_top_relations = eval_data.top_relations.shape[0]
        self.scores = numpy.zeros(dtype=float, shape=(self.num_pairs, num_top_relations))
        self.labels = numpy.zeros(dtype=bool, shape=(self.num_pairs, num_top_relations))
        self.pairs_scored = numpy.zeros(dtype=bool, shape=(self.num_pairs,))

        self.is_top_50 = torch.zeros(size=(eval_data.num_kb_relations,), dtype=torch.bool, device=self.device,
                                     requires_grad=False)
        self.is_top_50[eval_data.top_relations - eval_data.num_dummy_relations] = True

        self.range_to_top_rel_idx = torch.ones(size=(eval_data.num_kb_relations,), dtype=torch.long,
                                               requires_grad=False)
        self.range_to_top_rel_idx[:] = eval_data.num_relations + 1  # dummy illegal value, nan can't be used: int array
        self.range_to_top_rel_idx[eval_data.top_relations - eval_data.num_dummy_relations] = \
            torch.arange(num_top_relations)

        # # NOTE: this is a bit silly: if we do this here, it should be for s_neighbours and o_neighbours also!
        # if self.only_seen_relations:
        #     self.relation_mask = eval_data.seen_so_neighbours[eval_data.num_dummy_relations:
        #                                                       eval_data.num_dummy_relations + scores_size].cpu()
        # else:
        self.relation_mask = numpy.ones(num_top_relations, dtype=bool)  #self.labels[0, :])
        # TODO: self.num_kb_relations isn't actually changed anywyere!
        self.num_kb_relations = self.relation_mask.shape[0]
        self.all_triples = torch.zeros(size=(self.batch_size, num_top_relations, 3),
                                       dtype=torch.long, device=self.device, requires_grad=False)
        # TODO: evaluate on top 50 relations... Oh, but NYT 2010 has only 47 relations anyway, so no difference!!
        self.all_triples[:, :, 2] = eval_data.top_relations

    def evaluate(self, epoch=None):
        """
        Run this evaluators model on it's data, iterating over inferred scores and corresponding labels
        :return: AU-PR value
        """
        with torch.no_grad():
            self.scorer.train(False)
            self.labels[:] = 0  # set all labels to zero
            # TODO: we could setup the labels once on init, then not change them...
            for batch_num, (entity_pairs, relation_indices) in \
                    enumerate(tqdm(OpenKIMemoryIndexedEvalPerPairDataLoader(self.eval_data, batch_size=self.batch_size),
                              desc=self.label)):
                if self.cuda:
                    entity_pairs_cuda = entity_pairs.cuda()  # shape (batch_size, 2)
                    # relation_indices = ... a tuple of tensors (batch_size, (n,) ) where n varies and can be zero. Note they are indices in KB_relations, starting from 0
                else:
                    entity_pairs_cuda = entity_pairs

                batch_size = entity_pairs.shape[0]

                triples = self.all_triples[:batch_size, :, :]
                triples[:, :, :2] = entity_pairs_cuda[:, :].unsqueeze(1)  # shape (batch_size, #relations, 3)
                # here we score all relations on the provided entity pairs
                scores = self.scorer(triples, batch_num=batch_num, eval_data_loader=self.eval_data,
                                     force_all_neighbours=True, only_seen_neighbours=self.only_seen_neighbours
                                     # shape (batch_size, #relations)
                                     )
                #scores = scores_all[self.eval_data.top_relations]
                # scores = scores / torch.sum(scores, dim=1).view(*scores.shape)  # turn scores into probabilities
                # NO! Our scores all have the same scale - normalising would undo that as we've missing relations!
                pair_tuples = (tuple(map(int, pair)) for pair in entity_pairs)
                pair_indices = list(self.entity_pair_indices[pair] for pair in pair_tuples)  # shape (batch_size,)
                self.pairs_scored[pair_indices] = True
                self.scores[pair_indices, :] = scores.cpu()
                for pair_idx, relation_idxs in zip(pair_indices, relation_indices):
                    self.labels[pair_idx,
                                self.range_to_top_rel_idx[relation_idxs[self.is_top_50[relation_idxs]]].cpu()] = 1

        all_scores = self.scores[:, self.relation_mask]
        labels = self.labels[:, self.relation_mask]
        if not numpy.all(self.pairs_scored):
            logger.warning(f"Only {sum(self.pairs_scored)} of {self.pairs_scored.shape} entity pairs have scores!!")
            all_scores = scores[self.pairs_scored, :]
            labels = labels[self.pairs_scored, :]
        num_up_saturated = numpy.sum((all_scores == 1.) | (all_scores == 2.))
        num_down_saturated = numpy.sum(all_scores == 0.)
        if num_up_saturated + num_down_saturated > 0:
            logger.warning(f"{num_up_saturated} scores are 1.0 or 2.0, {num_down_saturated} are 0.0, "
                           f"out of {len(all_scores.flat)}")

        # NOTE: we do P-R curve on all entity pair + kb relation combinations!
        precision, recall, _ = precision_recall_curve(labels.flat, all_scores.flat)
        if len(precision) <= 2:
            common_scores = Counter(numpy.round(all_scores.flat, 4)).most_common(20)
            logger.warning(f"Degenerate model! All scores are {common_scores}")
            return precision[0], DEGENERATE_MODEL_MARKER  # all scores were the same, likely 1.0;
            # return AP=-1 to signify a problem

        auc_pr = auc(recall, precision)
        average_precision = average_precision_score(labels.flat, all_scores.flat)
        if self.t_board is not None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title(f'Riedel 2010 NYT data: AP={average_precision:0.2f}; AUC={auc_pr:0.2f}')
            self.t_board.add_figure(f"{self.data_split}_P-R_Curve", fig, epoch if epoch is not None else 0)
            logger.info(f"Added P-R figure to tensorboard at epoch {epoch if epoch is not None else 0: 3d}, "
                        f"AUC {auc_pr:.4f} average precision {average_precision:.4f}")
        return auc_pr, average_precision


class OpenKIMapEvaluator(OpenKIEvaluator):
    def __init__(self, scorer: nn.Module, eval_data: OpenKiGraphDataHandler, batch_size, cuda, device,
                 only_seen_neighbours, other_evaluator: 'OpenKIMapEvaluator' = None, label="Evaluation"):
        super().__init__(scorer, eval_data, batch_size, cuda, device, only_seen_neighbours, other_evaluator, label)
        if other_evaluator is None:
            self.map_meter = mAPMeter()
            self.other_evaluator = None
        else:
            self.map_meter = other_evaluator.map_meter
            self.other_evaluator = other_evaluator
            assert self.map_meter is not None, "other evaluator has no map_meter in evaluator constructor!"

    def evaluate(self, epoch=None):
        """
        Run full evaluation cycle on this evaluators data set.
        :return: MAP value for this evaluator and it's data + model
        """
        with torch.no_grad():
            self.map_meter.reset()
            if self.other_evaluator is not None:
                # NOTE: dev data entity pairs are disjoint with train data pairs, hence the following is valid
                for scoring1, scoring2 in zip_longest(self.calculate_scores(), self.other_evaluator.calculate_scores()):
                    if scoring1 is None:
                        scores, labels = scoring1
                    elif scoring2 is None:
                        scores, labels = scoring2
                    else:
                        scores1, labels1 = scoring1
                        scores2, labels2 = scoring2
                        scores = torch.cat((scores1, scores2))
                        labels = torch.cat((labels1, labels2))
                    self.map_meter.add(scores, labels)  # scores and labels shape (1, #pairs)
                    # add: input is NxK where we rank the N examples for each of K classes
                    # N examples (single relations here)
                    # K classes (entity pairs here)
                    # labels: 1 if example n has class k (here: this relation applies to pair k)
            else:
                for scores, labels in self.calculate_scores():
                    self.map_meter.add(scores, labels)  # scores and labels shape (#ranked, #ranking_per)

            return self.map_meter.value()

    def calculate_scores(self):
        raise NotImplementedError()

    def add_scores_and_labels(self, scores, labels):
        """
        Add scores and assocaited labels to the evaluator (eg: during an inference epoch).
        Input is NxK where we rank the N examples (alongside previously or subsequently added examples) for each of the
        K classes. All submitted scores should be of the same scale for stable ranking, any scalar value will do.
        Examples for each class are ranked independently, so scores for different classes need not be comparable.
        :param scores: (N, K) tensor of scores
        :param labels: (N, K) tensor of labels, all 0. or 1.
        :return: Void
        """
        self.map_meter.add(scores, labels)

    def get_map(self):
        """
        Calculate Mean Average Precision (MAP) for scores and labels stored in this evaluator.
        Average precision is calculated for each of the K classes given a ranking of added examples.
        These precision values are then averaged over classes.
        :return: MAP value
        """
        with torch.no_grad():
            return self.map_meter.value()


class OpenKIMapEvaluatorRankingRelations(OpenKIMapEvaluator):
    def __init__(self, scorer: nn.Module, eval_data: OpenKiEvalPerRelationDataHandler, batch_size, cuda, device,
                 only_seen_neighbours, other_evaluator: 'OpenKIMapEvaluatorRankingRelations' = None,
                 label="Evaluation", t_board=None):
        """
        Evaluate MAP by ranking the most frequent KB relations for each entity pair. Note that the threshold 50 is
        implemented in the data reader.
        :param scorer: trained model for calculating scores for (s, o, p) triples
        :param eval_data: an OpenKiEvalPerRelationDataHandler (iterates all entity pairs for selected relations,
                probably one relation at a time)
        :param batch_size: for iteration over eval data
        :param cuda: apply tensor.cuda() to batches
        :param device: torch.device for new tensors
        :param only_seen_neighbours: do not include neighbour relations/predicates unseen in training data
        :param other_evaluator: for evaluation also on a second data source (typically train data)
        :param label: a text label for logging etc...
        """
        super().__init__(scorer, eval_data, batch_size, cuda, device, only_seen_neighbours, other_evaluator, label)

        so_pairs = self.eval_data.entity_pair_list
        self.so_pair_index = {so: i for i, so in enumerate(so_pairs)}

        self.all_triples = torch.cat(
            (torch.tensor(so_pairs, device=self.device, requires_grad=False, dtype=torch.long),
             torch.zeros((len(so_pairs), 1), device=self.device, requires_grad=False, dtype=torch.long)), 1
        ).reshape((len(so_pairs), 1, 3))
        self.labels = torch.zeros(1, (len(so_pairs)), dtype=torch.uint8, device=self.device, requires_grad=False)
        #  self.labels has dtype uint8 for compatability with torchnet.meter.mAPMeter

    # def _setup_train_pairs(self, train_data: OpenKIReverbTrainDataReaderOld):
    #     """
    #     Load and evaluate train data pairs and store in self.train_scores and self.train_labels, both with shape
    #     (#predicates,). We're assuming this set of subject/object pairs and those in eval/test data are disjoint.
    #     :param train_data:
    #     """
    #     if train_data is None:
    #         self.train_scores = torch.zeros((0,))
    #         self.train_labels = torch.zeros((0,))
    #     else:
    #         self.train_scores = torch.zeros((0,))
    #         self.train_labels = torch.zeros((0,))
    #         # TODO: WITH_TRAIN_PAIRS: implement setting up train data pairs self.train_scores, self.train_labels

    def calculate_scores(self):
        """
        Run this evaluators model on it's data, iterating over inferred scores and corresponding labels
        :return:
        """
        with torch.no_grad():
            for batch_num, positive_triple_lists in \
                    enumerate(tqdm(OpenKIMemoryIndexedEvalPerRelationDataLoader(self.eval_data, batch_size=1),
                         desc=self.label)):
                for positive_triples in positive_triple_lists:
                    if len(positive_triples):
                        if self.cuda:
                            positive_triples = positive_triples.cuda()  # shape (#+ve_so_pairs, 3)

                        # assume the relation is the same for all (true if it's a OpenKiEvalPerRelationDataHandler
                        this_relation = positive_triples[0, 2]
                        self.all_triples[:, :, 2] = this_relation  # shape (#so_pairs, 1, 3)
                        scores = self.scorer(self.all_triples, batch_num=batch_num, eval_data_loader=self.eval_data,
                                             force_all_neighbours=True,
                                             only_seen_neighbours=self.only_seen_neighbours)  # shape (#so_pairs, 1)
                        scores = scores.view(1, -1)  # shape (1, #so_pairs)
                        # scores = scores / torch.sum(scores, dim=1).view(*scores.shape)  # scores into probabilities
                        # NO! Our scores all have the same scale by our training
                        # - this would undo that as we've missing relations!

                        positive_indices = [self.so_pair_index[(int(s), int(o))] for s, o, _ in positive_triples]
                        self.labels[:, :] = 0
                        self.labels[0, positive_indices] = 1  # set observed triples to 1  # shape (1, #so_pairs)

                        yield scores, self.labels  # scores and labels shape (1, #pairs)
                    else:
                        logger.info("empty eval triples list!")


class OpenKIMapEvaluatorRankingPairs(OpenKIMapEvaluator):
    def __init__(self, scorer: nn.Module, eval_data: OpenKiEvalPerPairDataHandler, batch_size, cuda, device,
                 only_seen_neighbours, other_evaluator: 'OpenKIMapEvaluatorRankingPairs' = None, label="Evaluation",
                 t_board=None):
        """
        Evaluate using entity pairs in dev/test data as queries (ranking relations for each query).
        The data reader will yield data for a subset of predicates/relations (all preds and rels; the 250 KB relations,
        the top 50 KB relations).
        :param scorer: trained model for calculating scores for (s, o, p) triples
        :param eval_data:
        :param batch_size: for iteration over eval data
        :param cuda: apply tensor.cuda() to batches
        :param device: torch.device for new tensors
        :param only_seen_neighbours: do not include neighbour relations/predicates unseen in training data
        :param other_evaluator: for evaluation also on a second data source (typically train data)
        :param label: a text label for logging etc...
        """
        super().__init__(scorer, eval_data, batch_size, cuda, device, only_seen_neighbours, other_evaluator, label)

        self.num_KB_relations = self.eval_data.first_predicate_index - self.eval_data.num_dummy_relations
        self.all_triples = torch.zeros((self.batch_size, self.eval_data.top_relations.shape[0], 3),
                                       dtype=int, device=self.device, requires_grad=False)
        self.all_triples[:, :, 2] = self.eval_data.top_relations

        self.labels = torch.zeros((self.batch_size, self.num_KB_relations), dtype=torch.uint8)

    def calculate_scores(self):
        with torch.no_grad():
            for batch_num, (entity_pairs, relation_indices) in \
                    enumerate(tqdm(OpenKIMemoryIndexedEvalPerPairDataLoader(self.eval_data, batch_size=self.batch_size),
                              desc="Evaluation")):
                if self.cuda:
                    entity_pairs = entity_pairs.cuda()  # shape (batch_size, 2)
                    # relation_indices = ... is it a tuple of tensors? (batch_size, (n,) ) where n varies
                    relation_indices = (relations.cuda() for relations in relation_indices)

                batch_size = entity_pairs.shape[0]

                triples = self.all_triples[:batch_size, :, :]
                triples[:, :, :2] = entity_pairs[:, :].unsqueeze(1)  # shape (batch_size, #relations, 3)
                scores = self.scorer(triples, batch_num=batch_num, eval_data_loader=self.eval_data,
                                     force_all_neighbours=True,
                                     only_seen_neighbours=self.only_seen_neighbours)  # shape (batch_size, #relations)
                # scores = scores / torch.sum(scores, dim=1).view(*scores.shape)  # turn scores into probabilities
                # NO! Our scores all have the same scale by our training
                # - this would undo that as we've missing relations!

                labels = self.labels[:batch_size, :]
                labels[:, :] = 0   # shape (batch_size, #relations)
                for i, relations in enumerate(relation_indices):
                    labels[i, relations] = 1

                yield scores, labels[:, self.eval_data.top_relations - self.eval_data.num_dummy_relations]
                # shapes both (batch_size, #top_relations)

                # input is NxK where we rank the N examples for each of K classes
                # N examples (a selection of entity pairs here)
                # K classes (all 50 top relations)
                # labels: 1 if example n has class k (here: pair n is related by relation k)
