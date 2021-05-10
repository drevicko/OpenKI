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
from itertools import starmap
from math import sqrt

# import numpy
from typing import Union, Callable, List, Tuple

from torch.nn import Parameter

from OpenKI import logger

import torch
from torch import nn, Tensor

from OpenKI.Constants import IGNORE_OPENKI_EMBEDS
from OpenKI.OpenKI_Data import OpenKiGraphDataHandler
from OpenKI.TextEncoders import EntityEmbedding


class MultipleRelationScorer(nn.Module):
    def __init__(self, *relation_scorers, weight_scores=True, normalise_weights=False, leaky_relu=False,
                 initial_weights: list = None, data_loader: OpenKiGraphDataHandler = None):
        super().__init__()
        self.scorers = nn.ModuleList(relation_scorers)
        self.data_loader = data_loader or self.scorers[0].data_loader
        assert all(getattr(scorer, "data_loader", None) is None or self.data_loader is scorer.data_loader
                   for scorer in self.scorers), "differing data loaders among scorers!"
        self.num_scores = len(self.scorers)
        self.weight_scorers = weight_scores  # and self.num_scores > 1
        self.normalise_weights = normalise_weights
        # TODO: initialise weights etc... differently for diffeerent scorers? Higher weight for nbhood?
        self.temperatures = torch.nn.Parameter(torch.tensor([1.0]*self.num_scores), requires_grad=True)
        self.thresholds = torch.nn.Parameter(torch.tensor([0.0]*self.num_scores), requires_grad=True)
        self.scorer_param_lists = ["temperatures", "thresholds"]
        if self.weight_scorers:
            if initial_weights is not None:
                assert len(initial_weights) == self.num_scores, f"{self.num_scores} initial weights needed, but got " \
                                                                f"{len(initial_weights)}!"
                initial_weights = [1.0 if w is None else w for w in initial_weights]
            else:
                initial_weights = [1.0] * self.num_scores
            self.weights = torch.nn.Parameter(torch.tensor(initial_weights), requires_grad=True)
            self.scorer_param_lists.append("weights")
            if leaky_relu:
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, triples, batch_num=None, eval_data_loader: OpenKiGraphDataHandler = None,
                force_all_neighbours=False, only_seen_neighbours=False):
        """
        Aggregates several logit scores as a non-negatively weighted sum of sigmoids of the logits with independent
        tunable sigmoid temperatures and thresholds.
        :param triples: a tensor shape (m, n, 3) of [s, o, p] entity and relation indices
        :param batch_num: batch number -- to aid avoidance of calculating ENE scores twice with ENE + dual scoring
        :param eval_data_loader: for evaluation, we provide extra neighbourhood evidence from dev/test data
        :param force_all_neighbours: do not limit the number of neighbours (typically for evaluation)
        :param only_seen_neighbours: do not include neighbour relations not seen in training (typically for evaluation)
        :return: aggregated score
        """
        scores = [scorer(triples, batch_num=batch_num, eval_data_loader=eval_data_loader,
                         force_all_neighbours=force_all_neighbours, only_seen_neighbours=only_seen_neighbours)
                  for scorer in self.scorers]
        #         [ (m, n) of logit scores for each scorer ]
        scores = torch.stack(scores, dim=-1)  # (m, n, #scorers) of logit
        scores = self.sigmoid((scores * self.temperatures) + self.thresholds)  # (m, n, #scores)
        if self.weight_scorers:
            weights = self.relu(self.weights)
            if self.normalise_weights:
                # NOTE: we cannot to weights *= weights.sum() as this is an in-place operation and screws bkwd pass!
                # noinspection PyAugmentAssignment
                weights = weights / weights.sum()
            # noinspection PyAugmentAssignment
            scores = scores * weights  # (m, n, #scores)
        scores[~torch.isfinite(scores)] = 0.
        return torch.sum(scores, dim=-1)


# class WithRelationEmbedding:
#     def __init__(self, relation_embedding: torch.nn.Embedding, predicate_encoder: nn.Embedding = None,
#                  predicate_encoding_aggregator=None, **kwargs):
#         super().__init__(**kwargs)
#         self.relation_embedding = relation_embedding
#         self.embedding_dim = relation_embedding.embedding_dim
#         self.predicate_encoder = predicate_encoder
#         self.predicate_encoding_aggregator = predicate_encoding_aggregator
#
#     def get_relation_embeddings(self, neighbours):
#         neighbour_embeddings = None
#         if self.predicate_encoding_aggregator != IGNORE_OPENKI_EMBEDS:
#             if type(neighbours) is list:  # it can be a list of tensors of different lengths
#                 # TODO: use nested tensor here once it's in a stable pytorch version...
#                 neighbour_embeddings = [self.relation_embedding(nb) for nb in neighbours]  # if nb is not None]
#             else:
#                 neighbour_embeddings = self.relation_embedding(neighbours)
#         if self.predicate_encoder is not None:
#             if type(neighbours) is list:
#                 neighbour_encodings = [self.predicate_encoder(nb) for nb in neighbours]
#             else:
#                 neighbour_encodings = self.predicate_encoder(neighbours)  # calculate textual relation encodings
#             if self.predicate_encoding_aggregator == IGNORE_OPENKI_EMBEDS:
#                 neighbour_embeddings = neighbour_encodings  # only use (text based) encodings
#             else:  # aggregate relation encodings and openki embeds
#                 neighbour_embeddings = self.predicate_encoding_aggregator(neighbour_embeddings, neighbour_encodings)
#         return neighbour_embeddings

class RelationEncoder(nn.Module):
    def __init__(self, relation_embedding: Union[nn.Embedding, 'EntailedEmbedding'], embed_dim: int,
                 predicate_encoder: nn.Embedding = None, predicate_encoding_aggregator=None):
        """
        Class providing embedding vectors from text predicates and kb relations. The relation_embedding is a learned
        vector for each predicate/relation. The predicate encoder is an optional extra encoder providing eg. an
        encoding of the text associated with a predicate instance. The aggregator provides a method to combine the two.
        :param relation_embedding:
        :param embed_dim:
        :param predicate_encoder:
        :param predicate_encoding_aggregator:
        """
        super().__init__()
        self.relation_embedding = relation_embedding
        self.embedding_dim = embed_dim
        if relation_embedding is not None:
            assert embed_dim == relation_embedding.embedding_dim
        # self.device = relation_embedding.weight.device
        self.predicate_encoder = predicate_encoder
        self.predicate_encoding_aggregator = predicate_encoding_aggregator
        self.send_entities_to_embedding = False  # isinstance(relation_embedding, EntailedEmbedding)
        if self.send_entities_to_embedding:
            self.device = relation_embedding.device
        else:
            self.device = relation_embedding.weight.device
        assert self.predicate_encoder is not None or \
               self.predicate_encoding_aggregator is None or \
               self.predicate_encoding_aggregator != IGNORE_OPENKI_EMBEDS, \
            f"either a predicate encoder or not ignoring openKI embeds is needed"

    def forward(self, relation_indices: torch.Tensor, seen_mask: torch.Tensor = None,
                entities: torch.Tensor = None) -> torch.Tensor:
        """
        :param relation_indices: shape (*, 2) tensor of indices and corresponding text indices. For backward
                                 compatibility, a shape (*, 1) tensor is treated as both relation and text indices.
        :param seen_mask: mask indicating which of the relation indices have been seen in training (and hence have
                          trained embeddings that can meaningfully count towards the result).
        :param entities: entity types for associated data points; used for looking up entailments.
        :return: shape (*, embed_dim) tensor of relation embeddings (learned embeddings optionally combined with text
                 based embeddings)
        """
        relation_embeddings = None
        if relation_indices.shape[-1] == 2:
            relation_text_indices = relation_indices[..., 1]
            relation_indices = relation_indices[..., 0]
        elif relation_indices.shape[-1] == 1:
            relation_text_indices = relation_indices[..., 0]
            relation_indices = relation_indices[..., 0]
        else:
            raise ValueError(f"Last dim of relation_indices not 1 or 2: {relation_indices.shape}")
        if relation_indices.shape != torch.Size((0,)):
            if self.predicate_encoding_aggregator is None or self.predicate_encoding_aggregator != IGNORE_OPENKI_EMBEDS:
                # retrieve learned relation embeds
                # if type(relation_indices) is list:  # it can be a list of tensors of different lengths
                #     relation_embeddings = [self.relation_embedding(nb) for nb in relation_indices]  # if nb is not None]
                # else:
                if self.send_entities_to_embedding:
                    # if len(entities.shape) == 0:
                    #     entities = entities.view(1)
                    embedding_params = (relation_indices, entities)
                else:
                    embedding_params = (relation_indices,)
                if seen_mask is None:
                    relation_embeddings = self.relation_embedding(*embedding_params)
                else:
                    # Only provide embeddings for "seen" relations
                    embed_shape = tuple(list(relation_indices.shape) + [self.embedding_dim])
                    relation_embeddings = torch.zeros(embed_shape, device=relation_indices.device, dtype=torch.float)
                    embedding_params = (param[seen_mask] for param in embedding_params)
                    relation_embeddings[seen_mask] = self.relation_embedding(*embedding_params)
            if self.predicate_encoder is not None:  # retrieve and aggregate with text based embeds
                # NOTE: missing texts can take any (fixed) index, and are represented as None in place of a string
                #       With separated texts/predicates, this should only be the 0th text.
                # NOTE: seen_mask is ignored here, since texts may be available irrespective of their presence in training
                if type(relation_text_indices) is list:
                    relation_encodings = [self.predicate_encoder(nb) for nb in relation_text_indices]
                else:
                    relation_encodings = self.predicate_encoder(relation_text_indices)  # calculate/retrieve text encodings
                if self.predicate_encoding_aggregator == IGNORE_OPENKI_EMBEDS:
                    relation_embeddings = relation_encodings  # only use (text based) encodings
                else:  # aggregate relation encodings and openki embeds
                    relation_embeddings = self.predicate_encoding_aggregator(relation_embeddings, relation_encodings)
        else:
            relation_embeddings = torch.tensor([], dtype=torch.float, device=self.device).view(0, self.embedding_dim)
        return relation_embeddings



class WithEntityEncoding:
    def __init__(self, entity_encoder: EntityEmbedding = None, entity_encoding_aggregator=None, encoder_dropout=0.5,
                 **kwargs):
        super().__init__(**kwargs)  # TODO: does this do anything?? Doesn't seem to create an error
        self.entity_encoder = entity_encoder
        if entity_encoder is not None:
            assert entity_encoding_aggregator is not None, "entity_encoding_aggregator required when " \
                                                           "entity_encoder supplied!"
        self.entity_encoding_aggregator = entity_encoding_aggregator
        self._cached_encodings = None
        self.dropout = nn.Dropout(encoder_dropout) if encoder_dropout is not None else None

    def combine_with_encoded_entities(self, entities, entity_embeds):
        # if self.entity_encoder is not None:
        entity_encodings = self.entity_encoder(entities)  # calculate entity (text) encodings
        if self.entity_encoding_aggregator == IGNORE_OPENKI_EMBEDS or \
                self.entity_encoding_aggregator is None or \
                entity_embeds is None:
            combined_embeds = entity_encodings  # use only entity encodings (ignore openki/predicates)
        else:
            if self.dropout is not None:
                # TODO: does dropout() preserve shape??
                entity_embeds = self.dropout(entity_embeds)
            combined_embeds = self.entity_encoding_aggregator(entity_embeds, entity_encodings)
        return combined_embeds


# class EntityEncoder(WithEntityEncoding, nn.Embedding):
#     def __init__(self, entity_embedding: nn.Embedding, **kwargs):
#         super().__init__(**kwargs)
#
#     def forward(self, entity_indices: torch.Tensor) -> torch.Tensor:
#         embeddings = super().forward(entity_indices)
#         return self.combine_with_encoded_entities(entity_indices, embeddings)


class ModuleAndEntityEncoding(WithEntityEncoding, nn.Module):
    pass  # just for type checking...


class EModelEntityEncoder(ModuleAndEntityEncoding):
    subjectObjectLabels = ["subject", "object"]
    """
    NOTE: Currently this model underperforms badly. It has not received further TLC!
    """

    def __init__(self, data_loader: OpenKiGraphDataHandler, subject_or_object, embed_dim,
                 relation_encoder: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.data_loader = data_loader
        self.embedding_dim = embed_dim
        self.relation_encoder = relation_encoder  # referenced from EntityNeighbourhoodScorer
        self.entity_embedding = nn.Embedding(data_loader.num_entities, self.embedding_dim, padding_idx=0)
        # self.entity_embedding = entity_embedding
        self.ourEntityIndex = self.subjectObjectLabels.index(subject_or_object)
        self.seen_entities = data_loader.seen_entities[self.ourEntityIndex]
        assert self.ourEntityIndex in (0, 1)

    def forward(self, entity_pairs, batch_num=None, eval_data_loader=None, force_all_neighbours=False,
                only_seen_neighbours=False):
        # Add dummy eval_data_loader parameter for compatiibility with EntityNeighbourhoodScorer.forward()
        entities = entity_pairs[:, :, self.ourEntityIndex]
        embeddings = self.entity_embedding(entities)
        if only_seen_neighbours:
            batch_seen = self.seen_entities[entities]
            if not torch.all(batch_seen):
                embeddings = torch.where(batch_seen.unsqueeze(-1), embeddings, torch.zeros_like(embeddings))
        # NOTE: we combine entity encodings AFTER zeroing unseen entity embeds
        if self.entity_encoder is not None:
            embeddings = self.combine_with_encoded_entities(entities, embeddings)
        return embeddings


class NeighbouringRelationsEntityEncoder(ModuleAndEntityEncoding):
    subjectObjectLabels = ["subject", "object"]

    def __init__(self, data_loader: OpenKiGraphDataHandler, subject_or_object, aggregator,
                 relation_encoder: Union[RelationEncoder, nn.Embedding], max_nbhd_preds=None, **kwargs):
        """
        For OpenKI aggregated entity neighbourhood calculation.
        :param data_loader: The knowledge graph in which to find the neighbourhood.
        :param relation_embedding: An embedding relation_index -> vector
                (in OpenKI, there is one for subjects, one for objects)
        :param subject_or_object: a string "subject" or "object" - are we encoding the subject neighbourhood or object?
        :param aggregator: aggregator function, takes a (*, n, D) batch of embeddings, returns aggregated (*, D) vecs
        :param entity_encoder: module that generates representations given entity id lists
        :param entity_encoding_aggregator: function to combine entity_neighbour_encoder output with OpenKI v^agg_su/obj
        :param relation_encoder: module that generates representations given predicate id lists
        """
        super().__init__(**kwargs)
        self.data_loader = data_loader
        self.ourEntityIndex = self.subjectObjectLabels.index(subject_or_object)
        assert self.ourEntityIndex in (0, 1)
        self.otherEntityIndex = 1 - self.ourEntityIndex
        self.aggregator = aggregator
        self.max_nbhd_preds = max_nbhd_preds
        self.relation_encoder = relation_encoder
        self.embedding_dim = relation_encoder.embedding_dim
        self.batch_num = None
        self._cached_encodings = None

    def forward(self, entity_pairs, batch_num=None, exclude_so_neighbours=False,
                eval_data_loader: OpenKiGraphDataHandler = None, force_all_neighbours=False,
                only_seen_neighbours=True, nans_to_zero=True):
        """
        Calculates the aggregated neighbourhood embeddings for candidate entity pairs.
        This is v^agg_subj or v^agg_obj in the paper.
        :param entity_pairs: candidate entity pairs with shape (n, m, 2): batch of n lists of m pairs
        :param batch_num: used to identify repeat calls for the same batch (cached values are used in that case)
        :param exclude_so_neighbours: whether to exclude relations between candidate entity pairs
        :param eval_data_loader: for evaluation, we provide extra neighbourhood evidence from dev/test data
        :return: aggregated neighbourhood relation vectors, shape (n, m, K): embed dim K
        :param force_all_neighbours: for evaluation, ignore max_nbhd_preds and always use all neighbours
        :param only_seen_neighbours: do not include neighbours not seen in training
        :param nans_to_zero: set all NaN values to zero
        """
        assert not exclude_so_neighbours, f"excluding links to the candidate object from neigbours not implemented!"
        assert self.entity_encoder is not None or \
               self.entity_encoding_aggregator is None or \
               self.entity_encoding_aggregator != IGNORE_OPENKI_EMBEDS, \
            f"either an entity encoder or not ignoring predicate neighbour embeds is needed"
        if batch_num == self.batch_num and not (batch_num is None or self._cached_encodings is None):
            if (entity_pairs.shape[0], entity_pairs.shape[1], self.relation_encoder.embedding_dim) \
                    == self._cached_encodings.shape:
                return self._cached_encodings
            else:
                logger.warning(f"cached batch shape mismatch: this batch ({entity_pairs.shape}, "
                               f"{self.relation_encoder.embedding_dim}); cached {self._cached_encodings.shape} "
                               f"at batch {batch_num} --- ignoring cached values!")

        self.batch_num = batch_num
        if entity_pairs.shape[-1] == 2 and len(entity_pairs.shape) == 3:
            our_entities = entity_pairs[:, :, self.ourEntityIndex]
        elif len(entity_pairs.shape) == 2:
            our_entities = entity_pairs  # this is when used within a tensor scorer
        else:
            raise ValueError(f"Entity pairs in ENE encoder with unexpected shape {entity_pairs.shape}!")
        our_entities_flat = our_entities.flatten()
        if eval_data_loader is None:
            neighbours = self.data_loader.get_s_or_o_neighbours(our_entities_flat, self.ourEntityIndex)
            # neighbours now has shape (n, 1) or (n,2)
            seen_nb_masks = [None for _ in neighbours]
        else:
            # NOTE: here we substitute eval_data_loader neighbour lists.
            neighbours = eval_data_loader.get_s_or_o_neighbours(our_entities_flat, self.ourEntityIndex)
            # neighbours now has shape (n, 1) or (n,2)
            seen_nb_masks: (list, torch.Tensor)
            if only_seen_neighbours:
                seen_nb_masks = [self.data_loader.seen_s_or_o_neighbours[self.ourEntityIndex, nb[:, 0]] for nb in
                                 neighbours]
                if not isinstance(self.relation_encoder, RelationEncoder) or \
                        self.relation_encoder.predicate_encoder is None:
                    with torch.no_grad():
                        # apply the masks now, as self.relation_encoder doesn't know how to deal with them
                        # NOTE: nb can be shape (l, 1) or (l, 2), but this indexing works in both cases
                        neighbours = [nb[mask] for nb, mask in zip(neighbours, seen_nb_masks)]
                        seen_nb_masks = [None for _ in neighbours]
            else:
                seen_nb_masks = [None for _ in neighbours]
        # TODO: SPEEDUP: do this using torch.isnan and bool indexing !!
        if self.max_nbhd_preds is not None and not force_all_neighbours:
            # randomly select self.max_nbhd_preds elements from each neighbour list
            # TODO: SPEEDUP: use padded tensors to do this all at once and on GPU
            mask: torch.Tensor
            for i, (nb, mask) in enumerate(zip(neighbours, seen_nb_masks)):
                if nb.shape[0] > self.max_nbhd_preds:
                    chosen = torch.randperm(nb.shape[0])[:self.max_nbhd_preds]
                    neighbours[i] = nb[chosen]
                    if mask is not None:
                        seen_nb_masks[i] = mask[chosen]
        if self.entity_encoding_aggregator is None or \
                self.entity_encoding_aggregator != IGNORE_OPENKI_EMBEDS:  # we're using pred. neighbour representations...
            try:
                if isinstance(self.relation_encoder, RelationEncoder):
                    # pass masks to relation_encoder, which will only mask learned relation embeds, not text repr.
                    neighbours_ = list(starmap(self.relation_encoder, zip(neighbours, seen_nb_masks, our_entities_flat)))
                #     TODO: looks like this is returning [100, embed_dim] from [100, 50] indices in neighbours!
                # foo
                else:
                    # relation_encoder is an Embedding; either no masks or already applied
                    neighbours_ = list(map(self.relation_encoder, (neighbour_t[..., 0] for neighbour_t in neighbours)))
            except RuntimeError:
                if self.relation_encoder.predicate_encoder is not None:
                    if isinstance(self.relation_encoder, RelationEncoder):
                        our_device = self.relation_encoder.predicate_encoder.weight.device
                        embed_device_msg = f" and relation word embeds are on " \
                                           f"{self.relation_encoder.relation_embedding.weight.device}"
                    else:
                        our_device = self.relation_encoder.weight.device
                        embed_device_msg = ""
                    logger.info(f"neighbours are on {neighbours[0].device} with passed data on {entity_pairs.device}, "
                                f"we are on {our_device}{embed_device_msg}")
                raise
            neighbours = neighbours_  # these are already embeddings
            neighbours_shape = (entity_pairs.shape[0], entity_pairs.shape[1], neighbours[0].shape[-1])
            self._cached_encodings = self.aggregator(neighbours).reshape(neighbours_shape)
            if nans_to_zero:
                self._cached_encodings = torch.where(torch.isnan(self._cached_encodings),
                                                     torch.zeros_like(self._cached_encodings),
                                                     self._cached_encodings)
        else:
            self._cached_encodings = None
        if self.entity_encoder is not None:
            self._cached_encodings = self.combine_with_encoded_entities(our_entities, self._cached_encodings)
        return self._cached_encodings  # this is v^agg_subj/obj


class EntityNeighbourhoodScorer(nn.Module):
    def __init__(self, entity_neighbour_encoder: ModuleAndEntityEncoding):
        super().__init__()
        self.entityEncoder = entity_neighbour_encoder
        self.relation_embedding = entity_neighbour_encoder.relation_encoder
        self.ourEntityIndex = entity_neighbour_encoder.ourEntityIndex
        self.relation_embedding_is_encoder = isinstance(self.relation_embedding, RelationEncoder)
        self.data_loader = entity_neighbour_encoder.data_loader

    def forward(self, triples, batch_num=None, eval_data_loader: OpenKiGraphDataHandler = None,
                force_all_neighbours=False, only_seen_neighbours=False):
        """
        :param triples: tensor shape (n, m, 3) or (n, m, 4): batch of n lists of m triples/quads (quads have txt idx)
        :param batch_num: used to determine if we can used cached encodings (ie: when it's the same as last time)
        :param eval_data_loader: extra OpenIE predicates in evaluation data (only when evaluating)
        :param force_all_neighbours: do not limit the number of other so relations for scoring if this is True
        :param only_seen_neighbours: do not use unseen neighbouring predicates in calculcation
        :return: tensor shape (n, m) of scores
        """
        candidate_relation_indexes = triples[:, :, 2:]  # shape (n, m, 1 or 2) --- (n, m) of p or p and t
        entity_pairs = triples[:, :, :2]          # shape (n, m, 2) --- (n, m) of s and o

        embeddings_shape = list(entity_pairs.shape)
        embeddings_shape[-1] = self.relation_embedding.embedding_dim
        embeddings_shape = tuple(embeddings_shape)
        # embeddings_shape = (entity_pairs.shape[0], entity_pairs.shape[1],
        #                     self.relation_embedding.embedding_dim)

        # shape (n, m, K) with embed dim K
        encoded_entities = self.entityEncoder(entity_pairs, batch_num=batch_num,
                                              eval_data_loader=eval_data_loader,
                                              force_all_neighbours=force_all_neighbours,
                                              only_seen_neighbours=only_seen_neighbours)  # v^agg_subj/obj
        # set nan's from new entities to zero here so they don't contribute to the sum below
        # encoded_entities = torch.where(torch.isnan(encoded_entities), torch.zeros_like(encoded_entities),
        #                                encoded_entities)  # this is now done in the entityEncoder

        if self.relation_embedding_is_encoder:
            candidate_relation_embeddings = self.relation_embedding(candidate_relation_indexes,
                                                                    entities=entity_pairs[:, :, self.ourEntityIndex])
        else:
            # This is an nn.Embedding, so we ignore indices of associated texts (if present)
            candidate_relation_embeddings = self.relation_embedding(candidate_relation_indexes[..., 0])
        candidate_relation_embeddings = candidate_relation_embeddings.reshape(embeddings_shape)
        # both have shapes (bach_size, #batch_items, embed_dim) = (n, m, K)
        return torch.sum(candidate_relation_embeddings * encoded_entities, dim=-1)  # S^ENE_subj/obj


class EntityPairAttentionRelationsScorer(nn.Module):
    """
    Base class for scorers that learn embeddings for predicates and relations and score by an attention mechanism over
    "neighbouring" (in the sense of sharing entity pairs) relations.
    """
    def __init__(self, data_loader: OpenKiGraphDataHandler, relation_embedding: torch.nn.Embedding,
                 aggregator, max_relation_pairs=None):
        super().__init__()
        self.data_loader = data_loader
        self.relation_embedding = relation_embedding
        self.embedding_dim = relation_embedding.embedding_dim
        self.max_relation_pairs = max_relation_pairs
        self.aggregator = aggregator
        self.softmax = nn.Softmax(dim=0)

    def get_other_relation_indices(self, triples, eval_data_loader: OpenKiGraphDataHandler = None,
                                   force_all_neighbours=False, only_seen_neighbours=False):
        """
        Returns the set of relation embeddings for relations in the training data between each entity pair
        :param triples: iterable over n*m entity pairs...
        :param eval_data_loader: list of tensors of relation indices
        :param force_all_neighbours: do not limit the number of other so relations for scoring if this is True
        :param only_seen_neighbours: exclude neighbours that were not seen during training
        :return: list of n*m (l,) int tensors of relation indices, where l varies (the number of relations for that
                 entity pair)
        """
        # other_relation_indices: list of n*m tensors with shapes (l, K) where l varies; K is embedding dimension;
        # one for each entity pair. For new entities, the entry is None.
        if eval_data_loader is None:
            other_relation_indices = self.data_loader.get_so_relations(triples)
        else:
            # For evaluation, get neighbours from provided dev/test data (train does not include these so pairs!)
            if only_seen_neighbours:
                other_relation_indices = eval_data_loader.get_so_relations(
                    triples, seen_relation_mask=self.data_loader.seen_so_neighbours
                )
            else:
                other_relation_indices = eval_data_loader.get_so_relations(triples)
        if self.max_relation_pairs is not None and not force_all_neighbours:
            # randomly select self.max_relation_pairs elements from each other_relation_indices list
            for i, other in enumerate(other_relation_indices):
                if other.shape[0] > self.max_relation_pairs:
                    chosen = torch.randperm(other.shape[0])[:self.max_relation_pairs]
                    other_relation_indices[i] = other[chosen]
        return other_relation_indices  # list of n*m (l,) or (l, 2) int tensors, where l varies, with or without texts

    def calculate_attention_weights(self, candidate_relation_embeds, other_relation_indices, entity_pairs,
                                    candidate_relation_indexes, eval_data_loader, force_all_neighbours,
                                    batch_num=None):
        raise NotImplementedError()

    def calculate_neighbour_aggregation(self, triples, batch_num=None, eval_data_loader: OpenKiGraphDataHandler = None,
                                        force_all_neighbours=False, only_seen_neighbours=False):
        candidate_relation_indexes = triples[:, :, 2:]  # shape (n, m, 1 or 2) - [p] from [s, o, p] triples
        entity_pairs = triples[:, :, :2]          # shape (n, m, 2) - [s, o] from [s, o, p] triples

        # TODO: To use entailments here, we really need to set it up for entity pair entailments (currently we only
        #       have single entity entailments to use with ENE). We'd pass entity_pairs to relation_embedding.
        candidate_relation_embeds = self.relation_embedding(candidate_relation_indexes).flatten(end_dim=-2)
        # TODO: Separate Texts: separate text indices and apply relation_embedding only to relation indices
        # (n*m, K, 1)

        # NOTE: the :3 below is to remove prospective text indices (not needed for finding so neighbours)
        other_relation_indices = self.get_other_relation_indices(triples[:, :, :3].view(-1, 3), eval_data_loader,
                                                                 force_all_neighbours=force_all_neighbours,
                                                                 only_seen_neighbours=only_seen_neighbours)
        # list of n*m (l,) or (l, 2) int tensors where l varies, with or without text indices

        # TODO: To use entailments here, we really need to set it up for entity pair entailments (currently we only
        #       have single entity entailments to use with ENE). We'd pass entity_pairs to relation_embedding.
        other_relation_embeds = tuple(self.relation_embedding(indices).flatten(end_dim=-2) for indices in other_relation_indices)
        # tuple of n*m (l, K) tensors where l varies (K is embedding dim)

        attention_weights = self.calculate_attention_weights(candidate_relation_embeds, other_relation_indices,
                                                             entity_pairs, candidate_relation_indexes,
                                                             eval_data_loader, force_all_neighbours, batch_num)
        # one (QueryScorer) or two (DualAttentionScorer) n*m tuples of (l,) weight tensors

        return candidate_relation_embeds, self.aggregator(other_relation_embeds, *attention_weights, embed_dim=self.embedding_dim)
        # (n*m, K) tensor of aggregated embeddings

    def forward(self, triples, batch_num=None, eval_data_loader: OpenKiGraphDataHandler = None,
                force_all_neighbours=False, only_seen_neighbours=False):
        """
        :param triples: tensor shape (n, m, 3): batch of n lists of m triples
        :param batch_num: used to determine if we can used cached encodings (ie: when it's the same as last time)
        :param eval_data_loader: extra OpenIE predicates in evaluation data (only when evaluating)
        :param force_all_neighbours: do not truncate the neighbour list (used when evaluating)
        :param only_seen_neighbours: do not include neighbours not seen in training (used when evaluating)
        :return: tensor shape (n, m) of scores
        """
        candidate_relation_embeds, weighted_sums = self.calculate_neighbour_aggregation(
            triples, batch_num, eval_data_loader, force_all_neighbours, only_seen_neighbours
        )
        # (n*m, K) tensor of aggregated embeddings

        return (weighted_sums * candidate_relation_embeds).sum(dim=-1).view(triples.shape[0], triples.shape[1])
        # (n, m) tensor of scores


class EntityPairAttentionNeighboursRelationEmbedding(EntityPairAttentionRelationsScorer):
    """
    For query-attention and dual-attention relation encoders in Tucker-like models
    """
    def __init__(self, concat_learned_and_neighbour_embeds=True,
                 subject_encoder: NeighbouringRelationsEntityEncoder = None,
                 object_encoder: NeighbouringRelationsEntityEncoder = None,  *args, **kwargs):
        self.use_dual_attention = subject_encoder is not None
        if self.use_dual_attention:
            aggregator = dual_attention_logit_aggregator
        else:
            aggregator = attention_aggregator
        super().__init__(aggregator=aggregator, *args, **kwargs)
        self.concatenate_embeds = concat_learned_and_neighbour_embeds
        if concat_learned_and_neighbour_embeds:
            self.embedding_dim *= 2
        self.subject_encoder = subject_encoder
        self.object_encoder = object_encoder
        # TODO: concat subject/object based query relation embeds also!

    def calculate_attention_weights(self, candidate_relation_embeds, other_relation_indices, entity_pairs,
                                    candidate_relation_indexes, eval_data_loader, force_all_neighbours, batch_num=None):
        if self.use_dual_attention:
            return DualAttentionRelationScorer.calculate_attention_weights(
                self, candidate_relation_embeds, other_relation_indices, entity_pairs,
                candidate_relation_indexes, eval_data_loader, force_all_neighbours, batch_num
            )
        else:
            return QueryRelationScorer.calculate_attention_weights(
                self, candidate_relation_embeds, other_relation_indices, entity_pairs,
                candidate_relation_indexes, eval_data_loader, force_all_neighbours, batch_num
            )

    def forward(self, triples, batch_num=None, eval_data_loader: OpenKiGraphDataHandler = None,
                force_all_neighbours=False, only_seen_neighbours=False):
        # We may see these parameters from the ENE entity encoder forward() signature:
        # nans_to_zero=True
        # exclude_so_neighbours=False,
        candidate_relation_embeds, weighted_sums = self.calculate_neighbour_aggregation(
            triples, batch_num, eval_data_loader, force_all_neighbours, only_seen_neighbours
        )
        # (n*m, K) tensor of aggregated embeddings

        if self.concatenate_embeds:
            return torch.cat((candidate_relation_embeds, weighted_sums), dim=-1)
        else:
            return weighted_sums


class QueryRelationScorer(EntityPairAttentionRelationsScorer):
    def __init__(self, data_loader: OpenKiGraphDataHandler, relation_embedding: torch.nn.Embedding,
                 max_relation_pairs=None):
        super().__init__(data_loader, relation_embedding, attention_aggregator, max_relation_pairs)

    def calculate_attention_weights(self, candidate_relation_embeds, other_relation_indices, entity_pairs,
                                    candidate_relation_indexes, eval_data_loader, force_all_neighbours, batch_num=None,
                                    return_logits=False):
        """
        :param candidate_relation_embeds: (n*m, K) tensor of K-dim embeddings
        :param other_relation_indices: list of n*m (l, ) int tensors where l varies
        :param entity_pairs: included for compatibility, not used here
        :param candidate_relation_indexes:  included for compatibility, not used here
        :param eval_data_loader: included for compatibility, not used here
        :param force_all_neighbours: do not sample from neighbour lists, use all of them (used in evaluation)
        :param batch_num: current batch number (used to to avoid duplicate calculations with ENE)
        :param return_logits: return attention scores as un-normalised logits
        :return: list of n*m (l, ) int tensors where l varies
        """
        # TODO: To use entailments here, we really need to set it up for entity pair entailments (currently we only
        #       have single entity entailments to use with ENE). We'd pass entity_pairs to relation_embedding.
        other_relation_embeds = [self.relation_embedding(indices) for indices in other_relation_indices]
        # list of n*m (l, K) tensors where l varies (K is embedding dim)
        # TODO: some entries in other_relation_embeds could be (0, K) tensors
        attention_weights = tuple((c.view(1, -1) * o).sum(dim=-1) for c, o in
                                  zip(candidate_relation_embeds, other_relation_embeds))  # n*m list of (l, ) tensors
        if not return_logits:
            attention_weights = tuple(self.softmax(w) for w in attention_weights)  # n*m list of (l, ) tensors
        return (attention_weights,)  # n*m list of (l, ) tensors


class DualAttentionRelationScorer(EntityPairAttentionRelationsScorer):
    def __init__(self, data_loader: OpenKiGraphDataHandler, relation_embedding: torch.nn.Embedding,
                 subject_encoder: NeighbouringRelationsEntityEncoder,
                 object_encoder: NeighbouringRelationsEntityEncoder, max_relation_pairs=None):
        super().__init__(data_loader, relation_embedding, dual_attention_logit_aggregator, max_relation_pairs)
        self.subject_encoder = subject_encoder
        self.object_encoder = object_encoder

    def calculate_attention_weights(self, candidate_relation_embeds, other_relation_indices, entity_pairs,
                                    candidate_relation_indexes, eval_data_loader, force_all_neighbours,
                                    batch_num=None):
        """
        :param candidate_relation_embeds: (n*m, K) tensor of K-dim embeddings
        :param other_relation_indices: list of n*m (l, ) int tensors where l varies;
                                       indices of other rels between respective entity pairs
        :param entity_pairs: (n, m, 2) - passed to sub/object_encoder
        :param candidate_relation_indexes:  (n, m, 1) - passed to sub/object_encoder.relation_embedding
        :param eval_data_loader: passed to sub/object_encoder
        :param force_all_neighbours: do not sample from neighbour lists, use all of them (used in evaluation)
        :param batch_num: current batch number (used to to avoid duplicate calculations with ENE)
        :return:
        """
        # TODO: #42 do separate Neighbour attention and dual attention models - dual takes two scorers
        query_weights = QueryRelationScorer.calculate_attention_weights(self, candidate_relation_embeds,
                                                                        other_relation_indices, entity_pairs,
                                                                        candidate_relation_indexes, eval_data_loader,
                                                                        batch_num, return_logits=True)
        query_weights = query_weights[0]  # it was a tuple length one, we want what's inside!
        # n*m list of (l, ) tensors
        # TODO: #55 nan loss in dual scorer - subject, object and query relation embeds have nans

        # Calculate entity neighbourhood v^agg_s/obj vectors (one s- and one o- vector for each entity pair)
        subject_vectors = self.subject_encoder(entity_pairs, batch_num=batch_num,
                                               eval_data_loader=eval_data_loader,
                                               force_all_neighbours=force_all_neighbours)
        object_vectors = self.object_encoder(entity_pairs, batch_num=batch_num,
                                             eval_data_loader=eval_data_loader,
                                             force_all_neighbours=force_all_neighbours)
        # # set nan's from new entities to zero here so they don't contribute to the sum below
        # --- now done in entity encoder
        # subject_vectors = torch.where(torch.isnan(subject_vectors), torch.zeros_like(subject_vectors),
        #                               subject_vectors).view(-1, self.subject_encoder.embedding_dim)
        # object_vectors = torch.where(torch.isnan(object_vectors), torch.zeros_like(object_vectors),
        #                              object_vectors).view(-1, self.object_encoder.embedding_dim)
        # both have shape (batch_size * #batch_items, embed_dim) = (n*m, K)
        # the following flattening produces the shape that would have been produced by above lines:
        subject_vectors = subject_vectors.flatten(end_dim=-2)
        object_vectors = object_vectors.flatten(end_dim=-2)

        # Get the subject- and object-embeddings for the other relations
        other_s_relation_embeds = (self.subject_encoder.relation_encoder(indices)
                                   for indices in other_relation_indices)
        other_o_relation_embeds = (self.object_encoder.relation_encoder(indices)
                                   for indices in other_relation_indices)
        # n*m lists of (l, K) tensors where l varies (K is embedding dim)

        # We pass logits to the aggregator, hence no softmax here (indeed renormalising in the dual aggregator makes
        # softmax here superfluous.
        neighbourhood_weights = tuple(
            (subj_vec.view(1, -1) * other_s_emb + obj_vec.view(1, -1) * other_o_emb).sum(dim=-1)
            for subj_vec, obj_vec, other_s_emb, other_o_emb
            in zip(subject_vectors, object_vectors, other_s_relation_embeds, other_o_relation_embeds)
        )  # n*m tuple of (l,) tensors

        return query_weights, neighbourhood_weights


# Aggregators for generating query representations from neighbours
def average_aggregator(embedding_batch, embed_dim=0):
    """
    aggregates by averaging embedding lists in each of the n batch entries
    :param embedding_batch: python list of n tensors of m x K-dim embedding vectors (m can vary)
    :param embed_dim: K, used if the embedding_lists is empty (n = 0) to return a (0, K) empty tensor
    :return: n x K tensor of averaged embeddings
    """
    if len(embedding_batch):
        batch_means = [t.mean(dim=0) for t in embedding_batch]
        return torch.stack(batch_means)
    else:
        return torch.tensor([]).view(-1, embed_dim)


def average_aggregator_normed(embedding_batch, sqrt_d=None, embed_dim=0):
    """
    aggregates by averaging embedding lists in each of the n batch entries using sqrt(#embeds per entry) in the average
    calculation, and scaling by sqrt(embed_dim). The resulting average vector has expected norm one, independent of
    the number of embeddings per entry and the embedding dimension.
    :param embedding_batch: python list of n tensors of m x K-dim embedding vectors (m can vary)
    :param sqrt_d: ... in case we have pre-calculated or want to override scaling by embed dimension
    :param embed_dim: K, used if the embedding_lists is empty (n = 0) to return a (0, K) empty tensor
    :return: n x K tensor of averaged embeddings
    """
    if len(embedding_batch):
        batch_means = [t.mean(dim=0) for t in embedding_batch]
        batch_sqrt_n = torch.tensor([sqrt(t.shape[0]) for t in embedding_batch],
                                    device=batch_means[0].device).unsqueeze(-1)
        if sqrt_d is None:
            sqrt_d = math.sqrt(embedding_batch[0].shape[-1])  # sqrt(embed_dim)
        return torch.stack(batch_means) * batch_sqrt_n / sqrt_d
    else:
        return torch.tensor([]).view(-1, embed_dim)


def attention_aggregator(embedding_lists, weights, embed_dim=0) -> torch.Tensor:
        """
        Returns a weighted sum of embeddings
        :param embedding_lists: list of n tensors of shape (l, K) embedding tensors (l can vary)
        :param weights: list of n tensors of shape (l,) weights (l can vary, but matches embeddings)
        :param embed_dim: K, used if the embedding_lists is empty (n = 0) to return a (0, K) empty tensor
        :return: weighted sum of embeddings, shape (n, K)
        """
        assert len(embedding_lists) == len(weights), f"aggregation weights different length to embeddings! weights len " \
                                                     f"{len(weights)}, embeds len {len(embedding_lists)}"
        if len(embedding_lists):
            if len(embedding_lists) == 1:
                return embedding_lists[0]  # (1, K) tensor with the single embedding (ie: no aggregation necessary)
            aggregated = torch.stack([torch.sum(emb * w.view(-1, 1), dim=0) for emb, w in zip(embedding_lists, weights)])
            return aggregated  # (n, K) tensor of aggregated embeddings
        else:
            return torch.tensor([]).view(-1, embed_dim)


def dual_attention_logit_aggregator(embedding_lists, weights_1, weights_2, embed_dim=0):
    """
    Returns an OpenKI dual attention calculation. Weights are multiplied, normed to sum 1, then used in a weighted sum
    over embeddings.
    :param embedding_lists: list of n tensors of shape (l, K) embedding tensors (l can vary)
    :param weights_1: list of n tensors of shape (l,) weights (l can vary, but matches embeddings)
    :param weights_2: list of n tensors of shape (l,) weights (l can vary, but matches embeddings)
    :param embed_dim: K, used if the embedding_lists is empty (n = 0) to return a (0, K) empty tensor
    :return: dual-weighted average of embeddings (or zero if l=0), shape (n, K)
    """
    assert len(embedding_lists) == len(weights_1), f"aggregation weights_1 different length to embeddings! " \
                                                   f"len(weights_1) = {len(weights_1)}, " \
                                                   f"len(weights_2) = {len(weights_2)}, " \
                                                   f"len(embeds) = {len(embedding_lists)}"
    assert len(embedding_lists) == len(weights_2), f"aggregation weights_2 different length to embeddings! " \
                                                   f"len(weights_1) = {len(weights_1)}, " \
                                                   f"len(weights_2) = {len(weights_2)}, " \
                                                   f"len(embeds) = {len(embedding_lists)}"
    if len(embedding_lists):
        logit_weights = tuple(lw1 + lw2 for lw1, lw2 in zip(weights_1, weights_2))
        logit_w_sums = (torch.logsumexp(lw, dim=0) for lw in logit_weights)
        logit_weights = tuple(lw - ls for lw, ls in zip(logit_weights, logit_w_sums))
        aggregated = torch.stack(
            [torch.sum(emb * torch.exp(lw).view(-1, 1), dim=0) for emb, lw in zip(embedding_lists, logit_weights)])
        return aggregated  # (n, K) tensor of aggregated embeddings
    else:
        # TODO: this needs to be put on the right device... Currently this line never gets executed...
        return torch.tensor([]).view(-1, embed_dim)


def max_pool_weighted_aggregator(embedding_lists, weights=None, embed_dim=0):
    # embedding_lists: len=batch_size, shapes=(#embeds, embed_dim)
    # weights:         len=batch_size, shapes=(#embeds,)

    # a few consistency checks...
    assert type(embedding_lists) is list and (weights is None or type(weights) is list)
    if len(embedding_lists) > 0:
        assert len(embedding_lists[0].shape) == 2 and (not weights or len(weights[0].shape) == 1)
        embed_dim = embedding_lists[0].shape[-1]
        zero_tensor = torch.zeros((embed_dim,), dtype=torch.float, device=embedding_lists[0].device)
        if weights is not None:
            assert len(embedding_lists) == len(weights)
            assert all(embeds.shape[0] == weight.shape[0] for embeds, weight in zip(embedding_lists, weights))

            # noinspection PyUnresolvedReferences
            pooled_embeds: List[Tensor] = [torch.max(embeds * weight.view(-1,1), dim=0).values
                                                          if len(embeds) > 0 else zero_tensor
                                                          for embeds, weight in zip(embedding_lists, weights)]
        else:
            # noinspection PyUnresolvedReferences
            pooled_embeds: List[Tensor] = [torch.max(embeds, dim=0).values
                                                          if len(embeds) > 0 else zero_tensor
                                                          for embeds in embedding_lists]
        # len=batch_size, shapes=(embed_dim,)
        return torch.stack(pooled_embeds)  # shape=(batch_size, embed_dim)
    else:
        return torch.tensor([]).view(-1, embed_dim)


def weighted_average_aggregator(embedding_lists, weights=None, embed_dim=None):
    if embed_dim is None:
        embed_dim = embedding_lists[0].shape[-1]
    zero_tensor = torch.zeros((embed_dim,), dtype=torch.float, device=embedding_lists[0].device)
    if len(embedding_lists) > 0:
        if weights is not None:
            batch_means = [torch.mean(embeds * weight.view(-1,1), dim=0)
                           if len(embeds) > 0 else zero_tensor
                           for embeds, weight in zip(embedding_lists, weights)]
        else:
            batch_means = [torch.mean(embeds, dim=0)
                           if len(embeds) > 0 else zero_tensor
                           for embeds in embedding_lists]
        return torch.stack(batch_means)
    else:
        return torch.tensor([]).view(-1, embed_dim)


def fuzzy_set_aggregator():
    # TODO: SURFACE FORMS: implement fuzzy set aggregator
    pass


# Aggregators for combining entity/predicate representations (eg: encoded from text) with OpenKI embeddings
def sum_combiner(embeds1, embeds2):
    return embeds1 + embeds2


def max_pool_combiner(embeds1, embeds2):
    return torch.max(embeds1, embeds2)


def concat_combiner(embeds1, embeds2):
    return torch.cat(embeds1, embeds2)


class ConcatFcCombiner(nn.Module):
    def __init__(self, input_dim_1, input_dim_2, output_dim, activation=nn.Tanh(), layer_norm=False):
        super().__init__()
        self.FC = nn.Linear(input_dim_1 + input_dim_2, output_dim)
        self.activation = activation
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None

    def forward(self, input_1, input_2):
        if input_1 is not None:
            out = self.FC(torch.cat((input_1, input_2), dim=-1))
        else:
            out = self.FC(input_2)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return self.activation(out) if self.activation is not None else out


class SumCombiner(nn.Module):
    def forward(self, input_1, input_2) -> torch.Tensor:
        return input_1 + input_2


class ConcatenatedEmbedding(nn.Module):
    def __init__(self, embedding_1: nn.Embedding, embedding_2: nn.Module):
        super().__init__()
        self.embedding_1 = embedding_1
        self.embedding_2 = embedding_2
        self.embedding_dim = embedding_1.embedding_dim + embedding_2.embedding_dim
        self.encoder_type_1 = type(embedding_1)
        self.encoder_type_2 = type(embedding_2)

    def forward(self, indices: torch.IntTensor, *args, **kwargs):
        embeds = [None, None]
        seen = kwargs.pop("seen", None)
        only_seen_neighbours = kwargs.get("only_seen_neighbours", None)
        if only_seen_neighbours is None:
            only_seen_neighbours = seen is not None

        for i, embedding, embed_type in zip((0, 1),
                                            (self.embedding_1, self.embedding_2),
                                            (self.encoder_type_1, self.encoder_type_2)):

            if embed_type is nn.Embedding:
                if only_seen_neighbours and seen is not None and not torch.all(seen):
                    embed_shape = tuple(list(indices.shape) + [embedding.embedding_dim])
                    embeds[i] = torch.zeros(device=indices.device, size=embed_shape, dtype=torch.float32)
                    embeds[i][seen] = embedding(indices[seen])
                else:
                    embeds[i] = embedding(indices)
            elif embed_type is RelationEncoder:
                embeds[i] = embedding(indices.unsqueeze(-1), seen_mask=seen)
            elif embed_type is NeighbouringRelationsEntityEncoder:
                # NOTE: NeighbouringRelationsEntityEncoder takes care of seen relations on it's own
                #       It also accepts entity pair indices with shape (m, n, 2) or entity indices with shape (m, n)
                embeds[i] = embedding(indices, *args, **kwargs)
            elif embed_type is ConcatenatedEmbedding:
                embeds[i] = embedding(indices, seen=seen, *args, **kwargs)
        return torch.cat(embeds, dim=-1)

# # code for an "evaluate expression" pycharm debug window to look for non-zero grads
# # when detect_anomaly stops with a nan grad
# import logging
# tensors[0].grad_fn.next_functions[0][0].next_functions
# t_stack  = [(tensors[0].grad_fn, tensors[0].grad)]
# num_to_report = 100
# num_processed = 0  # 46757197 total grad fn's in graph from
# while len(t_stack):
#     gf, g = t_stack.pop()
#     if g != 0:  # and g is not None:
#         msg = "    "*len(t_stack) + f"{len(t_stack)}"
#         if getattr(gf, "metadata", None) is None:
#             msg = msg + f"NO-metadata\n  ---  grad {g}"
#         else:
#             msg = msg + gf.metadata["traceback_"][-1] + f"  ---  grad {g}"
#         msg += f"  --- #{num_processed}: grad {g}"
#         logging.warning(msg)
#         num_to_report -= 1
#         if num_to_report <= 0:
#             logging.warning(f"processed {num_processed} total...")
#             break
#     num_processed += 1
#     if not num_processed % 1e7:
#         logging.warning(f"processed {num_processed}...")
#     if not num_processed % 1e8:
#         break
#     if getattr(gf, "next_functions", None) is not None:
#         t_stack.extend(gf.next_functions)
# logging.warning(f"processed {num_processed}... all done!")