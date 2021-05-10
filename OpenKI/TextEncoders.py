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
from abc import ABCMeta
from typing import Sequence

from OpenKI import logger
import re
from functools import partial
from pathlib import Path

import torch
from torch import nn
import numpy as np
import transformers

from OpenKI.OpenKI_Data import OpenKiGraphDataHandler
from OpenKI.UtilityFunctions import refuse_cuda


class CachedEmbeddings(nn.Embedding):
    def __init__(self, texts: 'Sequence' = None, requires_grad=True, fc_out_dim=None, cache_file_location: Path = None,
                 embeds_on_cpu=False, fc_layer_norm=False, no_activation=False, text_type_boundaries=None,
                 *args, **kwargs):
        """
        Orchestrates text embedding generation and application of a fully connected layer when fc_out_dim is provided.
        :param texts: A sequence of strings.
        :param requires_grad: As per nn.Embedding.
        :param fc_out_dim: If provided, an FC layer is applied after embedding lookup.
        :param cache_file_location: Name of the cache file in which to store these embeddings.
        :param embeds_on_cpu: To save GPU memory, do embedding lookup on CPU (bool). Not recommended!
        :param fc_layer_norm: Apply layer norm to the output of the FC layer (bool).
        :param no_activation: The FC layer will not be provided with an activation function (bool).
        :param text_type_boundaries: sequence of text indices. The FC layer will be provided with a flag indicating in
                which section of texts a provided text index lies.
        """
        # TODO: construct cache file name (location + embed_type + word_embed_name + "_cached.pt")
        # TODO: do lazy embeds also (check that full word embedding is smaller than embeds for each NYT text!)
        self.texts = texts
        self.text_type_boundaries = text_type_boundaries
        if cache_file_location is not None and cache_file_location.is_file():
            # NOTE: FastTextWordEmbedding sets word_embeddings to None
            logger.info(f"Loading cached entity word embeddings from {cache_file_location}")
            cached = torch.load(cache_file_location)
            super().__init__(num_embeddings=cached["num_embeds"], embedding_dim=cached["embed_dim"], padding_idx=-1,
                             *args, **kwargs)
            self.load_state_dict(cached["self"])
            self.load_cached_vars(cached)
        else:
            logger.info(f"Compiling text embeddings...")
            super_args, super_kwargs = self.setup_embedding_args(*args, **kwargs)
            super().__init__(*super_args, **super_kwargs)
            self.setup_embeddings(*args, **kwargs)
            if cache_file_location is not None:
                logger.info(f"Saving text embeddings to {cache_file_location}")
                torch.save({"num_embeds": self.num_embeddings, "embed_dim": self.embedding_dim,
                            "self": self.state_dict(), **self.vars_to_cache()},
                           cache_file_location)
        if fc_out_dim is not None:
            # nn.Linear already normalises by sqrt(out_dim) to obtain ENE std deviation of 1.0
            fc_in_dim = self.embedding_dim
            if self.text_type_boundaries:
                fc_in_dim += 1
            self.FC = nn.Linear(fc_in_dim, fc_out_dim)  # / math.sqrt(fc_out_dim)
            if no_activation:
                self.activation = None
            else:
                self.activation = nn.Tanh()
            if fc_layer_norm:
                self.fc_layer_norm = nn.LayerNorm(fc_out_dim)
            else:
                self.fc_layer_norm = None
            self.out_embed_dim = fc_out_dim
        else:
            self.FC = None
            self.out_embed_dim = self.embedding_dim
            if self.text_type_boundaries:
                self.out_embed_dim += 1
        self.weight.requires_grad = requires_grad
        self.embeds_on_cpu = embeds_on_cpu
        if embeds_on_cpu:  # monkeypatch cuda() on embedding weights so it never goes to GPU
            self.weight.cuda = refuse_cuda.__get__(self.weight)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.embeds_on_cpu:
            embeds = super().forward(indices.cpu()).to(indices.device)
        else:
            embeds = super().forward(indices)
        if self.text_type_boundaries:
            type_indicators = torch.zeros_like(indices)
            for boundary in self.text_type_boundaries:
                type_indicators += indices >= boundary
            embeds = torch.cat((embeds, type_indicators.to(torch.float32).unsqueeze(-1)), dim=-1)
        return self.apply_fc(embeds)

    def apply_fc(self, embeds):
        if self.FC is not None:
            # TODO: whats the expected norm of dot products of random normal vectors?
            #       how (and should) we normalise FC? Initialise with /sqrt(d)? normalise FC(embeds)?
            #       as it is, it essentially is a dimension selector? *** pytorch does this for us!!! ***
            #       Do we track gradients of FC? (well, it is degenerate even BEFORE any updates!)
            if self.fc_layer_norm is None:
                embeds = self.FC(embeds)
            else:
                embeds = self.fc_layer_norm(self.FC(embeds))
            if self.activation is None:
                return embeds
            else:
                return self.activation(embeds)
        else:
            return embeds

    def load_cached_vars(self, cached: dict):
        pass

    def vars_to_cache(self):
        return dict()

    def setup_embedding_args(self, *args, **kwargs):
        """
        returns a iterable (*args) and dict (**kwargs) of args for nn.Embedding (mostly num_embeds and embedding_dim)
        """
        raise NotImplementedError(f"Please implement setup_embedding_args() returning a dict of args for nn.Embedding")

    def setup_embeddings(self, *args, **kwargs):
        raise NotImplementedError(f"Please implement setup_embeddings() to populate self.weight of nn.Embedding")


class BertTextEmbedding(CachedEmbeddings):
    def __init__(self, bert_pipeline: 'str' = 'feature-extraction', fine_tune=False, cls_only=False,
                 cache_file_location: Path = None, bert_device=0, *args, **kwargs):
        # TODO: (spans) add spans metadata as constructor input and store in instance variable
        self.bert_pipeline = bert_pipeline
        self.bert = None
        self.fine_tune = fine_tune
        self.cls_only = cls_only
        self.bert_device = bert_device
        if self.fine_tune:
            logger.warning(f"finetuning BERT not implemented yet, ignoring")
        if self.cls_only:
            cache_file_location = re.sub('embeds', 'cls_only_embeds', str(cache_file_location))
            cache_file_location = Path(cache_file_location)
        super().__init__(cache_file_location=cache_file_location, *args, **kwargs)

    def setup_embedding_args(self, *args, **kwargs):
        self.bert = transformers.pipeline(self.bert_pipeline, device=self.bert_device)
        # NOTE: default 'feature-extraction' pipeline is a distilbert cased model
        # TODO: somehow bert ends up on CPU...
        kwargs['num_embeddings'] = len(self.texts)
        kwargs['embedding_dim'] = self.bert.model.config.dim  # this is 768
        return args, kwargs

    def setup_embeddings(self, *args, **kwargs):
        with torch.no_grad():
            sqrt_d = math.sqrt(self.bert.model.config.dim)
            for i, text in enumerate(self.texts):
                # TODO: (spans) get entity spans and take average of embeds in the spans
                #       But... for this we need an embedding for each entity span occurring in the training data...
                #       That means either storing tokenised and running BERT during training, or...
                #       Storing span embeds for each entity+text occurrence in the training data
                #       (ie: a dict keyed by ... entity ids?)
                #       Either way, we'd need a more complex lookup process.
                if text is None:
                    self.weight[i, :].fill_(0.)
                    continue
                # tokens = self.bert._parse_and_tokenize(text, truncation=True)
                tokens = self.bert.tokenizer(text=text, truncation=True, return_tensors='pt')
                bert_representation = self.bert._forward(tokens, return_tensors=True)
                if self.cls_only:
                    # CLS token representation only
                    self.weight[i, :] = bert_representation[0, 0, :]
                else:
                    # mean of word representations (without CLS and SEP)
                    # mean dividing by sqrt(n) instead of n makes norm of mean independent of n
                    # scaling by sqrt(embed_dim) makes expected norm of mean ~1.0
                    sqrt_n = math.sqrt(bert_representation.shape[-2])
                    self.weight[i, :] = bert_representation[:, 1:-1].mean(dim=1).squeeze() * sqrt_n / sqrt_d
            self.texts = None
            self.bert = None


# TODO: class BertSpansEmbedding(BertTextEmbedding):
# texts provides a sequence of text, spans. The embedding is then average BERT rep. of span tokens.
# setup_embedding_args(): 'num_embeddings' should be ok (since the sequence has texts repeated anyway)
# setup_embeddings(): here we get the sequence of spans alongside the text.
#                     if the span sequence is empty, I guess we default to mean/CLS as per current class

# The following borrowed from BoxWasserstein...
class TextEmbedding(CachedEmbeddings):
    def __init__(self, word_embeddings=None, vocab=None, word_delimiter: str = ' ', allow_missed=True,
                 aggregator=partial(torch.mean, dim=0), *args, **kwargs):
        """
        A class for whole text embeddings through averaging fixed token word embeddings. The averaged embeddings are
        cached so as to avoid calculating them each time.
        :param word_embeddings: word embedding lookup table (2d numpy array or tensor)
        :param vocab: token vocabulary, same length as first dim of word_embeddings
        :param texts: the texts to embed as an iterator of strings (or None if no text is available)
        :param word_delimiter: texts are tokenised by splitting on this string, defaults to ' '
        :param allow_missed: if false, raise an exception if a text has no tokens in vocab (default True, in which case
            token-less texts are represented by random embeddings and the number of such texts is reported)
        :param requires_grad:
        :param aggregator: function used to aggregate token embeddings for each text (default mean)
        :param fc_out_dim: if provided, embeddings are passed through a fully connected layer with tanh activation
        :param cache_file_location: location of cached aggregated embeddings
        :param embeds_on_cpu: if True, embeddings are kept on cpu (default False)
        :param args:
        :param kwargs:
        """
        # TODO: do lazy embeds also (check that full word embedding is smaller than embeds for each NYT text!)
        self.word_embeddings = word_embeddings
        self.vocab = vocab
        self.word_delimiter = word_delimiter
        self.allow_missed = allow_missed
        self.missed_entities = None
        self.aggregator = aggregator
        super().__init__(*args, **kwargs)

    def setup_embedding_args(self, *args, **kwargs):
        assert self.word_embeddings is not None and self.vocab is not None and self.texts is not None
        kwargs['num_embeddings'] = len(self.texts)
        kwargs['embedding_dim'] = self.word_embeddings.shape[1]
        return args, kwargs

    def setup_embeddings(self, *args, **kwargs):
        # super() above generates random (normal) embeddings - these remain for entities without embeddings
        self.missed_entities = 0  # should be missed_texts, but don't want to change cache file keys of same name
        with torch.no_grad():
            for i, text in enumerate(self.texts):
                if text is None:
                    self.weight[i, :].fill_(0.)
                    continue
                ent_words = text.split(self.word_delimiter)
                # embed_indexes = [emb_idx for emb_idx in (vocab.get(ent_word, None) for ent_word in ent_words)
                #               if emb_idx is not None]  # prefer for loop for easier debugging...
                embed_indexes = []
                for ent_word in ent_words:
                    emb_idx = self.vocab.get(ent_word, None)
                    if emb_idx:
                        embed_indexes.append(emb_idx)
                if embed_indexes:
                    # TODO: we could do  * sqrt_n / sqrt_d as per bert embeds... but it works so...
                    self.weight[i, :] = self.aggregator(self.word_embeddings[embed_indexes])
                else:
                    pass
                    self.missed_entities += 1  # in this case the default random (normal) embeddings remain
        if self.missed_entities > 0:
            if not self.allow_missed:
                raise ValueError(f"Missed {self.missed_entities} text embeddings with allow_missed=False!")
            logger.warning(f"Missed {self.missed_entities} text embeddings out of {self.num_embeddings}: using "
                           f"random gaussian embeddings!")

    def load_cached_vars(self, cached: dict):
        self.missed_entities = cached.get("missed_entities", None)

    def vars_to_cache(self):
        return {"missed_entities": self.missed_entities}


class RandomTextEmbedding(CachedEmbeddings):
    def __init__(self, embed_dim=768, cache_file_location: Path = None, *args, **kwargs):
        self.embed_dim = embed_dim
        cache_file_location = re.sub('embeds.pt$', f'{embed_dim}_embeds.pt', str(cache_file_location))
        cache_file_location = Path(cache_file_location)
        super().__init__(cache_file_location=cache_file_location, *args, **kwargs)

    def setup_embedding_args(self, *args, **kwargs):
        kwargs['num_embeddings'] = len(self.texts)
        kwargs['embedding_dim'] = self.embed_dim  # this is 768 for bert-like model
        return args, kwargs

    def setup_embeddings(self, *args, **kwargs):
        with torch.no_grad():
            for i, text in enumerate(self.texts):
                if text is None:
                    self.weight[i, :].fill_(0.)
                    continue
                self.weight[i, :].normal_() / math.sqrt(self.embed_dim)  # fill with mean zero, sd 1.0
            self.texts = None


class EntityEmbedding(CachedEmbeddings, metaclass=ABCMeta):
    def __init__(self, dataset: OpenKiGraphDataHandler, *args, **kwargs):
        """
        Create entity embeddings as the sum of pre-computed word embeddings of entity words.
        :param word_embeddings: a 2-d numpy array --- vocab_index x embedding_dims
                (use None to force loading cached embeddings)
        :param vocab: a dict with token string keys, vocab index int values
        :param dataset: a dataset with an entities variable which is a list of entity strings
        :param word_delimiter: entity words are separated by this character (default ' ')
        :param allow_missed: use random embeddings if no entity words have embeddings (default) else raise a ValueError.
        :param requires_grad: set requires_grad on the embedding weights to this value (default True)
        :param cache_file_location: file to load/store calculated embeddings
                (default None, overrides all other parameters when loading)
        """
        super().__init__(texts=dataset.entities, *args, **kwargs)


class RelationEmbedding(CachedEmbeddings, metaclass=ABCMeta):
    def __init__(self, dataset: OpenKiGraphDataHandler, *args, **kwargs):
        """
        Create entity embeddings as the sum of pre-computed word embeddings of entity words.
        :param word_embeddings: a 2-d numpy array --- vocab_index x embedding_dims
                (use None to force loading cached embeddings)
        :param vocab: a dict with token string keys, vocab index int values
        :param dataset: a dataset with an entities variable which is a list of entity strings
        :param word_delimiter: entity words are separated by this character (default ' ')
        :param allow_missed: use random embeddings if no entity words have embeddings (default) else raise a ValueError.
        :param requires_grad: set requires_grad on the embedding weights to this value (default True)
        :param cache_file_location: file to load/store calculated embeddings
                (default None, overrides all other parameters when loading)
        """
        super().__init__(texts=dataset.relation_texts, *args, **kwargs)


class FastTextEmbedding(TextEmbedding):
    def __init__(self, embedding_file: Path, cache_file_location: Path, *args, **kwargs):
        """
        Create entity embeddings as the sum of fasttext pre-computed word embeddings of entity words.
        :param embedding_file: is a fastText embeddings file: n,dims on first row; wd, embedding... on subsequent rows
        :param entities_file: see EntityEmbedding()
        :param dataset: a dataset with an entities variable which is a list of entity strings
        :param word_delimiter: entity words are separated by this character (default ' ')
        :param allow_missed: use random embeddings if no entity words have embeddings (default) else raise a ValueError.
        :param requires_grad: set requires_grad on the embedding weights to this value (default True)
        TODO: use fasttext's dynamic embeddings (currently we only use static 2M embeddings)
       """
        super().__init__(*args, **FastTextEmbedding.load_fast_text(embedding_file, cache_file_location),
                         cache_file_location=cache_file_location, **kwargs)

    @staticmethod
    def load_fast_text(embedding_file: Path, cache_file: Path):
        if cache_file.is_file():
            logger.info(f"Not loading fasttext embeddings since cached embeddings exist at {cache_file}")
            return {"word_embeddings": None, "vocab": None}
        logger.info(f"Loading fasttext embeddings from {embedding_file}")
        with embedding_file.open('r', encoding='utf-8', newline='\n', errors='ignore') as fin:
            try:
                # np.loadtxt is written in pure python anyway, best do our own!
                n, d = map(int, fin.readline().split())
                embeddings = np.zeros(shape=(n, d), dtype=np.float32)
                vocab = {}
                for i, row in enumerate(fin):
                    row_bits = row.rstrip().split(' ')
                    vocab[row_bits[0]] = i
                    embeddings[i, :] = [float(x) for x in row_bits[1:]]
                    # self.vocab.append(row_bits[0])
                assert len(row_bits) == d + 1, f"not enough embedding values {len(row_bits)}, expected {d+1} in " \
                                               f"row {row} with bits {row_bits}"
            except (IndexError, ValueError):
                if 'i' in vars():
                    print(i, row)
                else:
                    print(f"problem in first row of embedding file {embedding_file}")
                raise
        return {"word_embeddings": torch.from_numpy(embeddings), "vocab": vocab}

# fasttext_folder = Path("/home/ian/data/fasttext-embedding/crawl-300d-2M.vec")


class FastTextEntityEmbedding(EntityEmbedding, FastTextEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FastTextRelationEmbedding(RelationEmbedding, FastTextEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class FastTextRelationPairEmbedding:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Bert encoders for relation pairs not implemented.")


class BertEntityTextEmbedding(EntityEmbedding, BertTextEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BertRelationTextEmbedding(RelationEmbedding, BertTextEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BertRelationPairTextEmbedding:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Bert encoders for relation pairs not implemented.")


class RandomEntityTextEmbedding(EntityEmbedding, RandomTextEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomRelationTextEmbedding(RelationEmbedding, RandomTextEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
