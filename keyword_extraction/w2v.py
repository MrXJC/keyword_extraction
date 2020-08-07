# -*- coding: utf-8 -*-
import os
from gensim.models import KeyedVectors
from gensim.matutils import unitvec
import numpy as np
from six import string_types
from copy import deepcopy
import mmap
import re
import jieba


def word_len(word):
    return len(re.findall('[a-zA-Z]', word))/3 + len(re.sub('[a-zA-Z]', '', word))


def get_oov_with_tokenize(w2v, word, tokenize=None, detail=False):
    word = word.lower()
    if word in w2v:
        return w2v[word]
    vec = None
    count = []
    if tokenize is None:
        tokenize = jieba.cut

    words = list(tokenize(word))
    if detail:
        print(words)
    for x in words:
        if x in w2v:
            length = word_len(word)
            if len(count) > 0:
                vec += w2v[x]*length
            else:
                vec = deepcopy(w2v[x])*length
            count.append(length)

    if len(count) == 0:
        for x in list(word):
            if x in w2v:
                length = word_len(word)
                if len(count) > 0:
                    vec += w2v[x]*length
                else:
                    vec = deepcopy(w2v[x])*length
                count.append(length)
        if len(count) == 0:
            return np.zeros(w2v.vector_size)
    vec = unitvec(vec/sum(count))
    return vec


class Embedding(object):

    """Docstring for Embedding. """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _load_vec(self):
        pass

    def __getitem__(self, word):
        pass

    def __contains__(self, word):
        pass

    def new_vector(self, v):
        return v if v is not None else np.random.randn(self.w2v.vector_size)

    @property
    def vector_size(self):
        pass

    def get_oov(self, word, detail=False):
        word = word.lower()
        if word in self:
            return self[word]
        vec = None
        count = []
        if self.tokenizer is None:
            words = list(jieba.cut(word))
        else:
            words = list(self.tokenizer.tokenize(word))
        if detail:
            print(words)
        for x in words:
            if x in self:
                length = word_len(word)
                if len(count) > 0:
                    vec += self[x]*length
                else:
                    vec = deepcopy(self[x])*length
                count.append(length)

        if len(count) == 0:
            for x in list(word):
                if x in self:
                    length = word_len(word)
                    if len(count) > 0:
                        vec += self[x]*length
                    else:
                        vec = deepcopy(self[x])*length
                    count.append(length)
            if len(count) == 0:
                return np.zeros(self.vector_size)
        vec = unitvec(vec/sum(count))
        return vec


class SwivelEmbedding(Embedding):

    def __init__(self, filename="../res/embedding/vecs", cols_filename=None, tokenizer=None):
        Embedding.__init__(self, tokenizer)
        self.vocab_filename = filename + '.vocab'
        self.rows_filename = filename + '.bin'
        self.cols_filename = cols_filename
        self._load_vec()

    def _load_vec(self):
        """Initializes the vectors from a text vocabulary and binary data."""
        with open(self.vocab_filename, 'r') as lines:
            self.idx2word = [line.split()[0] for line in lines]
            self.word2idx = {word: idx for idx, word in enumerate(self.idx2word)}

        n = len(self.idx2word)

        with open(self.rows_filename, 'r') as rows_fh:
            rows_fh.seek(0, os.SEEK_END)
            size = rows_fh.tell()

            # Make sure that the file size seems reasonable.
            if size % (4 * n) != 0:
                raise IOError(
                    'unexpected file size for binary vector file %s' % self.rows_filename)

            # Memory map the rows.
            dim = int(size / (4 * n))
            rows_mm = mmap.mmap(rows_fh.fileno(), 0, prot=mmap.PROT_READ)
            rows = np.matrix(
                np.frombuffer(rows_mm, dtype=np.float32).reshape(n, dim))

            # If column vectors were specified, then open them and add them to the
            # row vectors.
            if self.cols_filename:
                with open(self.cols_filename, 'r') as cols_fh:
                    cols_mm = mmap.mmap(cols_fh.fileno(), 0, prot=mmap.PROT_READ)
                    cols_fh.seek(0, os.SEEK_END)
                    if cols_fh.tell() != size:
                        raise IOError('row and column vector files have different sizes')

                    cols = np.matrix(
                        np.frombuffer(cols_mm, dtype=np.float32).reshape(n, dim))

                    rows += cols
                    cols_mm.close()

            # Normalize so that dot products are just cosine similarity.
            self.vecs = rows / np.linalg.norm(rows, axis=1).reshape(n, 1)
            self.vecs = self.vecs.A
            rows_mm.close()
        self._vector_size = self.vecs.shape[1]

    def neighbors(self, query, topn=10):
        """Returns the nearest neighbors to the query (a word or vector)."""
        if isinstance(query, string_types):
            idx = self.word2idx.get(query)
            if idx is None:
                return None

            query = self.vecs[idx]

        neighbors = np.dot(self.vecs, query.T)

        return sorted(
            zip(self.idx2word, neighbors.flat),
            key=lambda kv: kv[1], reverse=True)[:topn+1]

    def __getitem__(self, word):
        idx = self.word2idx.get(word)
        return self.new_vector() if idx is None else self.vecs[idx]

    def __contains__(self, word):
        return word in self.word2idx

    @property
    def vector_size(self):
        return self._vector_size


class GensimEmbedding(Embedding):

    def __init__(self, vec_path='', cache_vec_path='', logger=None, tokenizer=None):
        Embedding.__init__(self, tokenizer)
        self.vec_path = vec_path
        self.cache_vec_path = cache_vec_path
        self.logger = logger
        self._load_vec()
        self._load_cache_vec()

    def _load_vec(self):
        if os.path.exists(self.vec_path):
            print(os.path.splitext(self.vec_path))
            if os.path.splitext(self.vec_path)[-1] == ".model":
                self.w2v = KeyedVectors.load(self.vec_path)
            else:
                self.w2v = KeyedVectors.load_word2vec_format(self.vec_path, binary=False)
            if self.logger:
                self.logger.info("Step One: load w2v file successful!")
        else:
            if self.logger:
                self.logger.warning("Step X: the w2v file not exists ! ")
            raise FileNotFoundError("The w2v file is not exists ! ")

    def _load_cache_vec(self):
        if os.path.exists(self.cache_vec_path):
            # self.cache_vec = zload(self.cache_vec_path)
            pass
        else:
            self.cache_vec = {}

    def new_vector(self, v):
        return v if v is not None else np.random.randn(self.w2v.vector_size)

    def __getitem__(self, word):
        v = None
        if word is None:
            return None
        if word in self.cache_vec:
            return self.cache_vec[word]
        if word in self.w2v:
            v = self.w2v[word]
        v = self.new_vector(v)
        self.cache_vec[word] = v
        return v

    def __contains__(self, word):
        return word in self.w2v

    @property
    def vector_size(self):
        return self.w2v.vector_size


class FunctionEmbedding(Embedding):

    def __init__(self, vec_path='', cache_vec_path='', logger=None, tokenizer=None):
        Embedding.__init__(self, tokenizer)
        self.vec_path = vec_path
        self.cache_vec_path = cache_vec_path
        self.logger = logger
        self.w2v = dict()
        self._load_vec()
        self._load_cache_vec()

    def _load_vec(self):
        if os.path.exists(self.vec_path):
            for line in open(self.vec_path, encoding='utf8'):
                query, vec = line.strip().split('\t')
                self.w2v[query] = np.array([float(ii) for ii in vec.split(',')])
            self.length = len(self.w2v)
            self._vector_size = len(vec.split(','))

    def _load_cache_vec(self):
        if os.path.exists(self.cache_vec_path):
            # self.cache_vec = zload(self.cache_vec_path)
            pass
        else:
            self.cache_vec = {}

    def new_vector(self, v):
        return v if v is not None else np.zeros(self.vector_size)

    def __getitem__(self, word):
        v = None
        if word is None:
            return None
        if word in self.cache_vec:
            return self.cache_vec[word]
        if word in self.w2v:
            v = self.w2v[word]
        v = self.new_vector(v)
        self.cache_vec[word] = v
        return v

    def __contains__(self, word):
        return word in self.w2v

    @property
    def vector_size(self):
        return self._vector_size


if __name__ == '__main__':
    # emb = FunctionEmbedding('res/embedding/vec_.txt')
    # print(f"vector_size: {emb.vector_size}")
    # emb = GensimEmbedding('res/embedding/word2vec/word2vec.model')
    # print(f"vector_size: {emb.vector_size}")
    # emb = SwivelEmbedding('res/embedding/vecs')
    # print(f"vector_size: {emb.vector_size}")
    # vec = get_oov_with_tokenize(emb, '上海软件开发工程师')
    # print(emb.neighbors(vec))
    pass
