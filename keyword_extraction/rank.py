# -*- coding: utf-8 -*-

import numpy as np
import math
from collections import Counter
from .w2v import GensimEmbedding
import copy
import itertools
from gensim.matutils import unitvec
EPS = 0.0000001

# c_matrix 共现矩阵 windows_size
# s_matrix 相似矩阵 w2v
# 都需要归一化
# n_vec 平均初始化
# p_vec 位置权重
# s_vec title 和desc 相关程度 similarity
# k_vec keyword in desc 打分
# matrix = (c_matrix*C_weight+s_matrix*S_weight) [C_weight, S_weight] => matrix_weights
# vec = n_vec*n_weight + p_vec*p_weight + s_vec*s_weight + k_vec*k_weight => vector_weights
# new_score = old_score*alpha + (1-alpha)*vec


class EnsembleRanker(object):
    def __init__(self, tokenizer, w2v=None, w2v_path=None, keyword_path=None):
        self.tokenizer = tokenizer
        if w2v_path:
            self.w2v = self.load_word2vec(w2v_path)
        else:
            self.w2v = w2v
        self.keyword_dict = self.load_keyword_dict(
            keyword_path) if keyword_path else None

    def keyword_rank(self,
                     sentence,
                     alpha=0.85,
                     beta_vector=1,
                     beta_matrix=1,
                     window_size=6,
                     iteration=20,
                     num_keyphrase=10,
                     threshold=0.001,
                     matrix_weights=[1, 0],
                     vector_weights=[0, 1, 0, 0],
                     topic=None,
                     with_weights=False,
                     merge_mode='mean',
                     is_expand=False,
                     pos_filter=None):
        """
        matrix_weights : weights of co-occurrence and w2v-cosine-similarity
        vector_weights: weights of avg, pos, sim, keyword features
        beta_matrix : the ratio between w2v-cosine-similarity and jaccard matrix
        beta_vector :
        merge_mode: mean or max
        """
        assert sum(matrix_weights) == 1 and sum(vector_weights) == 1
        if pos_filter:
            wordlist = self.tokenizer.pos_tokenize(sentence, pos_filter)
        else:
            wordlist = self.tokenizer.tokenize(sentence)

        if len(wordlist) == 0:
            wordlist = self.tokenizer.tokenize(sentence)
        # import ipdb; ipdb.set_trace()
        self._expand_words = []
        self._duplicate_words = []

        length = len(wordlist)
        self.word2idx = {w: i for i, w in enumerate(set(wordlist))}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        assert len(self.word2idx) == len(self.idx2word)
        n = len(self.word2idx)

        if n < 1:
            return list(set(wordlist))

        n_vec = np.ones(n) / n
        # initial the position weights
        p_vec = np.zeros(n)
        c_matrix = np.zeros((n, n))
        if matrix_weights[0] > EPS or vector_weights[1] > EPS:
            co_occ_dict = {k: [] for k, v in self.word2idx.items()}
            for i, w in enumerate(wordlist):
                if vector_weights[1] > EPS:
                    p_vec[self.word2idx[w]] += float(1 / (i + 1))
                if matrix_weights[0] <= EPS:
                    continue
                for window_idx in range(1, math.ceil(window_size / 2) + 1):
                    if i - window_idx >= 0:
                        co_occ_dict[w].append(wordlist[i - window_idx])
                    if i + window_idx < length:
                        co_occ_dict[w].append(wordlist[i + window_idx])
            if vector_weights[1] > EPS:
                p_vec = p_vec / np.maximum(p_vec.sum(), EPS)

            if matrix_weights[0] > EPS:
                for w, co_list in co_occ_dict.items():
                    cnt = Counter(co_list)
                    for co_word, freq in cnt.most_common():
                        c_matrix[self.word2idx[w]][
                            self.word2idx[co_word]] = freq
                c_matrix = c_matrix / np.maximum(c_matrix.sum(axis=0), EPS)
                c_matrix = c_matrix.T

        s_matrix = self.get_matrix_by_similarity(beta_matrix) if matrix_weights[1] > EPS else np.zeros((n, n))
        # get final matrix
        matrix = c_matrix * matrix_weights[0] + s_matrix * matrix_weights[1]

        s_vec = self.get_weight_between_words_and_topic(topic, beta_vector) if vector_weights[2] > EPS and topic else np.zeros(n)
        k_vec = self.get_weight_words_in_keyword() if vector_weights[3] > EPS and self.keyword_dict else np.zeros(n)

        # get final vector
        vec = n_vec * vector_weights[0] + p_vec * vector_weights[1] + s_vec * vector_weights[2] + k_vec * vector_weights[3]
        if np.sum(vec) == 0:
            vec = n_vec / n_vec.sum()
        else:
            vec = vec / vec.sum()
        self.p_vec, self.s_vec, self.k_vec, self.vec = p_vec, s_vec, k_vec, vec

        scores, _ = self.initial_scores(n)
        # compute final principal eigenvector
        for i in range(iteration):
            old_scores = copy.deepcopy(scores)
            scores = self.update_scores(matrix, vec, scores, alpha)
            diff = np.linalg.norm(old_scores - scores)
            if diff < threshold:
                break

        result = []
        for x in np.argsort(scores * -1):
            result.append([self.idx2word[x], scores[x]])
        self.result = result
        # self.correct()
        if is_expand:
            result = self.expand_words(result, sentence, num_keyphrase*2)
        self.expand_result = result
        keywords, word_count = self.deduplicate(result, s_matrix, num_keyphrase)
        self.keywords = keywords
        return keywords[: num_keyphrase] if with_weights else [x[0] for x in keywords][:num_keyphrase]

    def debug_by_word(self, word):
        idx = self.word2idx[word]
        print('p_vec: ', self.p_vec[idx], np.where(np.argsort(self.p_vec*-1) == idx)[0][0])
        print('s_vec: ', self.s_vec[idx], np.where(np.argsort(self.s_vec*-1) == idx)[0][0])
        print('k_vec: ', self.k_vec[idx], np.where(np.argsort(self.k_vec*-1) == idx)[0][0])
        print('vec: ', self.vec[idx], np.where(np.argsort(self.vec*-1) == idx)[0][0])

    def details(self):
        print('Expand Words:', self._expand_words)
        print('Dedupilicated Words', self._duplicate_words)

    def sentence_rank(self,
                      sentence,
                      alpha=0.85,
                      iteration=20,
                      num_keyphrase=10,
                      threshold=0.001,
                      matrix_weights=[1, 0],
                      vector_weights=[0, 1, 0, 0],
                      topic=None,
                      with_weights=False):
        pass

    def initial_scores(self, n):
        scores = np.ones(n) / n
        return scores, sum(scores)

    @staticmethod
    def update_score(matrix, vec, scores, i, alpha):
        """
        single update
        :param matrix:
        :param vec:
        :param i:
        :param alpha:
        :return:
        """
        score = (1 - alpha) * vec[i] + alpha * np.sum(matrix[:, 0] * scores.T)
        return score

    @staticmethod
    def update_scores(matrix, vec, scores, alpha):
        """
        multi update
        :param matrix:
        :param vec:
        :param scores:
        :param alpha:
        :return: new scores
        """
        scores = (1 - alpha) * vec + alpha * np.dot(scores, matrix)
        return scores

    def get_weight_between_words_and_topic(self, topic, beta_vector):
        n = len(self.word2idx)
        s_vec = np.zeros(n)
        # w2v raito 100%
        if beta_vector != 1 or self.w2v is None:
            for word, idx in self.word2idx.items():
                s_vec[idx] = self.calculate_jaccard_similarity(word, topic)

        if self.w2v:
            s_vec_w2v = np.zeros(n)
            topic_vec = self.sentence_vector(
                sen=self.tokenizer.tokenize(topic))
            for word, idx in self.word2idx.items():
                s_vec_w2v[idx] = self.calculate_cosine_similarity(
                    self.w2v.get_oov(word), topic_vec)
            s_vec = (1 - beta_vector) * s_vec + beta_vector * s_vec_w2v
        return s_vec

    def get_weight_words_in_keyword(self):
        n = len(self.word2idx)
        k_vec = np.zeros(n)
        for word, idx in self.word2idx.items():
            k_vec[idx] = self.keyword_dict.get(word, 0)
        k_vec = k_vec / np.maximum(np.sum(k_vec), EPS)
        return k_vec

    def get_matrix_by_similarity(self, beta_matrix):
        n = len(self.word2idx)
        s_matrix = np.zeros((n, n))
        # w2v ratio 100%
        if beta_matrix != 1 or self.w2v is None:
            s_matrix = self.get_matrix_by_jaccard()
        if self.w2v:
            s_matrix_w2v = self.get_matrix_by_w2v()
            s_matrix = (1-beta_matrix) * s_matrix + beta_matrix * s_matrix_w2v
        return s_matrix

    def get_matrix_by_w2v(self):
        num = len(self.word2idx)
        matrix = np.zeros([num, num])
        self.doc_vec = {
            value: self.w2v.get_oov(key)
            for key, value in self.word2idx.items()
        }
        for i, j in itertools.product(range(num), repeat=2):
            if i > j:
                matrix[i][j] = self.calculate_cosine_similarity(
                    self.doc_vec[i], self.doc_vec[j])
        matrix += matrix.T
        matrix = matrix / (matrix.sum(axis=0) + EPS)
        matrix = matrix.T
        return matrix

    def get_matrix_by_jaccard(self):
        num = len(self.word2idx)
        matrix = np.zeros([num, num])
        doc_vec = {value: key for key, value in self.word2idx.items()}
        for i, j in itertools.product(range(num), repeat=2):
            if i > j:
                matrix[i][j] = self.calculate_jaccard_similarity(doc_vec[i], doc_vec[j])
        matrix += matrix.T
        matrix = matrix / np.maximum(matrix.sum(axis=0), EPS)
        matrix = matrix.T
        return matrix

    def get_matrix_by_co(self, wordlist):
        pass

    def sentence_vector(self, sen):
        """
        :param sen:
        :return:
        """
        vec = np.array([self.w2v.get_oov(word) for word in sen if word in self.w2v])
        if len(vec) > 0:
            vec = np.sum(vec, axis=0) / len(vec)
        else:
            vec = np.zeros(self.w2v.vector_size)
        return vec

    @staticmethod
    def calculate_cosine_similarity(a, b):
        """
        :param a:
        :param a:
        :return:
        """
        a = unitvec(a)
        b = unitvec(b)
        return (np.dot(a, b) + 1)/2

    @staticmethod
    def calculate_jaccard_similarity(a, b):
        return len(set(a) & set(b)) / len(set(a) | set(b))

    @staticmethod
    def load_word2vec(path):
        if type(path) == str:
            return GensimEmbedding(path)
        return path

    @staticmethod
    def load_keyword_dict(path):
        return {
            x.split('\x01')[0]: int(x.split('\x01')[-1].strip())
            for x in open(path).readlines()
        }

    @staticmethod
    def correct():
        pass

    def deduplicate(self, result, s_matrix, num_keyphrase=10):
        data = [result[0]]
        idx = 0
        for idx, x in enumerate(result[1:]):
            if len(data) + len(result) - idx - 1 <= num_keyphrase:
                data.extend(result[idx+1:])
                break
            flag = True
            for y in data:
                # w2v
                if self.w2v:
                    vec = self.w2v.get_oov(x[0])
                    if self.calculate_cosine_similarity(self.w2v.get_oov(y[0]), vec) > 0.8:
                        if len(x[0]) > len(y[0]):
                            data.append(x)
                            data.remove(y)
                            self._duplicate_words.append(y)
                        flag = False
                        break

                if flag and min(len(x[0]), len(y[0])) - self.longestCommonSubsequence(x[0], y[0]) < 2:
                    if len(x[0]) > len(y[0]):
                        flag = False
                        data.append(x)
                        data.remove(y)
                        self._duplicate_words.append(y)
                    flag = False
                    break

            if flag:
                data.append(x)
            if len(data) == num_keyphrase:
                break
        return data, idx

    @staticmethod
    def longestCommonSubsequence(A: str, B: str) -> int:
        m, n = len(A), len(B)
        ans = 0
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if A[i - 1] == B[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    ans = max(ans, dp[i][j])
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return ans

    def expand_words(self, keywords, sentence, num_keyphrase):
        STATUS = "YN"
        status = STATUS[1]
        concat_dict = {'n', 'vn'}
        words = set()
        word = ''
        for x in self.tokenizer.pos_tokenizer.cut(sentence):
            if x.word in self.tokenizer.stop_words:
                words.add(word)
                word = ''
                status = STATUS[1]
                continue
            if status == STATUS[1]:
                if x.flag in concat_dict:
                    status = STATUS[0]
                    word += x.word
            elif status == STATUS[0]:
                if x.flag in concat_dict:
                    word += x.word
                else:
                    words.add(word)
                    word = ''
                    status = STATUS[1]

        if word != '':
            words.add(word)
        words = list(words)
        new_keywords = []
        for x in keywords:
            flag = True
            for y in words:
                if y != x[0] and ((len(x[0]) < 3 and y.endswith(x[0])) or (5 > len(x[0]) > 2 and x[0] in y)):
                    flag = False
                    self._expand_words.append(y)
                    new_keywords.append([y, x[1]])
                    words.remove(y)
            if flag:
                new_keywords.append(x)
            if len(new_keywords) == num_keyphrase:
                break
        return new_keywords

    def wer(s1, s2):
        # build mapping of words to integers
        return Lev.distance(s1, s2)


class TextRanker(object):
    def __init__(self, tokenizer):
        self.ranker = EnsembleRanker(tokenizer)

    def keyword_rank(self,
                     sentence,
                     alpha=0.85,
                     window_size=6,
                     iteration=20,
                     num_keyphrase=10,
                     threshold=0.001,
                     with_weights=False):
        return self.ranker.keyword_rank(sentence,
                                        alpha=alpha,
                                        window_size=window_size,
                                        iteration=iteration,
                                        num_keyphrase=num_keyphrase,
                                        threshold=threshold,
                                        with_weights=with_weights,
                                        matrix_weights=[1, 0],
                                        vector_weights=[1, 0, 0, 0],
                                        topic=None)
        pass

    def sentence_rank(self):
        pass


class PositionRanker(object):
    def __init__(self, tokenizer):
        self.ranker = EnsembleRanker(tokenizer)

    def keyword_rank(self,
                     sentence,
                     alpha=0.85,
                     window_size=6,
                     iteration=20,
                     num_keyphrase=10,
                     threshold=0.001,
                     with_weights=False):
        return self.ranker.keyword_rank(sentence,
                                        alpha=alpha,
                                        window_size=window_size,
                                        iteration=iteration,
                                        num_keyphrase=num_keyphrase,
                                        threshold=threshold,
                                        with_weights=with_weights,
                                        matrix_weights=[1, 0],
                                        vector_weights=[0, 1, 0, 0],
                                        topic=None)

    def sentence_rank(self):
        pass


class TextRankerW2V(object):
    def __init__(self, tokenizer, w2v):
        self.ranker = EnsembleRanker(tokenizer, w2v)

    def keyword_rank(self,
                     sentence,
                     alpha=0.85,
                     window_size=6,
                     iteration=20,
                     num_keyphrase=10,
                     threshold=0.001,
                     with_weights=False):
        return self.ranker.keyword_rank(sentence,
                                        alpha=alpha,
                                        window_size=window_size,
                                        iteration=iteration,
                                        num_keyphrase=num_keyphrase,
                                        threshold=threshold,
                                        with_weights=with_weights,
                                        matrix_weights=[0, 1],
                                        vector_weights=[1, 0, 0, 0],
                                        topic=None)
        pass

    def sentence_rank(self):
        pass


class PositionRankerW2V(object):
    def __init__(self, tokenizer, w2v):
        self.ranker = EnsembleRanker(tokenizer, w2v)

    def keyword_rank(self,
                     sentence,
                     alpha=0.85,
                     window_size=6,
                     iteration=20,
                     num_keyphrase=10,
                     threshold=0.001,
                     with_weights=False):
        return self.ranker.keyword_rank(sentence,
                                        alpha=alpha,
                                        window_size=window_size,
                                        iteration=iteration,
                                        num_keyphrase=num_keyphrase,
                                        threshold=threshold,
                                        with_weights=with_weights,
                                        matrix_weights=[0, 1],
                                        vector_weights=[0, 1, 0, 0],
                                        topic=None)
        pass

    def sentence_rank(self):
        pass
