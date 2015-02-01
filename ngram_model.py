from collections import defaultdict
from math import log

class NGramSequenceModel(object):
    "Maximum likelihood Ngram model."
    def __init__(self, N):
        self.N = N
        self.frequencies = defaultdict(lambda : 1e-10)
        self.total = 0
        self.STOP = 'STOP'
        self.FIRST_CHARACTER_TEMPLATE = '*_{0}'
        self.starting_tokens = [self.FIRST_CHARACTER_TEMPLATE.format(-(N-i)) for i in range(1, N)]


    def fit(self, sequences):
        for sequence in sequences:
            ngrams = self._make_ngrams(sequence)
            self.total += len(ngrams) - 1
            for ngram in ngrams:
                self.frequencies[ngram] += 1

    def predict(self, sequences):
        return [self._get_sequence_log_probability(sequence) for sequence in sequences]

    def _make_ngrams(self, sequence):
        ngrams = []
        augmented_sequence = self.starting_tokens + sequence + [self.STOP]
        starting_point_of_last_ngram = len(augmented_sequence) - (self.N - 1)
        for i in range(starting_point_of_last_ngram):
            ngrams.append(tuple(augmented_sequence[i:i+self.N]))
        return ngrams

    def _get_sequence_log_probability(self, sequence):
        ngrams = self._make_ngrams(sequence)
        return sum([self._get_ngram_log_probability(ngram) for ngram in ngrams])

    def _get_ngram_log_probability(self, ngram):
        return log(1.0*self.frequencies[ngram]/self.total)