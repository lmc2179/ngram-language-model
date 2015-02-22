from collections import defaultdict
from math import log

class NGramMaker(object):
    def __init__(self, N):
        self.N = N
        self.STOP = 'STOP'
        self.FIRST_CHARACTER_TEMPLATE = '*_{0}'
        self.starting_tokens = [self.FIRST_CHARACTER_TEMPLATE.format(-(N-i)) for i in range(1, N)]

    def make_ngrams(self, sequence):
        ngrams = []
        augmented_sequence = self.starting_tokens + sequence + [self.STOP]
        starting_point_of_last_ngram = len(augmented_sequence) - (self.N - 1) - 1
        for i in range(starting_point_of_last_ngram):
            ngrams.append(tuple(augmented_sequence[i:i+self.N]))
        return ngrams

class AbstractNGramFrequencyModel(object):
    """Model which stores the frequencies of ngrams occurring in a language.
    Also stores the frequency of frequencies"""
    def __init__(self, N):
        self.N = N
        self.ngram_maker = NGramMaker(N)
        self.frequency_tree = NGramFrequencyTree(N)

    def fit(self, sequences):
        for sequence in sequences:
            ngrams = self._make_ngrams(sequence)
            [self.frequency_tree.add_ngram_observation(ngram) for ngram in ngrams]

    def predict(self, sequences):
        return [self._get_sequence_log_probability(sequence) for sequence in sequences]

    def _get_sequence_log_probability(self, sequence):
        ngrams = self._make_ngrams(sequence)
        log_likelihood = sum([self._get_ngram_log_probability(ngram) for ngram in ngrams])
        return log_likelihood

    def _make_ngrams(self, sequence):
        return self.ngram_maker.make_ngrams(sequence)

    def _get_ngram_log_probability(self, ngram):
        return log(self._get_ngram_probability(ngram))

    def _get_ngram_probability(self, ngram):
        raise NotImplementedError

class AddOneNgramModel(AbstractNGramFrequencyModel):
    def _get_ngram_probability(self, ngram):
        total, frequency = self.frequency_tree.get_ngram_frequency(ngram)
        adjusted_frequency = frequency+1
        adjusted_total = total + self.frequency_tree.get_unique_count()
        return 1.0*(adjusted_frequency)/adjusted_total

class NGramFrequencyTree(object):
    def __init__(self, N):
        self.N = N
        self.base_ngram_tree = defaultdict(int) #TODO: Consolidate as one data structure with custom node type
        self.frequency_tree = defaultdict(lambda: defaultdict(int))
        self.unique_ngram_count = 0

    def add_ngram_observation(self, ngram):
        preceding_elements, last_element = self._partition_ngram(ngram)
        self.base_ngram_tree[preceding_elements] += 1
        if self.frequency_tree[preceding_elements][last_element] == 0:
            self.unique_ngram_count += 1
        self.frequency_tree[preceding_elements][last_element] += 1

    def get_ngram_frequency(self, ngram):
        preceding_elements, last_element = self._partition_ngram(ngram)
        return self.base_ngram_tree[preceding_elements], self.frequency_tree[preceding_elements][last_element]

    def _partition_ngram(self, ngram):
        *head, tail = ngram
        return tuple(head), tail

    def get_unique_count(self):
        return self.unique_ngram_count

