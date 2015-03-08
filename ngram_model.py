from collections import defaultdict
from math import log
import sampler

STOP = 'STOP'

class NGramMaker(object):
    def __init__(self, N):
        self.N = N
        self.STOP = STOP
        self.FIRST_CHARACTER_TEMPLATE = '*_{0}'
        self.starting_tokens = [self.FIRST_CHARACTER_TEMPLATE.format(-(N-i)) for i in range(1, N)]

    def get_starting_tokens(self):
        return self.starting_tokens

    def get_stop_token(self):
        return self.STOP

    def make_ngrams(self, sequence):
        ngrams = []
        augmented_sequence = self.starting_tokens + sequence + [self.STOP]
        starting_point_of_last_ngram = len(augmented_sequence) - (self.N - 1) - 1
        for i in range(starting_point_of_last_ngram+1):
            ngrams.append(tuple(augmented_sequence[i:i+self.N]))
        return ngrams

class AbstractNGramFrequencyModel(object):
    """Model which stores the frequencies of ngrams occurring in a language.
    Also stores the frequency of frequencies"""
    def __init__(self, N):
        self.N = N
        self.ngram_maker = NGramMaker(N)
        self.frequency_tree = NGramFrequencyTree()

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

class NGramFrequencyTree(object):
    def __init__(self):
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

    def get_continuation_probability(self, ngram_stem, continuation):
        base_count, continuation_count = self.get_ngram_frequency(tuple(list(ngram_stem) + [continuation]))
        return 1.0*continuation_count/base_count

    def get_all_ngram_stems(self):
        return self.base_ngram_tree.keys()

    def get_all_continuations(self, ngram_stem):
        return self.frequency_tree[ngram_stem].keys()

    def _partition_ngram(self, ngram):
        *head, tail = ngram
        return tuple(head), tail

    def get_unique_count(self):
        return self.unique_ngram_count

class NGramSampler(object):
    def __init__(self, sequence_tree, default_initial_stem=None):
        self.sequence_tree = sequence_tree
        self.samplers = self._init_samplers(sequence_tree)
        self.default_initial_stem = default_initial_stem

    def _init_samplers(self, sequence_tree):
        return {key:self._build_sampler(sequence_tree, key) for key in sequence_tree.get_all_ngram_stems()}

    def _build_sampler(self, ngram_tree, ngram_stem):
        ngram_continuations = ngram_tree.get_all_continuations(ngram_stem)
        probabilities = [ngram_tree.get_continuation_probability(ngram_stem, cont) for cont in ngram_continuations]
        sampler_obj = sampler.Multinomial_Sampler(probabilities, ngram_continuations)
        return sampler_obj

    def sample_sequence(self):
        import copy
        sampled_sentence = copy.deepcopy(self.default_initial_stem)
        N = len(self.default_initial_stem) + 1
        while sampled_sentence[-1] != STOP:
            stem = tuple(sampled_sentence[-(N-1):])
            next_char = self.samplers[stem].sample()
            sampled_sentence.append(next_char)
        trimmed_sampled_sentence = sampled_sentence[len(self.default_initial_stem):-1]
        return trimmed_sampled_sentence