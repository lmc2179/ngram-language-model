import copy

class Tokenizer(object):
    PUNCTUATION = ',.\'\\";:/?!'
    def __init__(self, delimiter=' '):
        self.delimiter = delimiter

    def process(self, sequences):
        tokenized_sequences = []
        for seq in sequences:
            preprocessed = self._preprocess_sequence(seq)
            tokenized = self._tokenize(preprocessed)
            tokenized_sequences.append(tokenized)
        return tokenized_sequences

    def _preprocess_sequence(self, seq):
        return seq

    def _tokenize(self, sequence):
        preprocessed_sequence = self._preprocess_punctuation(sequence)
        return preprocessed_sequence.split(self.delimiter)

    def _preprocess_punctuation(self, sequence):
        sequence_copy = copy.deepcopy(sequence)
        for punctuation_mark in self.PUNCTUATION:
            sequence_copy = sequence_copy.replace(punctuation_mark, self.delimiter + punctuation_mark)
        return sequence_copy
