class Tokenizer(object):
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
        return sequence.split(self.delimiter)
