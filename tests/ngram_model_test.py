import sys
sys.path.append('..')
import unittest

import ngram_model

class NgramTest(unittest.TestCase):
    def test_addone_unigram(self):
        train_sequences = [['the','dog','runs'],['the','dog','jumps']]*10
        test_sequences = [['the','dog','runs'],['ich','habe','eine','katze']]
        unigram_model = ngram_model.AddOneNgramModel(N=1)
        unigram_model.fit(train_sequences)
        assert unigram_model.predict(test_sequences) ==  [-3.989709101833799, -16.635532333438686]

    def test_frequency_tree(self):
        train_sequences = [['the','dog','runs'],['the','dog','jumps']]
        unigram_tree = ngram_model.NGramFrequencyTree(N=1)
        for sequence in train_sequences:
            for unigram in sequence:
                unigram_tree.add_ngram_observation([unigram])
        self._assert_ngram_frequency(unigram_tree, ['the'], 6, 2)


    def _assert_ngram_frequency(self, tree, sequence, expected_total, expected_sequence_count):
        sequence_total, sequence_count = tree.get_ngram_frequency(sequence)
        assert sequence_total == expected_total and sequence_count == expected_sequence_count

if __name__ == '__main__':
    unittest.main()
