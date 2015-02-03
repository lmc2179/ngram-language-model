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
        assert unigram_model.predict(test_sequences) ==  [-2.865628753509567, -15.356652498877365]
