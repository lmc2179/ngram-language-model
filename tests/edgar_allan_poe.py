import sys
sys.path.append('..')
import unittest
import utilities
import re

class EdgarAllanPoeTest(unittest.TestCase):
    def test_corpus_sampling_end_to_end(self):
        filename = 'test_corpus/edgar_allan_poe.txt'
        self.run_corpus_trigram_sampling_end_to_end(filename)

    # @unittest.skip('Performance issues still need resolving')
    def test_corpus_sampling_end_to_end_long(self):
        filename = 'test_corpus/edgar_allan_poe_long.txt'
        self.run_corpus_trigram_sampling_end_to_end(filename)

    def run_corpus_trigram_sampling_end_to_end(self, filename):
        n=3
        poe_corpus = open(filename)
        poe_document = poe_corpus.read()
        poe_document = poe_document.replace('--', ' -- ')
        preprocessor = utilities.DocumentPreProcessor()
        poe_sentences = preprocessor.preprocess(poe_document)
        poe_sentences = [sentence for sentence in poe_sentences if len(sentence.split('n')) > n]
        print('Processing {0} sentences'.format(len(poe_sentences)))
        poe_model = utilities.SentenceSamplerUtility(poe_sentences, n)
        samples = [poe_model.get_sample() for i in range(100)]
        for s in samples:
            print(s)