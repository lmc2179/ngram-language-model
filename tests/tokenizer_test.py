import sys
sys.path.append('..')
import unittest

import tokenizer

class TokenizerTest(unittest.TestCase):
    def test_tokenizer(self):
        test_strings = ['a b c','d e f']
        tokenized_strings = [['a','b','c'], ['d', 'e', 'f']]
        T = tokenizer.Tokenizer()
        assert T.process(test_strings) == tokenized_strings

if __name__ == '__main__':
    unittest.main()