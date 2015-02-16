import sys
sys.path.append('..')
import unittest

import tokenizer

class TokenizerTest(unittest.TestCase):
    def test_tokenizer(self):
        test_strings = ['The man who is tall is happy.','Is the man who is tall happy?']
        tokenized_strings = [['The','man','who','is','tall','is','happy','.'],
                             ['Is', 'the', 'man','who','is','tall','happy','?']]
        T = tokenizer.Tokenizer()
        assert T.process(test_strings) == tokenized_strings

if __name__ == '__main__':
    unittest.main()