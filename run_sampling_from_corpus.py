import sys
import argparse
import utilities
# Parse Args
args = sys.argv[1:]
p = argparse.ArgumentParser(description='A script for constructing an NGram sampler and sampling from it.')
p.add_argument('-input_file', help='input filepath')
p.add_argument('-collapse_whitespace', help='convert all whitespace to a single space', action="store_true")
p.add_argument('-punct_as_newline', help='use punctuation (!?.) as sentence terminators in addition to newlines', action="store_true")
p.add_argument('-number_samples', help='number of samples from language model to put in output file')
p.add_argument('-ngram_order', help='length of n-grams (value of n)')
parsed_args = p.parse_args(args)

n = int(parsed_args.ngram_order)
number_samples = int(parsed_args.number_samples)

# Read file
corpus = open(parsed_args.input_file)
document = corpus.read()
preprocessor = utilities.DocumentPreProcessor()
sentences = preprocessor.preprocess(document)
sentences = [sentence for sentence in sentences if len(sentence.split('n')) > n]

# Build Samples
model = utilities.SentenceSamplerUtility(sentences, n)
samples = [model.get_sample() for i in range(number_samples)]
for s in samples:
    print(s)