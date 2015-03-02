import tokenizer
import ngram_model
import re

PUNCTUATION = ',.\'\\";:/?!'

class SentenceSamplerUtility(object):
    def __init__(self, sentences, n):
        ngram_maker = ngram_model.NGramMaker(n)
        ngram_tree = self._construct_ngram_tree_from_sentences(sentences, ngram_maker)
        self.sampler = ngram_model.NGramSampler(ngram_tree, default_initial_stem=ngram_maker.starting_tokens)

    def _construct_ngram_tree_from_sentences(self, sentences, ngram_maker):
        T = tokenizer.Tokenizer()
        ngram_tree = ngram_model.NGramFrequencyTree()

        tokenized_sentences = T.process(sentences)
        ngram_sequences = [ngram_maker.make_ngrams(tokenized_sentence) for tokenized_sentence in tokenized_sentences]
        [ngram_tree.add_ngram_observation(ngram) for sequence in ngram_sequences for ngram in sequence]
        return ngram_tree

    def get_sample(self):
        sampled_sequence = self.sampler.sample_sequence()
        sampled_sentence = ' '.join(sampled_sequence)
        for p in PUNCTUATION:
            sampled_sentence = sampled_sentence.replace(' '+p, p)
        return sampled_sentence

class DocumentPreProcessor(object):
    END_OF_SENTENCE_CHARS = '?.!'

    def __init__(self, reduce_whitespace=True, add_line_end_to_punctuation=True):
        self.preprocess_methods = []
        if reduce_whitespace:
            self.preprocess_methods.append(self._reduce_whitespace)
        if add_line_end_to_punctuation:
            self.preprocess_methods.append(self._add_line_end_to_punctuation)
        self.preprocess_methods.append(self.split_on_newline)

    def preprocess(self, document):
        for preprocess in self.preprocess_methods:
            document = preprocess(document)
        return document

    def _add_line_end_to_punctuation(self, document):
        processed_document = document.replace('\n', ' ')
        for eos in self.END_OF_SENTENCE_CHARS:
            processed_document = processed_document.replace(eos+' ', eos+'\n')
        return processed_document

    def _reduce_whitespace(self, document):
        return re.sub("\s+"," ", document)

    def split_on_newline(self, document):
        return document.split('\n')