# ngram-language-model
An implementation of a HMM Ngram language model in python.

Currently implements basic NGram analysis, and provides an interface to create samplers from your favorite corpus.

Use run_sampling_from_corpus.py to create samples trained on a corpus in a text file.

For more info about the input arguments, type
```
run_sampling_from_corpus.py -h
```

For more control, you can import the SentenceSamplerUtility class from the utilities module, which provides a convenient wrapper around the mechanics of sampler construction.

-------------------------------------------------------

Some highlights from a trigram model trained on the collected works of Edgar Allan Poe from Project Gutenberg (included in the tests directory):
```
"And yet all was blackness and vacancy."
"Notwithstanding the obscurity which thus oppressed me."
"And it was the groan of mortal terror."
"Among this nation of necromancers there was wine."
```

Interestingly, only the shorter sentences seem vaguely comprehensible in this model. Poe's love of stringing short relative clauses together with commas causes longer sentences to descend into trippy comma-segmented context switches:
```
And so with combativeness, with a sudden elevation in turpitude, whose success at guessing in the contemplation of natural glory mingled at length, upon application of the Marchesa di Mentoni, (who for some time, much of what the more distressing, the more bitterly did I shudder to name them at all, the enthusiasm, and she trembled and very bitterly wept; but to define the day of my soul!
```
