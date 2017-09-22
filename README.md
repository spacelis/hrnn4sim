# hrnn4sim

Hierarchical RNN can be useful for encoding sequences of tokens where
tokens may share similar features when they are seen as sequences. The
examples of such tokens may be in the forms of compound words, mispelt
words and make up words.

## Dependencies

The code base is developed around Keras, Tensorflow and NLTK.

## Usage Example

The model was developed for matching addresses entered by users in different
databases. Most of time the addresses will be recorded in the exact same 
string of text. But occasionally, the text can be different for the same
addresses. A fuzzy matching algorithm backed by HRNN model can be useful,
as it can learn the the concept of words and their role in the address line.
