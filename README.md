# HRNN

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

### Running address matching on Google Cloud

Make sure you have setup Google Cloud SDK on your computer and followed [the
guide for ML Engine](https://cloud.google.com/ml-engine/docs/quickstarts/command-line).
In the root directory of this codebase, run the following command

```
gcloud ml-engine jobs submit training hrnn_test_train8 \
  --region=us-east1 \
  --scale-tier=BASIC_GPU \
  --runtime-version=1.2 \
  --job-dir=gs://hrnn/hrnn_test_train8 \
  --package-path=hrnn4sim \
  --module-name=hrnn4sim.example.addr_match \
  -- -m HRNN -e 2 gs://hrnn/data/training_examples.csv
```

or, if you prefer CPU-only version

```
gcloud ml-engine jobs submit training hrnn_test_train8 \
  --region=us-east1 \
  --runtime-version=1.2 \
  --job-dir=gs://hrnn/hrnn_test_train8 \
  --package-path=hrnn4sim \
  --module-name=hrnn4sim.example.addr_match \
  -- -m HRNN -e 2 gs://hrnn/data/training_examples.csv
```
