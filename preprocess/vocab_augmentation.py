#!/usr/bin/env python3
import re
import os
import argparse
import json
import numpy as np
import pickle
from collections import Counter
import utils
from nltk.corpus import wordnet as wn

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])

parser = argparse.ArgumentParser()
parser.add_argument('--input_vocab', required=True, help='path to vocab.json')
parser.add_argument('--glove_pt', help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
parser.add_argument('--vocab_json', required=True, help='path to output vocab')
parser.add_argument('--hierarchy', required=True, help='path to output hierarchy')
args = parser.parse_args()
assert os.path.isdir(args.input_knowledge_folder)

augmentation_vocab = {}

print('Loading vocab')
with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)
for term in vocab['terms']:
    synsets = wn.synsets(term)
    relevant_terms = FindRelevantTerms(synsets)
    augmentation_vocab[term] = relevant_terms

#augmented_terms_perquestion = []
for i in range(len(vocab['term_per_question'])):
    augmented_terms = [augmentation_vocab[term] for term in terms]
    augmented_terms = flatten(augmented_terms)
    vocab['term_per_question'][i] = augmented_terms


print('write vocab')
with open(args.vocab_json, 'w') as f:
    json.dump(vocab, f)

with open(args.hierarchy, 'w') as f:
    json.dump(augmentation_vocab, f)