from ast import arg
import sys
import argparse
import base64
import os
import csv
import itertools
import pickle
import json
from utils import *

csv.field_size_limit(sys.maxsize)

import h5py
import torch.utils.data
import numpy as np
from tqdm import tqdm

# concept parser
import conceptnet_lite
# wikitext parser
import wikitextparser as wtp
from concept_score import Tfidf, Rels, thres
from utils.misc import create_PV, create_RM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_vocab', required=True, help='path to vocab.json')
    parser.add_argument('--input_knowledge_folder', required=True, help='path to knowledge sources')
    parser.add_argument('--glove_pt', help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--input_hierarchy', help='path to input hierarchy')
    parser.add_argument('--vocab_json', required=True, help='path to output vocab')
    parser.add_argument('--topology_json', required=True, help='path to output topology_json')
    parser.add_argument('--relation_vocab', required=True, help='path to output relation_vocab')
    parser.add_argument('--concept_property', required=True, help='path to output concept_property')
    parser.add_argument('--property_vocab', required=True, help='path to output property_vocab')
    parser.add_argument('--property', action='store_true')
    parser.add_argument('--hierarchy', action='store_true')
    args = parser.parse_args()
    assert os.path.isdir(args.input_knowledge_folder)

    if args.property:
        concept_property, property_vocab = MatchPropertyByVocab(args)
        print('write')
        with open(args.concept_property, 'w') as f:
            json.dump(concept_property, f)

        with open(args.property_vocab, 'w') as f:
            json.dump(property_vocab, f)

    elif args.hierarchy:
        topology_json, relation_vocab = MatchRelationByVocab(args)
        print('write')
        with open(args.concept_property, 'w') as f:
            json.dump(topology_json, f)

        with open(args.property_vocab, 'w') as f:
            json.dump(relation_vocab, f)