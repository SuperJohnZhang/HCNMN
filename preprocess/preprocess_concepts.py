# this file preprocess multiple external knowledge sources and store concepts in h5py


from ast import arg
import sys
import argparse
import base64
import os
import csv
import itertools
import pickle
import json

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
    parser.add_argument('--glove_pt', help='glove pickle file, should be a map whose key are words and value are word vectors represented by numpy arrays. Only needed in train mode')
    parser.add_argument('--vocab_json', required=True, help='path to output vocab')
    parser.add_argument('--hierarchy', required=True, help='path to output hierarchy')
    args = parser.parse_args()
    assert os.path.isdir(args.input_knowledge_folder)

    p2c = args.input_knowledge_folder + 'conceptnet.json'
    p2w = args.input_knowledge_folder + 'wikitext.json'
    q_gqa = "path/to/gqa_question"
    q_vqa = "path/to/vqa_question"

    alpha = np.array([0.6, 0.8])

    print('loading questions from %s' % (q_gqa+ '\\' +q_vqa))
    questions = []
    #terms = []
    with open(q_gqa, 'rb') as f:
        obj = pickle.load(f)
        gqa_questions = obj['questions']
        # gqa_questions_len = obj['questions_len']
        # gqa_q_image_indices = obj['image_idxs']
        gqa_answers = obj['answers']
        questions.append(gqa_questions+gqa_answers)
        # terms+=gqa_questions.split()
        # terms+=gqa_answers.split()

    with open(q_vqa, 'rb') as f:
        obj = pickle.load(f)
        vqa_questions = obj['questions']
        # vqa_questions_len = obj['questions_len']
        # vqa_q_image_indices = obj['image_idxs']
        vqa_answers = obj['answers']
        questions.append(vqa_questions+vqa_answers)
        # terms+=vqa_questions.split()
        # terms+=vqa_answers.split()

    #terms, counts = np.unique(terms)
    glove = pickle.load(open(args.glove_pt, 'rb'))
    mavex = pickle.load(open(args.mavex_pt, 'rb'))

    # extract terms as the Cid of concepts
    with open(p2c, 'rb') as f:
        cn = json.load(f)

    with open(p2w, 'rb') as f:
        wi = json.load(f)
    
    objects = np.unique(cn['object']+wi['object'])
    properties = np.unique(np.unique(cn['property'])+wi['property'])
    relationships = np.unique(np.unique(cn['relation']+wi['property']))

    terms_pool = [objects, properties, relationships]
    scores = []

    for i in range(3):
        candidate = terms_pool[i]

        tfidf = Tfidf()
        response1 = tfidf.fit_transform(candidate, questions)

        refs = Rels()
        response2 = refs.fit_transform(candidate, questions)

        scores.append(response1*alpha[0]+response2*alpha[1])


    o_terms = objects[scores[0]>thres[0]]
    p_terms = properties[scores[1]>thres[2]]
    r_terms = relationships[scores[1]>thres[2]]


    # specify feature size
    num_terms = [len(o_terms), len(p_terms), len(r_terms)]
    dim_v = 2048
    dim_l = 300

    # process o terms
    path = args.output_folder
    with h5py.File(path+'o_concepts.h5', 'w', libver='latest') as fd:
        v_features = fd.create_dataset('v_features', shape=(num_terms[0]. dim_v), dtype='float32')
        l_features = fd.create_dataset('f_features', shape=(num_terms[0]. dim_l), dtype='float32')
        Cids = fd.create_dataset('Cids', shape=(num_terms[0],), dtype='int32')
        types = fd.create_dataset('types', shape=(num_terms[0],), dtype='int32')
        RM = fd.create_dataset('widths', shape=(num_terms[2],num_terms[0]), dtype='float32')
        PV = fd.create_dataset('heights', shape=(num_terms[1],), dtype='float32')

        for i in range(len(o_terms)):
            Cids[i] = i
            l_features[i] = glove(o_terms[i])
            v_features[i] = mavex(o_terms[i])
            types[i] = 0
            RM[i] = create_RM(o_terms[i], 0, cn, wi)
            PV[i] = create_PV(o_terms[i], 0, cn, wi)

    # process p terms
    path = args.output_folder
    with h5py.File(path+'p_concepts.h5', 'w', libver='latest') as fd:
        v_features = fd.create_dataset('v_features', shape=(num_terms[1]. dim_v), dtype='float32')
        l_features = fd.create_dataset('f_features', shape=(num_terms[1]. dim_l), dtype='float32')
        Cids = fd.create_dataset('Cids', shape=(num_terms[1],), dtype='int32')
        types = fd.create_dataset('types', shape=(num_terms[1],), dtype='int32')
        PV = fd.create_dataset('heights', shape=(num_terms[0],), dtype='float32')

        for i in range(len(p_terms)):
            Cids[i] = i
            l_features[i] = glove(p_terms[i])
            v_features[i] = mavex(p_terms[i])
            types[i] = 0
            RM[i] = create_RM(p_terms[i], 1, cn, wi)
            PV[i] = create_PV(p_terms[i], 1, cn, wi)

    # process r terms
    path = args.output_folder
    with h5py.File(path+'r_concepts.h5', 'w', libver='latest') as fd:
        v_features = fd.create_dataset('v_features', shape=(num_terms[2]. dim_v), dtype='float32')
        l_features = fd.create_dataset('f_features', shape=(num_terms[2]. dim_l), dtype='float32')
        Cids = fd.create_dataset('Cids', shape=(num_terms[2],), dtype='int32')
        types = fd.create_dataset('types', shape=(num_terms[2],), dtype='int32')
        RM = fd.create_dataset('widths', shape=(num_terms[0],num_terms[0]), dtype='float32')

        for i in range(len(r_terms)):
            Cids[i] = i
            l_features[i] = glove(r_terms[i])
            v_features[i] = mavex(r_terms[i])
            types[i] = 0
            RM[i] = create_RM(r_terms[i], 2, cn, wi)
            PV[i] = create_PV(r_terms[i], 2, cn, wi)


if __name__ == '__main__':
    main()