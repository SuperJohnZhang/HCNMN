import json
import numpy as np
import pickle
"""
Utilities for preprocessing sequence data.

Special tokens that are in all dictionaries:

<NULL>: Extra parts of the sequence that we should ignore
<START>: Goes at the start of a sequence
<END>: Goes at the end of a sequence, before <NULL> tokens
<UNK>: Out-of-vocabulary words
"""

SPECIAL_TOKENS = {
  '<NULL>': 0,
  '<START>': 1,
  '<END>': 2,
  '<UNK>': 3,
}


def tokenize(s, delim=' ',
      add_start_token=True, add_end_token=True,
      punct_to_keep=None, punct_to_remove=None):
  """
  Tokenize a sequence, converting a string s into a list of (string) tokens by
  splitting on the specified delimiter. Optionally keep or remove certain
  punctuation marks and add start and end tokens.
  """
  if punct_to_keep is not None:
    for p in punct_to_keep:
      s = s.replace(p, '%s%s' % (delim, p))

  if punct_to_remove is not None:
    for p in punct_to_remove:
      s = s.replace(p, '')

  # if delim='' then regard the whole s as a token
  tokens = s.split(delim) if delim else [s]
  if add_start_token:
    tokens.insert(0, '<START>')
  if add_end_token:
    tokens.append('<END>')
  return tokens


def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None, add_special=None):
  token_to_count = {}
  tokenize_kwargs = {
    'delim': delim,
    'punct_to_keep': punct_to_keep,
    'punct_to_remove': punct_to_remove,
  }
  for seq in sequences:
    seq_tokens = tokenize(seq, **tokenize_kwargs,
                    add_start_token=False, add_end_token=False)
    for token in seq_tokens:
      if token not in token_to_count:
        token_to_count[token] = 0
      token_to_count[token] += 1

  token_to_idx = {}
  if add_special:
    for token in SPECIAL_TOKENS:
      token_to_idx[token] = len(token_to_idx)
  for token, count in sorted(token_to_count.items()):
    if count >= min_token_count:
      token_to_idx[token] = len(token_to_idx)

  return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
  seq_idx = []
  for token in seq_tokens:
    if token not in token_to_idx:
      if allow_unk:
        token = '<UNK>'
      else:
        raise KeyError('Token "%s" not in vocab' % token)
    seq_idx.append(token_to_idx[token])
  return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
  tokens = []
  for idx in seq_idx:
    tokens.append(idx_to_token[idx])
    if stop_at_end and tokens[-1] == '<END>':
      break
  if delim is None:
    return tokens
  else:
    return delim.join(tokens)

def FindRelevantTerms():
  return

def MatchPropertyByVocab(args):
  glove = pickle.load(open(args.glove_pt, 'rb'))
  p2c = args.input_knowledge_folder + 'conceptnet.json'
  p2w = args.input_knowledge_folder + 'wikitext.json'
  with open(p2c, 'rb') as f:
    cn = json.load(f)

  with open(p2w, 'rb') as f:
    wi = json.load(f)
  
  with open(args.vocab_json, 'rb') as f:
    vocabs = json.load(f) 
  
  # we first create a dictionary to map each term with its relevant distinguishable property
  term2property={}
  for term in vocabs['terms']:
    cn_property = cn[term]['hasProperty']
    wi_text = wi[term]
    relevant_properties = [property for property in cn_property if property in wi_text]
    term2property[term] = relevant_properties

  property_vocab = {}
  all_property_vocab = sorted([term2property.values()])
  property_vocab['properties'] = all_property_vocab
  # convert term2propery to vector
  for term in vocabs['terms']:
    pv = np.zeros(len(all_property_vocab))
    relevant_properties = term2property[term]
    for i in range(len(all_property_vocab)):
      if all_property_vocab[i] in relevant_properties:
        pv[i]=1
    term2property[term] = pv
  # next create a copy of property vectors corresponds to each concept per question.
  concept_property = {}
  concept_property['property_vector_per_question'] = []
  for terms in vocabs['term_per_question']:
    concept_property.append([term2property[term] for term in terms])
  concept_property['property_vector_per_question'] = np.array(concept_property['property_vector_per_question'])
  property_vocab['properties'] = glove(all_property_vocab)
  return property_vocab, concept_property

def MatchRelationByVocab(args):
  glove = pickle.load(open(args.glove_pt, 'rb'))
  p2c = args.input_knowledge_folder + 'conceptnet.json'
  p2w = args.input_knowledge_folder + 'wikitext.json'

  with open(p2c, 'rb') as f:
    cn = json.load(f)

  with open(p2w, 'rb') as f:
    wi = json.load(f)
  
  with open(args.vocab_json, 'rb') as f:
    vocabs = json.load(f)

  # we first extract a list of all the relevant relationships
  relation = sorted(cn['relation']+wi['relation'])
  relation_vocab = {}
  

  # next create a copy of concept affinity matrix corresponds to each concept per question.
  topology_json = {}
  topolopy_by_questions = []
  for terms in vocabs['term_per_question']:
    affinity = np.zeros([len(terms), len(terms)])
    t_ids = get_ids_from_list(terms, vocabs['terms'])
    for i in range(len(terms)):
      t_id =t_ids[i]
      relations = sorted(cn['relation'][terms[i]]+wi['relation'][terms[i]])
      for relation in relations:
        r_id = get_id_from_list(relation, relation_vocab)
        neighbors = cn['relation'][relation][terms[i]]
        t_sub = [term for term in terms if term in neighbors]
        affinity[t_id][t_sub]==r_id
    topolopy_by_questions.append(affinity)
    topology_json['topology_per_question'] = np.array(topolopy_by_questions)
  relation_vocab['relation'] = glove(relation)
  return topology_json, relation_vocab

def get_id_from_list(entry, full):
  for i in range(len(full)):
    if full[i] == entry:
        return i
  return -1

def get_ids_from_list(entry, full):
  ids = -np.ones_like(entry)
  for e in range(len(entry)):
    for i in range(len(full)):
      if full[i] == entry[e]:
          ids[e] == i
  return ids[i]