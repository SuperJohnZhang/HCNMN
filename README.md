# Toward multi-granularity decision-making: explicit visual reasoning with hierarchical knowledge

### Data and Knowledge Prepartion
1. Download and unpack Visual Genome images as well as the annotations, class info and image meta-data
2. Get ConceptNet from [conceptNet](https://github.com/commonsense/conceptnet5/) or through the [link](https://conceptnet-lite.fra1.cdn.digitaloceanspaces.com/conceptnet.db.zip)
3. Download [WikiText-2](https://huggingface.co/datasets/wikitext/tree/main)
4. Follow [MAVEx](https://github.com/jialinwu17/MAVEX) to preprocess the v features from VG
5. Download [Glove](https://nlp.stanford.edu/projects/glove/) pretrained word vectors
6. WordNet is available at [nltk](https://www.nltk.org/howto/wordnet.html) package


### Prepare Hierarchical Concept Graph
1. Preprocess questions to obtain train_questions.pt and vocab.json
```
python preprocess_questions.py --glove_pt </path/to/generated/glove/pickle/file> --input_questions_json </your/path/to/v2_OpenEnded_mscoco_train2014_questions.json> --input_annotations_json </your/path/to/v2_mscoco_train2014_annotations.json> --output_pt </your/output/path/train_questions.pt> --vocab_json </your/output/path/vocab.json> --mode train
``` 

2. Preprocess vocab.json with ontology information from wordNet to obtain augmented concepts (vocab.json) and the concept hierarchy (hierarchy.json)
```
python vocab_augmentation.py --input_vocab </your/input/path/vocab.json> --glove_pt </path/to/generated/glove/pickle/file> --vocab_json </your/output/path/vocab.json> --hierarchy </your/output/path/vocab.json> --wordnet_base
```

3. Combine the WordNet categorical hierarchy (hierarchy.json) with other downloaded knowledge sources to obtain the full hierarchical concept graph topology (topology.json) and the vocab list of cross-concept relationship (relation.json).
```
python knowledge_incorporation.py --input_vocab </your/input/path/vocab.json> --knowledge_dir </your/input/knowledge/dir> --glove_pt </path/to/generated/glove/pickle/file> --input_hierarchy </your/input/hierarchy.json> --topology_json </your/output/path/topology.json> --relation_vocab </your/output/path/relation.json> --hierarchy
```

4. Extract distinguishable concept property (concept_property.json) vectors and property list (property.json) by incorporation knowledge with all downloaded knowledge sources.
```
python knowledge_incorporation.py --input_vocab </your/input/path/vocab.json> --knowledge_dir </your/input/knowledge/dir> --glove_pt </path/to/generated/glove/pickle/file> --property_vocab </your/output/path/property.json> --concept_property </your/output/path/concept_property.json>  --property
```

5. Download grounded features from paper LXMERT [repo](https://github.com/airsplay/lxmert.git)

6. Preprocess features
```
python preprocess_features.py --input_tsv_folder /your/path/to/trainval_36/ --output_h5 /your/output/path/trainval_feature.h5
```
6. generate Hierarchical concept graph from all the downloaded data files to initialize concepts features
```
python preprocess_concepts.py --input_knowledge_folder /your/path/to/knowledge/sources --output_folder /your/output/path/concepts --glove_pt /your/path/to/glove/features mavex_pt /your/path/to/mavex/features
```

### Train and Evaluate the Model (HCNMN)

1. Train the model and get valuation results
```
python train.py --input_dir <path/to/preprocessed/data/folder> --concept <path/to/preprocessed/concept/folder> --concept_property <path/to/concept_property.json> --topology <path/to/topology.json> --relation_list <path/to/relation.json> --property_list <path/to/property.json> --save_dir </path/for/checkpoint> --model HCNMN  --T_ctrl 3 --stack_len 4 --cuda 1 --val
```

