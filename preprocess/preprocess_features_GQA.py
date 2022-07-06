import numpy as np
import h5py
import base64
import json
import csv
from tqdm import tqdm
import itertools
import sys
import os
maxInt = sys.maxsize
csv.field_size_limit(maxInt)

fd = open("/media/largeDisk/yifeng/GQA/vg_gqa_imgfeat/vg_gqa_obj36.tsv", 'r')
# train_scene_graph_file = open("/media/largeDisk/yifeng/GQA/sceneGraphs/train_sceneGraphs.json")
# train_scene_graph = json.load(train_scene_graph_file)
# val_scene_graph_file = open("/media/largeDisk/yifeng/GQA/sceneGraphs/val_sceneGraphs.json")
# val_scene_graph = json.load(val_scene_graph_file)

path = "data/kg.h5"

objects_path = "path/to/objects"

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]

reader = csv.DictReader(fd, delimiter='\t', fieldnames=FIELDNAMES)
readers = []
readers.append(reader)
reader = itertools.chain.from_iterable(readers)
features_shape = (
    148854,
    2048, # dim_vision,
    36, # 36 for fixed case, 100 for the adaptive case
)
boxes_shape = (
    features_shape[0],
    4,
    36,
)

with h5py.File(path, 'w', libver='latest') as fd:
    features = fd.create_dataset('features', shape=features_shape, dtype='float32')
    image_ids = fd.create_dataset('ids', shape=(features_shape[0],), dtype='S10')
    boxes = fd.create_dataset('boxes', shape=boxes_shape, dtype='float32')
    widths = fd.create_dataset('widths', shape=(features_shape[0],), dtype='int32')
    heights = fd.create_dataset('heights', shape=(features_shape[0],), dtype='int32')
    kgs = fd.create_dataset('kgs', shape=(features_shape[0],features_shape[2],features_shape[2]), dtype='int32')
    for i, item in enumerate(tqdm(reader, total=features_shape[0])):
        image_ids[i] = item['img_id']
        widths[i] = item['img_h']
        heights[i] = item['img_w']

        buf = base64.decodestring(item['features'].encode('utf8'))
        array = np.frombuffer(buf, dtype='float32')
        array = array.reshape((-1, 2048)).transpose()
        features[i, :, :array.shape[1]] = array

        buf = base64.decodestring(item['boxes'].encode('utf8'))
        array = np.frombuffer(buf, dtype='float32')
        array = array.reshape((-1, 4)).transpose()
        boxes[i, :, :array.shape[1]] = array

        buf = base64.decodestring(item['objects_id'].encode('utf8'))
        array = np.frombuffer(buf, dtype='int64')
        array = array.reshape((-1,36)).transpose()


        