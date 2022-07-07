import numpy as np
import torch
from torch import nn

class Concepts:
    def __init__(self, concepts):
        self.concepts=[]
        kwargs={}
        for i in range(len(concepts)):
            kwargs["Cid"] = concepts['Cids'][i]
            kwargs["v_features"] = concepts['Cids'][i]
            kwargs["l_features"] = concepts['Cids'][i]
            kwargs["PV"] = concepts['PVs'][i]
            kwargs["RM"] = concepts['RMs'][i]
            kwargs["type"] = concepts['types'][i]
            concept = Concept(concepts[i])
            self.concepts.append(concept)

    def out(self):
        return self.concepts


class Concept:
    def __init__(self, **kwargs):
        self.CId = kwargs['CId']
        self.v_features = kwargs['v_features']
        self.l_features = kwargs['l_features']
        self.PV = kwargs['PV']
        self.RM = kwargs['RM']
        self.type = kwargs['type']

    def __str__(self):
        return self.CId