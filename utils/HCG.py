import numpy as np
from torch import nn

# helper function to return external knowledge for each question

def HCG_generator(concept_vis, concept_lin, topology, relation, concept_property, property):
    HCG_pool = HCG(concept_vis, concept_lin, topology, relation, concept_property, property)
    return HCG


class HCG:
    def __init__(self, concept_vis, concept_lin, topology, relation, concept_property, property) :
        self.concept_vis = concept_vis
        self.concept_lin = concept_lin
        self.topology = topology
        self.relation = relation
        self.concept_property = concept_property
        self.property = property
        

    def __getitem__(self, key):
        target_HCG = {
            'concept_vis': self.concept_vis[key],
            'concept_lin': self.concept_lin[key],
            'topology': self.topology[key],
            'relation': self.relation[key],
            'concept_property': self.concept_property[key],
            'property': self.property[key],
        }
        return target_HCG