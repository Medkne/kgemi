import torch 
import numpy as np
import torch.nn.functional as F


class Losses:
    def __init__(self, args, model, num_batches):
        self.args = args
        self.model = model
        self.num_batches = num_batches
    

    def pair_loss(self, head, relation, tail, n_head, n_rel ,n_tail):
        pos_scores = self.model.forward(head, relation, tail)
        neg_scores = self.model.forward(n_head, n_rel, n_tail)
        return torch.sum(F.relu(self.args.margin + pos_scores - neg_scores)) + \
        (self.args.reg_lambda * self.model.l2_loss() / self.num_batches)
    
    
    def point_loss(self, head, relation, tail, label):
        scores = self.model.forward(head, relation, tail)
        return torch.sum(F.softplus(- label * scores)) \
        + (self.args.reg_lambda * self.model.l2_loss() / self.num_batches)
