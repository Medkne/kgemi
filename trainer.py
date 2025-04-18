from model import SimplE, DistMult, TransE
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import os 
from Losses import * 

class Trainer:
    def __init__(self, train_data, num_ent, num_rel, ent2id, rel2id ,args):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if args.model == "SimplE":
            self.model = SimplE(num_ent, num_rel, args.emb_dim, self.device)
        elif args.model == "DistMult":
            self.model = DistMult(num_ent, num_rel, args.emb_dim, self.device)
        elif args.model == "TransE":
            self.model = TransE(num_ent, num_rel, args.emb_dim, self.device)
        else:
            raise ValueError("Not implemented !")
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.train_data = train_data
        self.args = args
        self.num_batches = self.num_batch()
        self.Loss = Losses(self.args, self.model, self.num_batches)
        self.batch_index = 0

    def num_batch(self):
        return int(math.ceil(float(len(self.train_data)) / self.args.batch_size))

    def train(self):
        self.model.train()

        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0,
            initial_accumulator_value=0.1  
        )

        for epoch in range(1, self.args.ne + 1):
            np.random.shuffle(self.train_data)
            last_batch = False
            total_loss = 0.0

            while not last_batch:
                pos_heads, pos_rels, pos_tails, neg_rels_inter = self.next_batch(self.args.batch_size, self.device)

                last_batch = self.was_last_batch()
                optimizer.zero_grad()

                if self.args.loss == 'Pairwise':
                    loss = self.Loss.pair_loss(pos_heads, pos_rels, pos_tails, pos_heads, neg_rels_inter ,pos_tails)
                else:
                    loss_pos =  self.Loss.point_loss(pos_heads, pos_rels, pos_tails, 1)
                    loss_neg_inter =  self.Loss.point_loss(pos_heads, neg_rels_inter, pos_tails, -1)
                    loss = loss_pos + loss_neg_inter
 
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print("Loss in iteration " + str(epoch) + ": " + str(total_loss))
        
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

    def was_last_batch(self):
        return (self.batch_index == 0)


    def next_pos_batch(self, batch_size):
        if self.batch_index + batch_size < len(self.train_data):
            batch = self.train_data[self.batch_index: self.batch_index+batch_size]
            self.batch_index += batch_size
        else:
            batch = self.train_data[self.batch_index:]
            self.batch_index = 0
        return np.append(batch, np.ones((len(batch), 1)), axis=1).astype("int") 


    def generate_neg(self, pos_batch):
        neg_batch_inter = np.ones(len(pos_batch))
        for i in range(len(pos_batch)):
            rel = pos_batch[i][1]
            false_interactions = self.rand_rel_except(rel)
            neg_batch_inter[i] =  false_interactions[0] 
        return neg_batch_inter
    
    def next_batch(self, batch_size, device):
        pos_batch = self.next_pos_batch(batch_size)
        neg_batch_inter = self.generate_neg(pos_batch)
        # Positives interactions
        pos_heads = torch.tensor(pos_batch[:, 0]).long().to(device)
        pos_rels = torch.tensor(pos_batch[:, 1]).long().to(device)
        pos_tails = torch.tensor(pos_batch[:, 2]).long().to(device)
        # Negative Interaction 
        neg_rels_inter = torch.tensor(neg_batch_inter).long().to(device)
        return pos_heads, pos_rels, pos_tails, neg_rels_inter
    


    def rand_rel_except(self, rel):
        relation_name = self.id2rel[rel]
        environment, actual_interaction_type = relation_name.split('_')
        possible_relations = [f"{environment}_Positive", f"{environment}_Neutral", f"{environment}_Negative"]
        filter = [r for r in possible_relations if r != relation_name and r in self.rel2id.keys()]
        if len(filter) == 2:
            random.shuffle(filter)
            return self.rel2id[filter[0]], self.rel2id[filter[1]]
        else:
            return [self.rel2id[filter[0]]]

    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + 'MKG' + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + str(chkpnt) + ".chkpnt")