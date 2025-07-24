import argparse
import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from model import SimplE, DistMult, TransE
from utils import KGDataset, accuracy
from torch.utils.data import DataLoader


class Losses:
    def __init__(self, args, model):
        self.args = args
        self.model = model    

    def pair_loss(self, head, relation, tail, n_head, n_rel ,n_tail):
        pos_scores = self.model.forward(head, relation, tail)
        neg_scores = self.model.forward(n_head, n_rel, n_tail)
        return torch.sum(F.relu(self.args.margin + pos_scores - neg_scores))
    
    
    def point_loss(self, head, relation, tail, label):
        scores = self.model.forward(head, relation, tail)
        return torch.sum(F.softplus(- label * scores))


class KnowledgeGraphTrainer:
    def __init__(self, train_dataset, valid_dataset, test_dataset, args):
        self.args = args
        self.set_seed(42)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)

        # Entities and Relations
        self.ent2id = self.train_dataset.ent2id
        self.rel2id = self.train_dataset.rel2id
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        # Model and training components
        self.model = self.load_model()
        self.loss_fn = Losses(self.args, self.model)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.args.lr, initial_accumulator_value=0.1)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def load_model(self):
        num_ent = self.train_dataset.num_ent()
        num_rel = self.train_dataset.num_rel()
        model_name = self.args.model
        if model_name == "SimplE":
            return SimplE(num_ent, num_rel, self.args.emb_dim, self.device).to(self.device)
        elif model_name == "DistMult":
            return DistMult(num_ent, num_rel, self.args.emb_dim, self.device).to(self.device)
        elif model_name == "TransE":
            return TransE(num_ent, num_rel, self.args.emb_dim, self.device).to(self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def rand_rel_except(self, rel_id):
        relation_name = self.id2rel[rel_id]
        environment, interaction_type = relation_name.split('_')
        candidates = [f"{environment}_Positive", f"{environment}_Neutral", f"{environment}_Negative"]
        candidates = [r for r in candidates if r in self.rel2id and r != relation_name]
        if len(candidates) > 1:
            random.shuffle(candidates)
        return self.rel2id[candidates[0]]
    

    def train(self):
        print("~~~~ Training ~~~~")
        for epoch in range(1, self.args.ne + 1):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                heads, rels, tails = [x.to(self.device) for x in batch]

                # Negative relation sampling
                neg_rels = torch.tensor(
                    [self.rand_rel_except(r.item()) for r in rels],
                    dtype=torch.long
                ).to(self.device)

                self.optimizer.zero_grad()
                if self.args.loss == 'Pairwise':
                    loss = self.loss_fn.pair_loss(heads, rels, tails, heads, neg_rels, tails) + self.args.reg_lambda * self.model.l2_loss() / len(self.train_loader)
                else:
                    loss_pos = self.loss_fn.point_loss(heads, rels, tails, 1)
                    loss_neg = self.loss_fn.point_loss(heads, neg_rels, tails, -1)
                    loss = loss_pos + loss_neg + self.args.reg_lambda * self.model.l2_loss() / len(self.train_loader)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch} - Loss: {total_loss:.4f}")
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

        self.evaluate()

    def save_model(self, epoch):
        print(f"Saving model at epoch {epoch}")
        directory = os.path.join("models", self.args.dataset)
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model, os.path.join(directory, f"{epoch}.chkpnt"))

    def evaluate(self):
        print("~~~~ Selecting Best Epoch on Validation ~~~~")
        best_acc = 0.0
        best_epoch = "0"
        epochs2test = [str(self.args.save_each * (i+1)) for i in range(self.args.ne // self.args.save_each)]

        for epoch in epochs2test:
            path = os.path.join("models", self.args.dataset, f"{epoch}.chkpnt")
            if not os.path.exists(path):
                continue
            model = torch.load(path, map_location=self.device)
            model.eval()
            acc, _, _ = accuracy(self.valid_dataset.triples, self.ent2id, self.rel2id, model)
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch

        print(f"Best epoch: {best_epoch}")
        print("~~~~ Testing Best Model ~~~~")
        best_model_path = os.path.join("models", self.args.dataset, f"{best_epoch}.chkpnt")
        best_model = torch.load(best_model_path, map_location=self.device)
        best_model.eval()

        acc, f1, classes = accuracy(self.test_dataset.triples, self.ent2id, self.rel2id, best_model)
        print(f"Accuracy: {round(acc, 2)}")
        print(f"F1-scores: {[round(f, 2) for f in f1]}")
        print("Classes:", classes)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1000, type=int)
    parser.add_argument('-lr', default=0.1, type=float)
    parser.add_argument('-reg_lambda', default=0.001, type=float)
    parser.add_argument('-dataset', default="MKG", type=str)
    parser.add_argument('-emb_dim', default=500, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-save_each', default=50, type=int)
    parser.add_argument('-model', default='SimplE', type=str)
    parser.add_argument('-margin', default=1., type=float)
    parser.add_argument('-loss', default='Pointwise', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Load datasets outside the trainer
    data_dir = os.path.join("datasets", args.dataset)
    train_dataset = KGDataset(data_dir, "train")
    valid_dataset = KGDataset(data_dir, "valid")
    test_dataset = KGDataset(data_dir, "test")

    # Pass everything to the trainer
    trainer = KnowledgeGraphTrainer(train_dataset, valid_dataset, test_dataset, args)
    trainer.train()
