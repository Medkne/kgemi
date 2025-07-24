import argparse
import torch
import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import euclidean
from utils import KGDataset, accuracy
from main import KnowledgeGraphTrainer


class NumpyTripleDataset(torch.utils.data.Dataset):
    def __init__(self, triples_np, ent2id, rel2id):
        self.triples = np.array(triples_np, dtype=np.int64)
        self.ent2id = ent2id
        self.rel2id = rel2id

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return tuple(torch.tensor(self.triples[idx], dtype=torch.long))

    def num_ent(self):
        return len(self.ent2id)

    def num_rel(self):
        return len(self.rel2id)


class Trainer(KnowledgeGraphTrainer):
    def train(self):
        for epoch in range(1, self.args.ne + 1):
            self.model.train()
            total_loss = 0

            for batch in self.train_loader:
                heads, rels, tails = [x.to(self.device) for x in batch]

                # Negative sampling
                neg_rels = torch.tensor(
                    [self.rand_rel_except(r.item()) for r in rels],
                    dtype=torch.long
                ).to(self.device)

                self.optimizer.zero_grad()
                if self.args.loss == 'Pairwise':
                    loss = self.loss_fn.pair_loss(heads, rels, tails, heads, neg_rels, tails)
                else:
                    loss_pos = self.loss_fn.point_loss(heads, rels, tails, 1)
                    loss_neg = self.loss_fn.point_loss(heads, neg_rels, tails, -1)
                    loss = loss_pos + loss_neg

                # Regularization
                loss += self.args.reg_lambda * self.model.l2_loss() / len(self.train_loader)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch} - Loss: {total_loss:.4f}")
            if epoch % self.args.save_each == 0:
                self.save_model(epoch)




def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=50, type=int)
    parser.add_argument('-lr', default=0.1, type=float)
    parser.add_argument('-reg_lambda', default=0.001, type=float)
    parser.add_argument('-dataset', default="MKG", type=str)
    parser.add_argument('-emb_dim', default=500, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-save_each', default=50, type=int)
    parser.add_argument('-model', default='SimplE', type=str)
    parser.add_argument('-margin', default=1., type=float)
    parser.add_argument('-loss', default='Pointwise', type=str)
    parser.add_argument('-related', default=5, type=int)
    return parser.parse_args()


def find_closest_strain_with_distances(df, target_name, n=5):
    """
    Finds the n closest microbial strains to a given target based on phylogenetic distance.

    Inputs:
        df: A DataFrame containing phylogenetic feature vectors for strains.
        target_name: Name of the target strain.
        n: Number of closest strains to return.

    Returns:
        closest_names: n closest strain names.
        closest_distances: Corresponding distances to the target strain.
    """
    features = ['phy_strain_component_0_x', 'phy_strain_component_1_x']    
    target_row = df[df['Bug 1'] == target_name]
    other_rows = df[df['Bug 1'] != target_name]
    
    target_features = target_row[features].values[0]
    other_features = other_rows[features].values
    
    distances = [euclidean(target_features, row) for row in other_features]
    closest_indices = np.argsort(distances)[:n]
    
    closest_names = other_rows.iloc[closest_indices]['Bug 1'].values
    closest_distances = np.array(distances)[closest_indices]
    
    return closest_names, closest_distances

def aggregate_embs_weighted(strains, distances, ent2id, model):
    """
    Generates a synthetic embedding for a target strain by aggregating embeddings of phylogenetically related strains.

    Inputs:
        strains: List of closest strains.
        distances: List of distances between the target strain and each related strain.
        ent2id: Dictionary mapping strain names to entity IDs.
        model: The trained embedding model.

    Returns:
        h_emb_weighted: Aggregated head embedding.
        t_emb_weighted: Aggregated tail embedding.
    """
    ids = [ent2id[s] for s in strains]
    h_emb = model.ent_h_embs(torch.tensor(ids, device=model.device)).detach()
    t_emb = model.ent_t_embs(torch.tensor(ids, device=model.device)).detach()
    
    weights = 1 / (distances + 1)
    weights = weights / weights.sum()  # normalize weights
    weights = torch.tensor(weights, device=model.device).float()
    
    # Apply weights
    h_emb_weighted = torch.sum(h_emb * weights.unsqueeze(1), dim=0)
    t_emb_weighted = torch.sum(t_emb * weights.unsqueeze(1), dim=0)
    
    return h_emb_weighted, t_emb_weighted

def Triples(df):
    """
    Transforms a tabular dataset into a knowledge graph.

    Input:
        path: Directory where the dataset is saved.

    Returns:
        A list of triples in the form (h, r, t), where h and t are microbial strains, 
        and r represents the carbon source environment and the interaction type.
    """
    data = df[df['Carbon'] != 'Water'].copy()
    all_triples = []
    data['Class'] = "Negative"
    data.loc[data['2 on 1: Effect'] > 0, 'Class'] = "Positive"
    data.loc[data['2 on 1: Effect'] == 0, 'Class'] = "Neutral"

    for _, row in data.iterrows():
        head = row['Bug 2']
        relation = row['Carbon'] + "_" + row['Class']
        tail = row['Bug 1']
        triple = (head, relation, tail)
        all_triples.append(triple)
    return all_triples


def triple2id(ds, entities, relations):
    triples = np.zeros((len(ds), 3), dtype=int)
    for i, t in enumerate(ds):
        triples[i] = np.array([entities[t[0]], relations[t[1]], entities[t[2]]])
    return triples


if __name__ == '__main__':
    args = get_parameter()
    path = "/home/mkhatban/coding/Microbial-interaction-prediction/Data/Features.csv"
    df = pd.read_csv(path, index_col=0)
    data = df[df['Carbon'] != 'Water'].copy()

    # For similarity
    df_phylo = data[['Bug 1', 'phy_strain_component_0_x', 'phy_strain_component_1_x']].drop_duplicates()
    strains = df_phylo['Bug 1'].values

    # Load global entity/relation mapping
    mapping_dataset = KGDataset(os.path.join("datasets", args.dataset))
    ent2id = mapping_dataset.ent2id
    rel2id = mapping_dataset.rel2id

    all_accuracies = {}
    f1_scores = {}

    for strain in strains:
        print(f"\n====== Training excluding strain: {strain} ======")
        # Train/test split based on target strain
        train_df = data[(data['Bug 1'] != strain) & (data['Bug 2'] != strain)].copy()
        test_df = data[(data['Bug 1'] == strain) | (data['Bug 2'] == strain)].copy()

        # Convert to triples
        train_triples = triple2id(Triples(train_df), ent2id, rel2id)
        test_triples = triple2id(Triples(test_df), ent2id, rel2id)

        train_dataset = NumpyTripleDataset(train_triples, ent2id, rel2id)
        trainer = Trainer(train_dataset, train_dataset, train_dataset, args)
        trainer.train()
        model = trainer.model

        # Aggregate embedding for unseen strain
        related_strains, distances = find_closest_strain_with_distances(df_phylo, strain, n=args.related)
        print(f"Closest strains to {strain}: {related_strains}")

        with torch.no_grad():
            strain_emb_h, strain_emb_t = aggregate_embs_weighted(related_strains, distances, ent2id, model)
            model.ent_h_embs.weight[ent2id[strain]] = strain_emb_h
            model.ent_t_embs.weight[ent2id[strain]] = strain_emb_t

        model.eval()
        acc, f1, class_labels = accuracy(test_triples, ent2id, rel2id, model)
        print(f"Accuracy = {acc:.4f}")
        print(f"F1-score = {[round(f, 2) for f in f1]}")

        all_accuracies[strain] = acc
        f1_scores[strain] = f1

    print("\n================ Summary =================")
    print(f"Mean Accuracy: {np.mean(list(all_accuracies.values())):.4f}")
    print("All Accuracies:", all_accuracies)
    print("F1 Scores per Strain:", f1_scores)
    print("Class Labels:", class_labels)


