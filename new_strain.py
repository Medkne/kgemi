from trainer import Trainer
from utils import Dataloader
import argparse
import torch
import time
from utils import accuracy
from sklearn.model_selection import KFold
import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd
from collections import Counter


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', default=1000, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-reg_lambda', default=0.03, type=float, help="l2 regularization parameter")
    parser.add_argument('-dataset', default="WN18", type=str, help="wordnet dataset")
    parser.add_argument('-emb_dim', default=10, type=int, help="embedding dimension")
    parser.add_argument('-neg_ratio', default=1, type=int, help="number of negative examples per positive example")
    parser.add_argument('-batch_size', default=1415, type=int, help="batch size")
    parser.add_argument('-save_each', default=5, type=int, help="validate every k epochs")
    parser.add_argument('-model', default='SimplE', type=str, help="Model = [TransE, ComplEx, SImplE, DistMult]")
    parser.add_argument('-margin', default=1., type=float, help="margin for pairwise loss")
    parser.add_argument('-loss', default= 'Pointwise', type=str, help="Loss = [Pairwise (default), Pointwise]")
    parser.add_argument('-related', default= 5, type=int, help="Number of related strains")

    args = parser.parse_args()
    return args

def read(path):
    """
    Reads triples from a text file.

    Input:
        path: Path to the file containing triples.

    Returns:
        A list of triples as (head, relation, tail).
    """
    triples = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        h, r, t = line.strip().split('\t')
        triples.append(tuple([h, r, t]))
    return triples



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


def Triples(data):
    """
    Converts a DataFrame of microbial interactions into a list of knowledge graph triples.

    Input:
        data: A pandas DataFrame containing microbial interaction data.

    Returns:
        A list of triples in the format (head, relation, tail), where 'relation' encodes the carbon source and interaction type.
    """
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
    """
    Maps a list of triples to their corresponding integer ID representations.

    Inputs:
        ds: List of triples (h, r, t).
        entities: Dictionary mapping entities to their IDs.
        relations: Dictionary mapping relations to their IDs.

    Returns:
        A NumPy array of triples represented by their respective IDs.
    """
    triples = np.zeros((len(ds), 3), dtype=int)
    for i, t in enumerate(ds):
        triples[i] = np.array([entities[t[0]], relations[t[1]], entities[t[2]]])
    return triples


if __name__ == '__main__':
    args = get_parameter()
    path = "/home/mkhatban/coding/Microbial-interaction-prediction/Data/Features.csv"
    df = pd.read_csv(path, index_col= 0)
    data = df[df['Carbon'] != 'Water'].copy()
    df = data[['Bug 1', 'phy_strain_component_0_x', 'phy_strain_component_1_x']].copy()
    df_unique = df.drop_duplicates()
    strains = df_unique['Bug 1'].values
    all_accuracies = {}
    f1_sc = {}
    dl = Dataloader("datasets/original_without")
    ent2id = dl.ent2id
    rel2id = dl.rel2id
    num_ent = dl.num_ent()
    num_rel = dl.num_rel()

    for strain in strains:
        train_df = data.loc[(data['Bug 1'] != strain) & (data['Bug 2'] != strain),:].copy()
        test_df = data.loc[(data['Bug 1'] == strain) | (data['Bug 2'] == strain),:].copy()
        related_strains, distances = find_closest_strain_with_distances(df_unique, strain, n = args.related)        
        print(f"Related strains for {strain} (n={args.related}): {related_strains}")
        #train/test triples
        train = triple2id(Triples(train_df), ent2id, rel2id)
        test = triple2id(Triples(test_df), ent2id, rel2id)
        print(f"~~~~ strain :{strain} ~~~~")

        trainer = Trainer(train, num_ent, num_rel, ent2id, rel2id ,args)
        trainer.train()
        model = trainer.model
        #eval
        model.eval()
        strain_embs = aggregate_embs_weighted(related_strains, distances, ent2id, model)
        with torch.no_grad():
            model.ent_h_embs.weight[ent2id[strain]] = strain_embs[0]
            model.ent_t_embs.weight[ent2id[strain]] = strain_embs[1]
        acc, f1, class_labels = accuracy(test, ent2id, rel2id, model)
        all_accuracies[strain] = acc
        f1_sc[strain] = f1
        print('accuracy = ', acc)
    print("==========================")
    print('Accuracy: ', np.mean(list(all_accuracies.values())))
#    print("mean : ", np.mean(all_accuracies))
#    print("std : ", np.std(all_accuracies))
    print('all accuracies : ', all_accuracies)
    print('F1-score : ', f1_sc)
    print("===========")
    print(class_labels)