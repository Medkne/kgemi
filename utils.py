import pandas as pd
import numpy as np
from random import shuffle
import copy
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import torch
import os
import shutil
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder


def Triples(path):
    """
    Transforms a tabular dataset into a knowledge graph.

    Input:
        path: Directory where the dataset is saved.

    Returns:
        A list of triples in the form (h, r, t), where h and t are microbial strains, 
        and r represents the carbon source environment and the interaction type.
    """

    df = pd.read_csv(path, index_col= 0)
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

def all_species(Triples):
    """
    Extracts all unique microbial strains from the knowledge graph.

    Input:
        Triples: A list of triples representing the knowledge graph.

    Returns:
        A list of all microbial strains present in the knowledge graph.
    """
    all = set()
    for h, r, t in Triples:
        all.add(h)
        all.add(t)
    return list(all)

def count_interactions(triples):
    """
    Counts the number of interactions for each microbial strain.

    Input:
        triples: A list of triples representing the knowledge graph.

    Returns:
        A dictionary where the keys are microbial strains and the values are their corresponding interaction counts.
    """
    all_S = all_species(triples)
    d = {}
    for species in all_S:
        count = 0
        for triple in triples:
            if species in triple:
                count += 1
        d[species] = count
    return d


def Random_split(triples, train_frac = 0.9):
    """
    Splits the knowledge graph into training, validation, and test sets.

    Input:
        triples: A list of triples representing the knowledge graph.
        train_frac: Proportion of the data to be used for training (must be <= 0.95).

    Returns:
        Three lists: train, valid, and test splits.
    """

    if train_frac > 0.95:
        raise ValueError("train_frac should be less than 95%")
    
    Triples = copy.deepcopy(triples)
    shuffle(Triples)
    n_train = int(train_frac * len(Triples))
    train = Triples[:n_train]
    valid_test = Triples[n_train:]
    n_valid_test = int(len(valid_test)/2)
    valid = valid_test[:n_valid_test]
    test = valid_test[n_valid_test:]
    return train, valid, test


def save_triples_to_file(triples, file_path):
    """
    Saves triples to a text file.

    Input:
        triples: A list of triples.
        file_path: Destination file path.
    """
    with open(file_path, 'w') as file:
        for head, relation, tail in triples:
            file.write(f'{head}\t{relation}\t{tail}\n')



def ent_rel2ids(ds_name):
    """
    Maps each entity and relation in the knowledge graph to a unique ID.

    Input:
        ds_name: Name of the dataset.

    Returns:
        None. Saves 'ent2ids.txt' and 'rel2ids.txt' in the dataset directory.
    """
    path = "datasets/" + ds_name + "/train.txt"
    entities = {}
    relations = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            triple = line.strip().split("\t")
            head = triple[0]
            rel = triple[1]
            tail = triple[2]
            if head not in entities:
                entities[head] = len(entities)
            if tail not in entities:
                entities[tail] = len(entities)
            if rel not in relations:
                relations[rel] = len(relations)
    
    with open("datasets/" + ds_name + "/ent2ids.txt", "w") as f:
        for k, v in entities.items():
            f.write(f"{k}\t{v}\n")

    with open("datasets/" + ds_name + "/rel2ids.txt", "w") as f:
        for k, v in relations.items():
            f.write(f"{k}\t{v}\n")



def create_data(path, data_name):
    """
    Transforms a raw dataset into a knowledge graph format.
    Splits the data into train/valid/test sets and saves them, along with ID mappings.

    Input:
        path: Path to the original dataset.
        data_name: Name of the new directory to save the processed data.

    Returns:
        None.
    """

    triples = Triples(path)
    train, valid, test = Random_split(triples, train_frac=0.9)

    # Create a directory
    dir_path = 'datasets/' + data_name + '/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # save
    save_triples_to_file(train, dir_path + 'train.txt')
    save_triples_to_file(test, dir_path + 'test.txt')
    save_triples_to_file(valid, dir_path + 'valid.txt')
    # Ids
    ent_rel2ids(data_name)



class Dataloader:
    def __init__(self, dir):
        self.dir = dir
        self.ent2id = self.readIds('entities')
        self.rel2id = self.readIds('relations')
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

    def load(self, train_valid_orTest):
        file_path = self.dir + "/" + train_valid_orTest + ".txt"
        triples = self.read(file_path)
        return triples

    def read(self, file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        triples = np.zeros((len(lines), 3), dtype=int)
        for i, line in enumerate(lines):
            triples[i] = np.array(self.triple2ids(line.strip().split("\t")))
        return triples
    
    def triple2ids(self, triple):
        return [self.get_ent_id(triple[0]), self.get_rel_id(triple[1]), self.get_ent_id(triple[2])]
                     
    def get_ent_id(self, ent):
        return self.ent2id[ent]
    
    def get_rel_id(self, rel):
        return self.rel2id[rel]

    def num_ent(self):
        return len(self.ent2id)
    
    def num_rel(self):
        return len(self.rel2id)

    
    def readIds(self, ent_or_rel):
        entities = {}
        relations = {}
        with open(self.dir + "/ent2ids.txt") as f:
            lines = f.readlines()
            for line in lines:
                ent = line.strip().split("\t")
                entities[ent[0]] = int(ent[1])
        with open(self.dir + "/rel2ids.txt") as f:
            lines = f.readlines()
            for line in lines:
                rel = line.strip().split("\t")
                relations[rel[0]] = int(rel[1])
        if ent_or_rel == 'entities':
            return entities
        else:
            return relations



def Predict_interaction(ent2id, rel2id, model, fact):
    """
    Predicts the interaction type for a given fact/triple.

    Input:
        ent2id: Dictionary mapping entities to IDs.
        rel2id: Dictionary mapping relations to IDs.
        model: Trained model for scoring triples.
        fact: A triple in the form (head, environment, tail).

    Returns:
        The predicted interaction type: 'Positive', 'Neutral', or 'Negative'.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    head_name, environment, tail_name = fact
    head_id = ent2id[head_name]
    tail_id = ent2id[tail_name]
    possible_relations = [f"{environment}_Positive", f"{environment}_Neutral", f"{environment}_Negative"]
    scores = []
    for possible_rel in possible_relations:
        if possible_rel in rel2id:
            rel_id = rel2id[possible_rel]
            score = model(torch.LongTensor([head_id]).to(device), torch.LongTensor([rel_id]).to(device), torch.LongTensor([tail_id]).to(device)).cpu().data.numpy().item()
        else:
                score = -9999
        scores.append(score)
    max_index = np.argmax(np.array(scores))
    return possible_relations[max_index].split('_')[1]




def accuracy(test_set, ent2id, rel2id, model):
    """
    Calculates the accuracy and F1-score for the model on a test set.

    Input:
        test_set: A list of test triples (as ID tuples).
        ent2id: Dictionary mapping entities to IDs.
        rel2id: Dictionary mapping relations to IDs.
        model: Trained model for prediction.

    Returns:
        acc: Overall accuracy.
        f1: F1-scores for each interaction type.
        class_labels: List of interaction types corresponding to the F1-scores.
    """

    true = []
    predicted = []
    id2ent = {v: k for k, v in ent2id.items()}
    id2rel = {v: k for k, v in rel2id.items()}
    for triple in test_set:
        head_id, rel_id, tail_id = triple
        head, rel, tail = id2ent[head_id], id2rel[rel_id] ,id2ent[tail_id] 
        environment, interaction = rel.split('_')
        pred = Predict_interaction(ent2id, rel2id, model, (head, environment, tail))
        true.append(interaction)
        predicted.append(pred)
    label_encoder = LabelEncoder()
    label_encoder.fit(true + predicted)
    
    true_encoded = label_encoder.transform(true)
    predicted_encoded = label_encoder.transform(predicted)
    
    acc = accuracy_score(true_encoded, predicted_encoded)
    f1 = f1_score(true_encoded, predicted_encoded, average=None)
    
    class_labels = label_encoder.classes_  


    acc = accuracy_score(true, predicted)
    f1 = f1_score(true, predicted, average=None)
    return acc, f1, class_labels
