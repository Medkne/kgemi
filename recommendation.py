import torch
from utils import KGDataset, accuracy
import random 
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def read(path):
    """
    Reads the triples from a knowledge graph file.

    Input:
        path: Path to the file where the triples are saved.

    Returns:
        A list of triples.
    """
    triples = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        h, r, t = line.strip().split('\t')
        triples.append(tuple([h, r, t]))
    return triples


def model_pred(head, rel, tail, dataset, model):
    """
    Calculates the model score for a given triple.

    Input:
        head: Head entity of the triple.
        rel: Relation in the triple.
        tail: Tail entity of the triple.
        dataset: An instance of the Dataloader class (see utils.py).
        model: A trained model.

    Returns:
        A score assigned by the model to the triple.
    """
    h = dataset.ent2id[head]
    t = dataset.ent2id[tail]
    r = dataset.rel2id[rel] 
    score = model(torch.LongTensor([h]).to(model.device), torch.LongTensor([r]).to(model.device), torch.LongTensor([t]).to(model.device)).cpu().data.numpy()
    return score.item()


def recommender(testing, dataset, model, interaction):
    """
    Ranks triples involving a microbial strain based on model scores.

    Input:
        testing: A list of triples in which the target strain is either the sender or receiver.
        dataset: An instance of the Dataloader class (see utils.py).
        model: A trained model.

    Returns:
        A pandas DataFrame with strains, interaction type, and their normalized scores.
    """
    scores = np.array([model_pred(t[0], interaction, t[2], dataset, model) for t in testing])
    selected_strains = [t[0] for t in testing]
    interactions = [t[1].split('_')[1] for t in testing]

    df = pd.DataFrame({'Strain': selected_strains, 'Score': scores, 'Interaction': interactions})
    df = df.sort_values(by='Score', ascending=False)

    min_score = df['Score'].min()
    max_score = df['Score'].max()
    df['Normalized Score'] = (df['Score'] - min_score) / (max_score - min_score)    

    return df


def plot(df):
    """
    Generates a bar plot of the normalized scores for microbial strains,
    colored by interaction type.
    """
    min_score = df['Score'].min()
    max_score = df['Score'].max()
    df['Normalized Score'] = (df['Score'] - min_score) / (max_score - min_score)

    df_strains = pd.read_excel("datasets/Mapping.xlsx", index_col=0)
    strain_id = {}
    for _, raw in df_strains.iterrows():
        if '1' in raw['Strain shorthand']:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match']) + ' 1'
        elif '2' in raw['Strain shorthand']:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match']) + ' 2'
        elif '3' in raw['Strain shorthand']:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match']) + ' 3'
        else:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match'])

    df['Full Name'] = df['Strain'].map(strain_id)
    df['Score percent'] = df['Normalized Score'] * 100

    # Map colors based on interaction type
    def get_color(rel):
        if 'Positive' in rel:
            return '#2ca02c'
        elif 'Negative' in rel:
            return '#d62728'
        elif 'Neutral' in rel:
            return '#1f77b4'
        return 'gray'

    df['Color'] = df['Interaction'].map(get_color)

    # Plot manually using matplotlib to control individual colors
    plt.figure(figsize=(5, 6))
    plt.barh(df['Full Name'], df['Score percent'], color=df['Color'], edgecolor='black')
    plt.ylabel('Sender Strain', fontsize=16)
    plt.xlabel('Score (%)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for label in plt.gca().get_yticklabels():
        label.set_fontstyle('italic')
    plt.gca().invert_yaxis()  # Highest score at top
    plt.grid(visible=False)
    plt.tight_layout()
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()


def main():
    data = KGDataset('datasets/MKG')

    test_el = [('BI', 'Sucrose_Positive', 'EL'),
        ('PR2', 'Sucrose_Negative', 'EL'),
        ('KA', 'Sucrose_Positive', 'EL'),
        ('PAg1', 'Sucrose_Negative', 'EL'),
        ('PAr', 'Sucrose_Negative', 'EL'),
        ('EC', 'Sucrose_Positive', 'EL'),
        ('PH', 'Sucrose_Negative', 'EL'),
        ('RP1', 'Sucrose_Negative', 'EL'),
        ('PP', 'Sucrose_Negative', 'EL'),
        ('PK', 'Sucrose_Positive', 'EL')]

    model_el = torch.load("models/EL_rec_model.chkpnt")
    interaction_el = 'Sucrose_Negative'
    test_par =  [('PAl', 'Lactose_Positive', 'PAr'),
        ('CF', 'Lactose_Positive', 'PAr'),
        ('RP2', 'Lactose_Positive', 'PAr'),
        ('KA', 'Lactose_Positive', 'PAr'),
        ('PAg1', 'Lactose_Neutral', 'PAr'),
        ('SF1', 'Lactose_Neutral', 'PAr'),
        ('RP1', 'Lactose_Positive', 'PAr'),
        ('PAg2', 'Lactose_Neutral', 'PAr'),
        ('PK', 'Lactose_Neutral', 'PAr'),
        ('LA', 'Lactose_Neutral', 'PAr')]


    model_par = torch.load("models/PAr_recommendation_model.chkpnt")
    interaction_par = 'Lactose_Positive'
    model_par.eval()
    model_el.eval()
    df_el = recommender(test_el, data,  model_el, interaction_el)
    plot(df_el)

    df_par = recommender(test_par, data,  model_par, interaction_par)
    plot(df_par)


if __name__ == '__main__':
    main()