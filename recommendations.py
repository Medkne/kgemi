import torch
from utils import Dataloader, accuracy
import random 
from trainer import Trainer
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
        interaction: The interaction type (relation) used for scoring.

    Returns:
        A pandas DataFrame with strains and their normalized scores, sorted in descending order.
    """
    scores = np.array([model_pred(t[0], interaction, t[2], dataset, model) for t in testing])
    selected_strains = [t[0] for t in testing]
    df = pd.DataFrame({'Strain':selected_strains, 'Score':scores}).sort_values(by = 'Score', ascending=False)

    min_score = df['Score'].min()
    max_score = df['Score'].max()
    df['Normalized Score'] = (df['Score'] - min_score) / (max_score - min_score)    

    return df

def plot(df):
    """
    Generates a bar plot of the normalized scores for microbial strains.

    Input:
        df: A DataFrame containing 'Strain', 'Score', and optionally 'Full Name'.

    Side Effects:
        Displays a bar plot.
    """
    min_score = df['Score'].min()
    max_score = df['Score'].max()
    df['Normalized Score'] = (df['Score'] - min_score) / (max_score - min_score)
    import pandas as pd
    df_strains = pd.read_excel("datasets/Mapping.xlsx", index_col= 0)
    strain_id = {}
    for _, raw in df_strains.iterrows():
        if '1' in raw['Strain shorthand']:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match']) +' 1'
        elif '2' in raw['Strain shorthand']:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match']) +' 2'
        elif '3' in raw['Strain shorthand']:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match']) +' 3'

        else:
            strain_id[raw['Strain shorthand']] = str(raw['Closest match'])
    df['Full Name'] = df['Strain'].map(strain_id)
    df['Score percent'] = df['Normalized Score'] * 100
    sns.set_theme(context='notebook', style='white')
    plt.figure(figsize=(5, 6))
    bar_plot = sns.barplot(y='Full Name', x='Score percent', data=df, color='#2A9D8F', edgecolor = 'black')
    plt.ylabel('Sender Strain', fontsize = 16)
    plt.xlabel('Score (%)', fontsize = 16)
    ax = plt.gca()  
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.grid(visible=False)
    plt.show()

def main():
    data = Dataloader('datasets/original_without')

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

    model_el = torch.load("EL_recommendation_model.chkpnt")
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


    model_par = torch.load("PAr_recommendation_model.chkpnt")
    interaction_par = 'Lactose_Positive'
    model_par.eval()
    model_el.eval()
    df_el = recommender(test_el, data,  model_el, interaction_el)
    plot(df_el)

    df_par = recommender(test_par, data,  model_par, interaction_par)
    plot(df_par)


if __name__ == '__main__':
    main()