import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from utils import KGDataset
import random
import numpy as np




def read(path):
    """
    Reads the triples from a knowledge graph file.

    Input:
        path: Path to the file containing triples.

    Returns:
        A list of triples in the format (head, relation, tail).
    """
    triples = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        h, r, t = line.strip().split('\t')
        triples.append(tuple([h, r, t]))
    return triples

def get_rel_emb(rel2id, model):
    """
    Retrieves and concatenates the direct and inverse embeddings for all relations.

    Input:
        rel2id: Dictionary mapping relations to their IDs.
        model: Trained model.

    Returns:
        A pandas DataFrame containing concatenated embeddings for each relation.
    """
    rel_dic = {}
    rel_inv_dict = {}
    relations = rel2id.keys()
    for rel in relations:
        id = rel2id[rel]
        id = torch.tensor([id], device = model.device)
        rel_emb = model.rel_embs(id).detach().cpu().numpy().flatten()
        rel_emb_inv = model.rel_inv_embs(id).detach().cpu().numpy().flatten()
        rel_dic[rel] = rel_emb
        rel_inv_dict[rel] = rel_emb_inv
    rel_df = pd.DataFrame.from_dict(rel_dic, orient='index')
    rel_df_inv = pd.DataFrame.from_dict(rel_inv_dict, orient='index')

    rel_df.columns = [f'rel_{i+1}' for i in range(rel_df.shape[1])]
    rel_df_inv.columns = [f'rel_inv_{i+1}' for i in range(rel_df_inv.shape[1])]
    rel_concat = pd.concat([rel_df, rel_df_inv], axis = 1)
    return rel_concat



def filter_merge(df):
    """
    Filters environments to include only those with all three interaction types (Positive, Negative, Neutral),
    and concatenates their respective relation embeddings.

    Input:
        df: DataFrame of relation embeddings.

    Returns:
        A DataFrame representing each environment as a vector concatenation of the three interaction types.
    """
    df = df.reset_index()
    df[['base_env', 'interaction_type']] = pd.DataFrame(df['index'].apply(lambda x:x.split('_')).tolist(), index=df.index)

    valid_envs = df.groupby('base_env')['interaction_type'].nunique()
    valid_envs = valid_envs[valid_envs == 3].index  # Keep only environments with exactly 3 interaction types
    filtered_df = df[df['base_env'].isin(valid_envs)]

    negative = filtered_df[filtered_df['interaction_type'] == 'Negative']
    positive = filtered_df[filtered_df['interaction_type'] == 'Positive']
    neutral = filtered_df[filtered_df['interaction_type'] == 'Neutral']
    merged_df = pd.merge(negative, positive, on='base_env', suffixes=('_negative', '_positive'))
    merged_df = pd.merge(merged_df, neutral, on='base_env')
    merged_df.set_index('base_env', inplace=True)
    merged_df = merged_df.select_dtypes(include=[float, int])
    return merged_df


def compute_pairwise_distances(df):
    """
    Computes cosine distances between environment embedding vectors.

    Input:
        df: DataFrame containing environment vectors.

    Returns:
        A pairwise cosine distance matrix as a pandas DataFrame.
    """
    distances = pdist(df.values, metric='cosine')
    distance_matrix = squareform(distances)
    distance_df = pd.DataFrame(distance_matrix, index=df.index, columns=df.index)
    return distance_df

def create_sorted_distance_dict(distance_df):
    """
    Create a distance dictionary where each environment is associated with a list of environments sorted in a descending order 
    based on the similarity.
    Input:
        distance_df: pairwise distance matrix
    Returns:
        dictionary of environments and their corresponding sorted list.
    """
    sorted_distance_dict = {}
    for env in distance_df.index:
        sorted_distances = distance_df[env].sort_values()
        sorted_envs = sorted_distances.index[sorted_distances.index != env].tolist()
        sorted_distance_dict[env] = sorted_envs
    return sorted_distance_dict


def plot_pairwise_distance_heatmap(distance_df):
    """
    Visualizes the pairwise cosine distance matrix using a heatmap.

    Input:
        distance_df: Pairwise distance matrix.
    """
    plt.figure(figsize=(25, 10))  # Adjust the figure size as needed
    sns.heatmap(distance_df, cmap="viridis", annot=True, fmt=".2f", cbar=True)
    plt.title("Pairwise Distance")
    plt.xlabel("Environment")
    plt.ylabel("Environment")
    plt.show()

def predict_interaction_type(test_set, d, environments, distance_df):
    """
    Predicts the interaction type for each test triple by identifying the closest environment
    (with a known interaction for the same strain pair) in the training/validation set.

    Input:
        test_set: List of test triples.
        d: Dictionary mapping (head, environment, tail) to interaction type (from training/validation).
        environments: Dictionary of each environment and its closest environments.
        distance_df: Distance matrix between environments.

    Returns:
        true: List of actual interaction types.
        pred: List of predicted interaction types.
        distances: List of distances between the original and substituted environment.
    """
    pred = []
    true = []
    distances = []
    for triple in test_set:
        head, rel, tail = triple
        env, interaction = rel.split('_')
        true.append(interaction)
        closest_envs = environments[env] # Default to the same env if closest not found
        t = (head, closest_envs[0], tail)
        i = 0
        while t not in d:
            i = i +1
            t = (head, closest_envs[i], tail)
        distance = distance_df.loc[env, closest_envs[i]]   

        distances.append(distance)
        predicted_interaction = d[t]
        pred.append(predicted_interaction)    
    return true, pred, distances

def threshold_pred(true, pred, distances):
    """
    Plots accuracy scores at increasing similarity thresholds between environments.

    Input:
        true: List of ground truth interaction types.
        pred: List of predicted interaction types.
        distances: List of distances used for substitution.
    """
    df = pd.DataFrame({'true':true, 'pred':pred, 'distance': distances})
    threshold_values = np.linspace(df['distance'].min(), df['distance'].max(), 10)  
    accuracies = []
    for threshold in threshold_values:
        filtered_df = df[df['distance'] <= threshold]
        acc = accuracy_score(filtered_df['true'], filtered_df['pred']) 
        accuracies.append(round(acc, 2) * 100) 

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_values, accuracies, marker = 'o', linestyle = '--', color = 'black')
    plt.xlabel('Threshold', fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 16)
    ax = plt.gca()  # Get current axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.ylim(69, 80)
    plt.show()

def main():
    train = read("datasets/MKG/train.txt")
    valid = read("datasets/MKG/valid.txt")
    test = read("datasets/MKG/test.txt")
    model_path = "models/MKG/final_original.chkpnt" #"models/MKG/final_original.chkpnt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location = device)
    data = KGDataset('datasets/MKG')
    rel2id = data.rel2id
    env_embeddings = get_rel_emb(rel2id, model)
    merged_df = filter_merge(env_embeddings)
    distance_df = compute_pairwise_distances(merged_df)
    sorted_distance_dict = create_sorted_distance_dict(distance_df)
    d = {}
    for triple in train + valid:
        head, rel, tail = triple
        env, interaction = rel.split('_')
        d[(head, env ,tail)] = interaction 

    # Environments for which at least one of the interaction type (positive, neutral or negative) is missing are filtered out
    test_filter = [t for t in test if t[1].split('_')[0] in sorted_distance_dict]
    true, pred, distances = predict_interaction_type(test_filter, d, sorted_distance_dict, distance_df)
    print("Results for the similarity-based approach")
    print('===================================')
    print(classification_report(true, pred))
    threshold_pred(true, pred, distances)

    print("Results for random method")
    print('===================================')
    for k in sorted_distance_dict.keys():
        random.shuffle(sorted_distance_dict[k])
    true, pred, _ = predict_interaction_type(test_filter, d, sorted_distance_dict, distance_df)
    print(classification_report(true, pred))


if __name__ == '__main__':
    main()


    




