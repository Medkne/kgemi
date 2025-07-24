import pandas as pd
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import MinMaxScaler
import argparse




parser = argparse.ArgumentParser()
parser.add_argument('-path', default="models/MKG/final_original.chkpnt", type=str, help="model path")
args = parser.parse_args()


def load_species_ids(filepath):
    """
    Loads microbial strain names and their corresponding IDs.

    Input:
        filepath: Path to the 'ent2ids.txt' file.

    Returns:
        A dictionary mapping strain names to integer IDs.
    """
    species_to_id = {}
    with open(filepath, 'r') as file:
        for line in file:
            species, _id = line.strip().split('\t')
            species_to_id[species] = int(_id)
    return species_to_id

def get_embeddings(model, species_ids, device):
    """
    Retrieves the head and tail embeddings for each strain and concatenates them into a single vector.
    Embeddings are scaled between 0 and 1 using MinMaxScaler.

    Inputs:
        model: The trained model containing entity embeddings.
        species_ids: Dictionary mapping strain names to their IDs.
        device: Torch device (CPU or CUDA).

    Returns:
        A pandas DataFrame containing the normalized embeddings for each strain.
    """
    head_embeddings = {}
    tail_embeddings = {}
    
    for species, _id in species_ids.items():
        _id_tensor = torch.tensor([_id], device = device)
        head_emb = model.ent_h_embs(_id_tensor).detach().cpu().numpy().flatten()
        tail_emb = model.ent_t_embs(_id_tensor).detach().cpu().numpy().flatten()
        head_embeddings[species] = head_emb
        tail_embeddings[species] = tail_emb

    head_embeddings_df = pd.DataFrame.from_dict(head_embeddings, orient='index')
    tail_embeddings_df = pd.DataFrame.from_dict(tail_embeddings, orient='index')

    head_embeddings_df.columns = [f'h_{i+1}' for i in range(head_embeddings_df.shape[1])]
    tail_embeddings_df.columns = [f't_{i+1}' for i in range(tail_embeddings_df.shape[1])]

    concat_embeddings = pd.concat([head_embeddings_df, tail_embeddings_df], axis = 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(concat_embeddings)
    strain_embs = pd.DataFrame(scaled, columns=concat_embeddings.columns, index= concat_embeddings.index)
    return strain_embs


def tsne_plot(embeddings):
    """
    Visualizes strain embeddings in 2D space using t-SNE.

    Input:
        embeddings: DataFrame containing strain embeddings.

    Returns:
        A t-SNE plot where each point is labeled by strain and colored by taxonomic order.
    """
    tsne = TSNE(n_components=2, random_state=45, perplexity=5) 
    tsne_results = tsne.fit_transform(embeddings)
    tsne_df = pd.DataFrame(tsne_results, index=embeddings.index, columns=['TSNE1', 'TSNE2'])
    mapi = pd.read_excel('datasets/Mapping.xlsx')
    tsne_df['Order'] = '..'
    for ind in tsne_df.index:
        tsne_df.loc[ind, 'Order'] = mapi.loc[mapi['Strain shorthand'] == ind, 'Order'].item()
    colors = {'Enterobacterales': 'cornflowerblue', 'Pseudomonadales': 'darksalmon'}  
    plt.figure(figsize=(12, 7))
    for species, point in tsne_df.iterrows():
        order = point['Order']
        color = colors[order]  
        plt.scatter(point['TSNE1'], point['TSNE2'], marker='o', color=color)
        plt.text(point['TSNE1'] + 3.5, point['TSNE2'] + 3, species, fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[order], markersize=10, label=order) for order in colors]
    plt.legend(handles=legend_elements, title='Order', fontsize = 16, bbox_to_anchor=(0.8, 0.9), title_fontsize = 16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    plt.show()


def cluster_viz(embeddings):
    """
    Displays a hierarchical clustering dendrogram of strain embeddings.

    Input:
        embeddings: DataFrame containing strain embeddings.

    Returns:
        A dendrogram visualizing hierarchical clustering of the strains.
    """
    Z = linkage(embeddings, method='ward', metric='euclidean')
    plt.figure(figsize=(12, 7))
    dend = dendrogram(
        Z,
        labels=embeddings.index,
        color_threshold = 0,
        leaf_rotation=45)
    plt.ylim(8.5, 20)
    plt.xlabel('Strains')
    plt.ylabel('Distance')
    plt.show()


def main():
    model_path = args.path #"/home/mkhatban/coding/pro1/models/interactions/850.chkpnt"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, device)
    species_ids = load_species_ids("datasets/MKG/ent2ids.txt")
    embeddings = get_embeddings(model, species_ids, device)
    tsne_plot(embeddings)
    cluster_viz(embeddings)

if __name__ == "__main__":
    main()