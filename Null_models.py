from collections import defaultdict, Counter
from sklearn.metrics import classification_report
import numpy as np


def null_model_A(test_triples):
    '''
    predicts the class of a triple based on the global distribution of interaction types
    The most frequent class is Negative
    '''
    negative_count = 0
    for triple in test_triples:
        if triple[1].split('_')[1] == 'Negative':
            negative_count += 1
    total_count = len(test_triples)
    negative_percentage = (negative_count / total_count) * 100
    print(f"Percentage of Negative interactions in test set: {negative_percentage:.2f}%")


def null_model_B(train_triples, test_triples): 
    '''
    predicts the class of a triple based on the distribution of interaction types in the carbon source environment
    in the training set.
    '''
    env_to_class_counts = defaultdict(list)
    for triple in train_triples:
        carbon_source = triple[1].split('_')[0]
        interaction_type = triple[1].split('_')[1]
        env_to_class_counts[carbon_source].append(interaction_type)
    # compute the most commen class for each carbon source
    env_to_most_common_class = {env: Counter(classes).most_common(1)[0][0] for env, classes in env_to_class_counts.items()}
    # predict the class for each triple in the test set
    true = []
    pred = []
    for triple in test_triples:
        carbon_source = triple[1].split('_')[0]
        true.append(triple[1].split('_')[1])
        pred.append(env_to_most_common_class[carbon_source])
    print(classification_report(true, pred))



def null_model_C(train_triples, test_triples):
    '''
    predict the interaction type in the test set based on the most frequent class involving a given receiver.
    '''
    receiver_to_class_counts = defaultdict(list)
    for triple in train_triples:
        receiver = triple[2]
        interaction_type = triple[1].split('_')[1]
        receiver_to_class_counts[receiver].append(interaction_type)
    # compute the most common class for each receiver strain
    receiver_to_most_common_class = {receiver: Counter(classes).most_common(1)[0][0] for receiver, classes in receiver_to_class_counts.items()}
    # predict the class for each triple in the test set
    true = []
    pred = []
    for triple in test_triples:
        receiver = triple[2]
        true.append(triple[1].split('_')[1])
        pred.append(receiver_to_most_common_class.get(receiver))
    print(classification_report(true, pred))





def null_model_D(train_triples, test_triples):
    '''
    predict the interaction type in the test set based on the most frequent class involving a given sender.
    '''
    sender_to_class_counts = defaultdict(list)
    for triple in train_triples:
        sender = triple[0]
        interaction_type = triple[1].split('_')[1]
        sender_to_class_counts[sender].append(interaction_type)
    # compute the most common class for each sender strain
    sender_to_most_common_class = {sender: Counter(classes).most_common(1)[0][0] for sender, classes in sender_to_class_counts.items()}
    # predict the class for each triple in the test set
    true = []
    pred = []
    for triple in test_triples:
        sender = triple[0]
        true.append(triple[1].split('_')[1])
        pred.append(sender_to_most_common_class.get(sender))
    print(classification_report(true, pred))



def null_model_E(train_triples, test_triples):
    '''
    predicts the majority interaction type for each (sender, receiver) pair across environments.
    '''

    pair_to_class_counts = defaultdict(list)
    for triple in train_triples:
        pair = (triple[0], triple[2])
        interaction_type = triple[1].split('_')[1]
        pair_to_class_counts[pair].append(interaction_type)
    # compute the most common class for each pair
    pair_to_most_common_class = {pair: Counter(classes).most_common(1)[0][0] for pair, classes in pair_to_class_counts.items()}
    # predict the class for each triple in the test set
    true = []
    pred = []
    for triple in test_triples:
        pair = (triple[0], triple[2])
        true.append(triple[1].split('_')[1])
        pred.append(pair_to_most_common_class.get(pair))
    print(classification_report(true, pred))

def read_triples(file_path):
    triples = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            triples.append((parts[0], parts[1], parts[2]))
    return triples


if __name__ == '__main__':

    training = read_triples("/home/mkhatban/coding/classification-kg/classification/datasets/original_without/train.txt")
    test = read_triples("/home/mkhatban/coding/classification-kg/classification/datasets/original_without/test.txt")
    valid = read_triples("/home/mkhatban/coding/classification-kg/classification/datasets/original_without/valid.txt")

    print("Results for Null Model A:")
    null_model_A(test)

    print("\nResults for Null Model B:")
    null_model_B(training, test)

    print("\nResults for Null Model C:")
    null_model_C(training, test)

    print("\nResults for Null Model D:")
    null_model_D(training, test)

    print("\nResults for Null Model E:")
    null_model_E(training, test)