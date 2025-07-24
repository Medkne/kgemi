import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
import itertools




def one_hot_encode_features(df):
    """
    One-hot encode 'Bug 1', 'Bug 2', and 'Carbon' columns in the given DataFrame.
    Returns a new DataFrame with the original columns plus the one-hot encoded columns.
    """
    # Encode the Class 
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

    # Encode Bug 1 and Bug 2
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    all_bugs = pd.concat([df['Bug 1'], df['Bug 2']]).unique().reshape(-1, 1)
    encoder.fit(all_bugs)

    bug1_ohe = encoder.transform(df['Bug 1'].values.reshape(-1, 1))
    bug2_ohe = encoder.transform(df['Bug 2'].values.reshape(-1, 1))

    bug1_ohe_df = pd.DataFrame(
        bug1_ohe, 
        columns=[f'Bug1_{cat}' for cat in encoder.categories_[0]], 
        index=df.index
    )
    bug2_ohe_df = pd.DataFrame(
        bug2_ohe, 
        columns=[f'Bug2_{cat}' for cat in encoder.categories_[0]], 
        index=df.index
    )

    data = pd.concat([df, bug1_ohe_df, bug2_ohe_df], axis=1)

    # Encode Carbon
    carbon_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    carbon_ohe = carbon_encoder.fit_transform(data[['Carbon']])
    carbon_ohe_df = pd.DataFrame(
        carbon_ohe, 
        columns=[f'Carbon_{cat}' for cat in carbon_encoder.categories_[0]], 
        index=data.index
    )
    data = pd.concat([data, carbon_ohe_df], axis=1)
    return data

def split_df_by_holdout_receiver_environments(df, heldout_receivers=['RP1', 'EA', 'PH', 'PR1', 'KA'], val_receiver = 'EC'):
    """
    Splits a DataFrame into train, test, and validation sets.
    - Validation: all rows where Bug 1 == val_sender
    - Test: all rows where Bug 1 is in heldout_receivers (excluding val_sender)
    - Train: all remaining rows

    Returns:
        train_df, test_df, val_df
    """
    val_mask = df['Bug 1'] == val_receiver
    test_mask = df['Bug 1'].isin(heldout_receivers) & (~val_mask)
    train_mask = ~(val_mask | test_mask)

    valid_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    train_df = df[train_mask].copy()
    # remove the original columns
    train_df = train_df.drop(columns=['Bug 1', 'Bug 2', 'Carbon'])
    valid_df = valid_df.drop(columns=['Bug 1', 'Bug 2', 'Carbon'])
    test_df = test_df.drop(columns=['Bug 1', 'Bug 2', 'Carbon'])

    return train_df, valid_df, test_df



def xgboost_hyperparameter_search(train_data, valid_data, test_data):
    # Prepare the data
    X_train = train_data.drop(columns=['Class'])
    y_train = train_data['Class']
    X_valid = valid_data.drop(columns=['Class'])
    y_valid = valid_data['Class']
    X_test = test_data.drop(columns=['Class'])
    y_test = test_data['Class']

    # Define the hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 500, 1000],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
        'gamma': [0, 0.5, 1],
        'reg_lambda': [0, 0.1, 1],
        'reg_alpha': [0, 0.1, 1]
    }

    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    best_f1 = -1
    best_params = None

    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        print('Parameters:', params)
        model = xgb.XGBClassifier(eval_metric='mlogloss', **params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        f1 = f1_score(y_valid, y_pred, average='macro')
        print("Validation F1 Score:", f1)
        if f1 > best_f1:
            best_f1 = f1
            best_params = params
    # Retrain on train+valid with best params
    X_train_full = pd.concat([X_train, X_valid])
    y_train_full = pd.concat([y_train, y_valid])
    final_model = xgb.XGBClassifier(eval_metric='mlogloss', random_state = 42, **best_params)
    final_model.fit(X_train_full, y_train_full)

    # Evaluate on test set
    y_test_pred = final_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')

    print("Best Parameters:", best_params)
    print("Test Accuracy:", test_acc)
    print("Test F1 Score:", test_f1)
    with open('best_xgboost_params_receiver.txt', 'a') as f:
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")

    return final_model, best_params, test_acc, test_f1



def main():
    data = pd.read_csv('/home/mkhatban/coding/Microbial-interaction-prediction/Data/Features.csv')
    data = data[data['Carbon'] != 'Water']
    data['Class'] = "Negative"
    data.loc[data['2 on 1: Effect'] > 0, 'Class'] = "Positive"
    data.loc[data['2 on 1: Effect'] == 0, 'Class'] = "Neutral"
    cols = ['Bug 1', 'Bug 2','Carbon', 'Class']
    df = data[cols].copy()
    df_encoded = one_hot_encode_features(df)
    train_data, valid_data, test_data = split_df_by_holdout_receiver_environments(
    df_encoded, 
    heldout_receivers=['RP1', 'EA', 'PH', 'PR1', 'KA'],
    val_receiver='EC')
    xgboost_hyperparameter_search(train_data, valid_data, test_data)

if __name__ == "__main__":
    main()



