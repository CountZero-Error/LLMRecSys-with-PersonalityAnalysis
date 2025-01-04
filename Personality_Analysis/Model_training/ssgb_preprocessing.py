from functorch.dim import split
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from nltk.corpus import wordnet
from datasets import Dataset
from sympy.codegen.ast import continue_
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import torch
import time
import nltk
import os

class preprocessing:
    def __init__(self, fi, labels, original_embedding, augmented_embedding):
        self.fi = fi
        self.labels = labels
        self.original_embedding = original_embedding
        self.augmented_embedding = augmented_embedding

        # Initialize the OneHotEncoder
        self.encoder = OneHotEncoder(sparse_output=False)

    def preview(self, df):
        print(f'[*] Data preview:')
        print('------------------------------------------------------------------------')
        print(df.info())

        print('\n')
        for i in range(5):
            print(f'{df.iloc[i]}\n')

        print('------------------------------------------------------------------------')

    def compute_embedding_consistency_loss(self, original_embeddings, augmented_embeddings):
        # calculate Euclidean distance for consistency_loss
        consistency_loss = torch.mean((original_embeddings - augmented_embeddings) ** 2)
        return consistency_loss

    def personality_label_encoding(self, personality_labels):
        """
        This function is to encode the labels of personalities
        """
        sample_labels = np.array(personality_labels).reshape(-1, 1)

        # Fit and transform the labels
        one_hot_encoded = self.encoder.fit_transform(sample_labels)

        return one_hot_encoded

    def label_encoding(self, labels):
        """
        This function is to encode users' labels
        """
        encoded_labels = []
        for label in tqdm(labels):
            n = np.array(label)
            e_label = self.encoder.transform(n.reshape(-1, 1))
            encoded_labels.append(e_label)

        return encoded_labels

    def standardization(self, numerical_features, data):
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        return data

    def convert_embedding(x):
        if isinstance(x, str):
            # If it's a string, assume it's a JSON-like list and convert
            return np.array(eval(x))
        elif isinstance(x, (list, np.ndarray)):
            # If it's already a list or array, convert it directly
            return np.array(x)
        else:
            # If it's an unexpected type, raise an error for manual inspection
            raise ValueError(f"[*] Unexpected type {type(x)} for embedding: {x}")

    def run(self):
        start = time.time()

        print(f'[*] Reading from {os.path.basename(self.fi)}...')
        df = pd.read_csv(self.fi)
        self.preview(df)

        df_len = df.shape[0]

        print(f'\n[*] Start transforming data({df_len} records)...')

        # proceed features
        if 'labels' not in df.columns:
            print("[*] Please re-upload dataset with labels")
            return

        labeled_df = df[df['labels'].notna()]
        unlabeled_df = df[df['labels'].isna()]
        original_embedding_df = original_embedding.to_pandas()
        augmented_embedding_df = augmented_embedding.to_pandas()
        labeled_user = labeled_df.merge(original_embedding_df, on='user_id')
        unlabeled_user = unlabeled_df.merge(original_embedding_df, on='user_id')

        # standardization
        numerical_features = ['average_time_float', 'purchase_ratio', 'brand_loyalty_ratio', 'average_price']
        labeled_user = self.standardization(numerical_features, labeled_user)
        unlabeled_user = self.standardization(numerical_features, unlabeled_user)

        # convert embeddings
        labeled_user['average_embedding'] = labeled_user['average_embedding'].apply(self.convert_embedding)
        unlabeled_user['average_embedding'] = unlabeled_user['average_embedding'].apply(self.convert_embedding)

        # combine features
        labeled_features = np.hstack([labeled_user[numerical_features].values, np.vstack(labeled_user['average_embedding'].values)])
        unlabeled_features = np.hstack([unlabeled_user[numerical_features].values, np.vstack(unlabeled_user['average_embedding'].values)])

        seed_indices = list(range(len(labeled_features)))
        all_features = np.vstack([labeled_features, unlabeled_features])

        # proceed labels
        one_hot_encoded = self.personality_label_encoding(self.labels)
        split_labels = labeled_user['labels'].str.split('.')

        encoded_labels = self.label_encoding(split_labels)

        return all_features, seed_indices, encoded_labels

if __name__ == '__main__':
    access_token = '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'
    original_embedding = load_dataset(
        "CookieLyu/Category_Codes",
        revision="1000k_average_embedded",
        token=access_token,
        variant='ori_category_average'
    )
    augmented_embedding = load_dataset(
        "CookieLyu/Category_Codes",
        revision="1000k_average_embedded",
        token=access_token,
        variant='aug_category_average'
    )