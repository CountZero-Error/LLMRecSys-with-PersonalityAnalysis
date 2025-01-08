from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import time
import os


class preprocessing:
    def __init__(self, fi, label_types, original_embedding, augmented_embedding):
        self.fi = fi
        self.label_types = label_types
        self.original_embedding = original_embedding
        self.augmented_embedding = augmented_embedding

        # Initialize the OneHotEncoder
        self.encoder = OneHotEncoder(sparse_output=False)

    def get_OneHotEncoder_label(self):
        return self.encoder.categories_

    def parse_labels(self, label_str):
        if isinstance(label_str, str):
            # Split the string by commas and strip whitespace
            processed_labels = [label.strip() for label in label_str.split(',')]
            return processed_labels
        else:
            raise ValueError(f"Input must be a string. Received type {type(label_str)}.")

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
        orig_embeddings_tensor = torch.from_numpy(original_embeddings).float()
        aug_embeddings_tensor = torch.from_numpy(augmented_embeddings).float()

        consistency_loss = torch.mean((orig_embeddings_tensor - aug_embeddings_tensor) ** 2)

        return consistency_loss

    def personality_label_encoding(self, labels):
        """
        This function is to encode the labels of personalities
        """
        sample_labels = np.array(labels).reshape(-1, 1)
        # Fit and transform the labels
        one_hot_encoded = self.encoder.fit_transform(sample_labels)

        # Mapping of categories
        print("\nCategories Mapping:")
        print(self.encoder.categories_)

        return one_hot_encoded

    def label_encoding(self, usr_labels):
        """
        This function is to encode users' labels
        """
        encoded_labels = []
        for label in tqdm(usr_labels):
            n = np.array(label)
            e_label = self.encoder.transform(n.reshape(-1, 1))
            encoded_labels.append(e_label)

        merged_labels = []
        for user_labels in encoded_labels:
            # Sum across all rows for each user's array
            merged_label = np.sum(user_labels, axis=0)
            # Ensure the resulting vector is binary (convert counts > 1 to 1)
            merged_label[merged_label > 1] = 1
            merged_labels.append(merged_label)

        return merged_labels

    def standardization(self, numerical_features, data):
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])

        return data

    def convert_embedding(self, x):
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

        if 'labels' not in df.columns:
            raise ValueError("[*] Dataset must include a 'labels' column.")

        # Split data into labeled and unlabeled subsets
        labeled_df = df[df['labels'].notna()]
        unlabeled_df = df[df['labels'].isna()]
        original_embedding_df = self.original_embedding['train'].to_pandas()
        augmented_embedding_df = self.augmented_embedding['train'].to_pandas()

        # Merge with embeddings
        labeled_user = labeled_df.merge(original_embedding_df, on='user_id')
        labeled_user = labeled_user.merge(augmented_embedding_df, on='user_id', suffixes=('_orig', '_aug'))

        unlabeled_user = unlabeled_df.merge(original_embedding_df, on='user_id')
        unlabeled_user = unlabeled_user.merge(augmented_embedding_df, on='user_id', suffixes=('_orig', '_aug'))

        # standardization
        numerical_features = ['average_time_float', 'purchase_ratio', 'brand_loyalty_ratio', 'average_price']
        labeled_user = self.standardization(numerical_features, labeled_user)
        unlabeled_user = self.standardization(numerical_features, unlabeled_user)

        # convert embeddings
        labeled_user['embedding_orig'] = labeled_user['average_embedding_orig'].apply(self.convert_embedding)
        labeled_user['embedding_aug'] = labeled_user['average_embedding_aug'].apply(self.convert_embedding)

        unlabeled_user['embedding_orig'] = unlabeled_user['average_embedding_orig'].apply(self.convert_embedding)
        unlabeled_user['embedding_aug'] = unlabeled_user['average_embedding_aug'].apply(self.convert_embedding)

        # Compute consistency loss
        consistency_loss = self.compute_embedding_consistency_loss(
            np.vstack(labeled_user['embedding_orig'].values),
            np.vstack(labeled_user['embedding_aug'].values)
        )

        # combine features (fused embeddings)
        a = 0.5
        labeled_features = np.hstack([
            labeled_user[numerical_features].values,
            a * np.vstack(labeled_user['embedding_orig'].values) + (1 - a) * np.vstack(
                labeled_user['embedding_aug'].values)
        ])

        unlabeled_features = np.hstack([
            unlabeled_user[numerical_features].values,
            a * np.vstack(unlabeled_user['embedding_orig'].values) + (1 - a) * np.vstack(
                unlabeled_user['embedding_aug'].values)
        ])

        seed_indices = list(range(len(labeled_features)))
        all_features = np.vstack([labeled_features, unlabeled_features])

        # Proceed with labels
        one_hot_encoded = self.personality_label_encoding(self.label_types)
        split_labels = labeled_user['labels'].str.split('.')

        encoded_usr_labels = self.label_encoding(split_labels)

        print(f"[*] Preprocessing completed in {time.time() - start:.2f} seconds.")
        print("[*] The result of OneHotEncoder is: ", one_hot_encoded)

        return all_features, seed_indices, encoded_usr_labels, consistency_loss


class GraphClustering:
    def __init__(self, features, k_neighbors, threshold, labels, seed_indices, consistency_loss, step):
        self.features = features
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.labels = labels
        self.seed_indices = seed_indices
        self.consistency_loss = consistency_loss
        self.step = step

    def construct_graph(self):
        """
        Construct a k-NN graph from data points using cosine similarity.
        """
        # Compute cosine similarity matrix
        similarity = cosine_similarity(self.features)
        n_samples = self.features.shape[0]
        adj_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Get indices of k most similar neighbors (excluding self)
            knn = np.argsort(similarity[i])[-(self.k_neighbors + 1):-1][::-1]

            for j in knn:
                # adjust weight using labels
                if i in self.seed_indices and j in self.seed_indices:
                    if np.argmax(self.labels[i]) == np.argmax(self.labels[j]):
                        # Increase weight for same-class neighbors
                        weight = similarity[i][j] * 2
                    else:
                        # Decrease weight for different-class neighbors
                        weight = similarity[i][j] * 0.5
                else:
                    # Optional: Apply threshold to filter noisy connections
                    weight = similarity[i][j] if similarity[i][j] > 0.1 else 0

                # Use similarity directly for the adjacency matrix
                adj_matrix[i][j] = weight
                adj_matrix[j][i] = weight

        return adj_matrix

    def compute_laplacian(self, adj_matrix):
        """
        Compute the random walk Laplacian matrix.
        """
        # Create degree matrix
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

        # P is transition matrix, which derived from the adjacency matrix A and the degree matrix D
        # P = D^(-1) * A
        # Each entry Pij represents the probability of transitioning from node i to node j during the random walk
        # L = I - P
        laplacian = np.eye(len(adj_matrix)) - np.matmul(np.linalg.inv(degree_matrix), adj_matrix)

        return laplacian

    def random_walk(self, laplacian, normalize=True):
        """
        Perform a random walk on the graph starting from the seed nodes.

        If X^(t) is the label score matrix at step t, the random walk process can be expressed as:
            X^(t+1) = P * X^(t) or equivalently X^(t+1) = (I - L) * X^(t)
        Here, L is the random walk Laplacian.

        label: list
        """
        n_samples = self.features.shape[0]
        n_classes = len(self.labels[0])

        # Validate inputs
        if len(self.labels) != len(self.seed_indices):
            raise ValueError("The number of labels must match the number of seed indices.")
        if not (0 <= np.min(self.seed_indices) < n_samples):
            raise ValueError("Seed indices must be within the range of data samples.")

        _label = []
        for x in self.labels:
            # Assume each x is a merged One-Hot encoded vector
            if isinstance(x, (np.ndarray, list)):
                merged_label = np.sum(x, axis=0) if len(x.shape) > 1 else x
                merged_label[merged_label > 1] = 1  # Ensure binary format
                _label.append(merged_label)
            else:
                raise ValueError(f"Unexpected label format for user: {x}")

        # Create empty labels for unlabeled users (all zeros)
        emp = [np.zeros(n_classes)] * (n_samples - len(self.seed_indices))

        # Combine seed and unlabeled labels
        label = _label + emp

        # Convert to matrix for further processing
        label_matrix = np.matrix(label)

        # Normalize the initial distribution for labeled nodes
        # initial_vector = initial_vector / (initial_vector.sum(axis=1, keepdims=True) + 1e-10)

        # Transition matrix (random walk matrix)
        transition_matrix = np.eye(n_samples) - laplacian

        # Perform the random walk with laplacian
        # matrix_power(), raise a square matrix to the (integer) power n.
        random_walk_result = np.linalg.matrix_power(transition_matrix, self.step).dot(label_matrix)
        # print(f'Random Walk Result:\n{type(random_walk_result)}\n')

        # Incorporate consistency loss as a regularizer
        # Add a fraction of consistency_loss to adjust the random walk result
        # random_walk_result += self.consistency_loss.item() * 0.5  # weight the consistency loss contribution
        random_walk_result += self.consistency_loss.item() * 0.5 + 0.01 * np.sum(np.square(np.array(label_matrix)),
                                                                                 axis=1, keepdims=True)

        # Normalize rows of the result (optional: interpret as probabilities)
        if normalize:
            row_sums = random_walk_result.sum(axis=1).reshape(-1, 1)
            random_walk_result = random_walk_result / (row_sums + 1e-10)

        return random_walk_result

    # Step 4: Candidate Set Extraction
    def extract_candidate_set(self, random_walk_result):
        """
        Extract candidate nodes with multiple labels based on random walk scores.
        """
        candidates = {}
        n_samples, n_classes = random_walk_result.shape

        for i in range(n_samples):
            # Find all classes for the current node that exceed the threshold
            labels_with_probs = [
                {"class": c, "prob": random_walk_result[i, c],
                 "percent": f"{round(random_walk_result[i, c] * 100, 2)}%"} for c in range(n_classes) if
                random_walk_result[i, c] > self.threshold]
            if labels_with_probs:
                candidates[i] = labels_with_probs

        return candidates


    def run(self):
        graph = self.construct_graph()
        lap = self.compute_laplacian(graph)
        result = self.random_walk(lap)

        candidate = self.extract_candidate_set(result)

        return candidate, graph


if __name__ == '__main__':
    fi = '/Users/cookie/Desktop/FYP/1000k/parameter_labeled.csv'
    # dir_o = args.OUTPUT_DIRECTORY
    label_types = ['Night_owl', 'Early_bird', 'Decisive', 'Brand_loyalty', 'Maker', 'Homebody', 'Culinarian', 'Geek',
                   'Photophile', 'Media_Aficionado', 'Audiophile', 'Fashionista', 'Lifestyle', 'Car_Enthusiast',
                   'Caregiver', 'Health_Enthusiast', 'Farm', 'Sport', 'high_consumer', 'Mid_Consumer']
    k_neighbors = 5
    threshold = 0.1
    step = 3

    access_token = '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'
    original_embedding = load_dataset(
        "CookieLyu/Category_Codes",
        revision="1000k_average_embedded",
        token=access_token
    )

    augmented_embedding = load_dataset(
        "CookieLyu/Category_Codes",
        revision="1000k_average_embedded_aug",
        token=access_token
    )

    # Step 1: Preprocessing
    print("[*] Running preprocessing...")
    preprocessor = preprocessing(fi, label_types, original_embedding, augmented_embedding)
    all_features, seed_indices, merged_labels, consistency_loss = preprocessor.run()
    OneHotLabels = preprocessor.get_OneHotEncoder_label()

    # Step 2: Semi-supervised Clustering
    print("[*] Running clustering...")
    cluster = GraphClustering(all_features, k_neighbors, threshold, merged_labels, seed_indices, consistency_loss, step)
    candidates, graph = cluster.run()

    # 打印候选节点结果
    print("[*] Candidates extracted from clustering:")
    nodes = {'user_id': [], 'labels': []}
    user_df = pd.read_csv(fi)
    for node, info in candidates.items():
        if node < 180:
            print(f"Node {node}: {info}")

        curr_nodes = []
        for _info in info:
            curr_nodes.append(OneHotLabels[0][int(_info['class'])])

        if curr_nodes != []:
            nodes['user_id'].append(user_df.loc[int(node), 'user_id'])
            nodes['labels'].append('.'.join(curr_nodes))
    print(f'{len(nodes['user_id'])} out of {len(candidates.keys())} extracted.')

    nodes_fo = '/Users/cookie/Desktop/labeled_users.csv'
    print(f'[*] Saving to {nodes_fo}...')
    nodes_df = pd.DataFrame(nodes)
    nodes_df.to_csv(nodes_fo, index=False)
