from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np

class clustering:
    def __init__(self,):

    def construct_graph(data, k=5):
        """
        Construct a k-NN graph from data points using cosine similarity.
        """
        # Compute cosine similarity matrix
        similarity = cosine_similarity(data)
        n_samples = data.shape[0]
        adj_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            # Get indices of k most similar neighbors (excluding self)
            k_neighbors = np.argsort(similarity[i])[-(k + 1):-1][::-1]

            for j in k_neighbors:
                # Use similarity directly for the adjacency matrix
                adj_matrix[i][j] = similarity[i][j]
                adj_matrix[j][i] = adj_matrix[i][j]

        return adj_matrix

    def compute_laplacian(self, graph):
        adj_matrix = nx.to_numpy_array(graph)
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        laplacian_matrix = degree_matrix - adj_matrix
        return laplacian_matrix

    def compute_laplacian(self, adj_matrix):
        """
        Compute the random walk Laplacian matrix.
        """

        # Create degree matrix
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

        # # Handle cases where the degree matrix might be singular
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     degree_matrix_inv = np.linalg.inv(degree_matrix)
        #     degree_matrix_inv[np.isinf(degree_matrix_inv)] = 0  # Replace infinities (from 1/0) with 0
        #     degree_matrix_inv = np.nan_to_num(degree_matrix_inv) # Replace NaNs (from 0/0) with 0

        # P is transition matrix, which derived from the adjacency matrix A and the degree matrix D
        # P = D^(-1) * A
        # Each entry Pij represents the probability of transitioning from node i to node j during the random walk
        # L = I - P

        # linalg.inv() compute the inverse of a matrix
        # matmul() matrix product of two arrays
        # np.eye() Return a 2-D array with ones on the diagonal and zeros elsewhere.
        # I = identify matrix 单位矩阵
        laplacian = np.eye(len(adj_matrix)) - np.matmul(np.linalg.inv(degree_matrix), adj_matrix)
        return laplacian

    def random_walk(self, laplacian, label, data, seed_indice, steps=3):
        """
        Perform a random walk on the graph starting from the seed nodes.

        If X^(t) is the label score matrix at step t, the random walk process can be expressed as:
            X^(t+1) = P * X^(t) or equivalently X^(t+1) = (I - L) * X^(t)
        Here, L is the random walk Laplacian.

        label: list
        """
        n_samples = data.shape[0]
        n_classes = label[0].shape[1]

        # Create empty labels for unlabeled usr
        emp = [np.array([0.] * n_classes)] * (n_samples - len(seed_indice))
        # print('emp: ', emp)
        _label = []
        for x in label:
            _label.append(np.sum(x, axis=0))

        label = _label + emp
        label_matrix = np.matrix(label)
        print(label_matrix.shape)
        print(label_matrix)

        # Normalize the initial distribution for labeled nodes
        # initial_vector = initial_vector / (initial_vector.sum(axis=1, keepdims=True) + 1e-10)

        # Transition matrix (random walk matrix)
        transition_matrix = np.eye(n_samples) - laplacian

        # Perform the random walk with laplacian
        # matrix_power(), raise a square matrix to the (integer) power n.
        random_walk_result = np.linalg.matrix_power(transition_matrix, steps).dot(label_matrix)

        # Normalize rows of the result (optional: interpret as probabilities)
        row_sums = random_walk_result.sum(axis=1).reshape(-1, 1)
        normalized_result = random_walk_result / (row_sums + 1e-10)

        return normalized_result

    # Step 4: Candidate Set Extraction
    def extract_candidate_set(self, random_walk_result, threshold=0.3):
        """
        Extract candidate nodes with multiple labels based on random walk scores.

        """
        candidates = {}
        n_samples, n_classes = random_walk_result.shape

        for i in range(n_samples):
            # Find all classes for the current node that exceed the threshold
            labels_with_probs = [
                (c, f"{random_walk_result[i, c] * 100:.2f}%") for c in range(n_classes) if
                random_walk_result[i, c] > threshold]
            if labels_with_probs:
                candidates[i] = labels_with_probs

        return candidates