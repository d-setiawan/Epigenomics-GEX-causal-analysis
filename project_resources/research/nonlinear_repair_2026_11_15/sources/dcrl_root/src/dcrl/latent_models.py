import random
import numpy as np


class DiverseBTree:
    def __init__(self, N, K, p0, p1, p2, p3, p4):
        self.K = K
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.tree_structure = []
        self.chain_structure = []
        self.N = N
        self.matrix = None

    def _generate_tree_structure(self):
        nearest_power_of_two_minus_1 = 2 ** (int(np.log2(self.K + 1))) - 1
        remaining_nodes = nearest_power_of_two_minus_1
        layers = []
        while remaining_nodes > 0:
            power_of_two = 2 ** (int(np.log2(remaining_nodes)))
            layers.append(power_of_two)
            remaining_nodes -= power_of_two
        self.tree_structure = layers

    def _generate_chain_structure(self):
        remaining_nodes = self.K - (2 ** (int(np.log2(self.K + 1))) - 1)
        self.chain_structure = [i for i in range(self.K - remaining_nodes, self.K)]

    def _generate_matrix(self):
        matrix = np.zeros((self.N, self.K))
        matrix[:, range(self.tree_structure[0])] = (np.random.rand(self.N, self.tree_structure[0]) < 0.5)
        ter_ind = 0
        for level in range(1, len(self.tree_structure)):
            num_nodes = self.tree_structure[level]
            new_matrix = np.zeros((self.N, num_nodes))
            parent = [i for i in range(ter_ind, ter_ind + 2 * num_nodes)]
            child = [i for i in range(ter_ind + 2 * num_nodes, ter_ind + 3 * num_nodes)]
            focus = matrix[:, parent]
            for node in range(num_nodes):
                col1, col2 = 2 * node, 2 * node + 1
                new_matrix[:, node] = np.where(
                    (focus[:, col1] == 1) & (focus[:, col2] == 1),
                    self.p0,
                    np.where(
                        (focus[:, col1] == 1) & (focus[:, col2] == 0),
                        self.p1,
                        np.where((focus[:, col1] == 0) & (focus[:, col2] == 1), self.p2, self.p3),
                    ),
                )
                matrix[:, child] = (np.random.rand(self.N, num_nodes) < new_matrix)
            ter_ind = ter_ind + 2 * num_nodes
        for k in range(ter_ind + 1, self.K):
            prev_state = matrix[:, k - 1]
            matrix[:, k] = np.where(prev_state == 1, np.random.rand(self.N) < 1 - self.p4, np.random.rand(self.N) < self.p4)
        self.matrix = matrix

    def build(self):
        self._generate_tree_structure()
        self._generate_chain_structure()
        self._generate_matrix()
        return self.tree_structure, self.chain_structure, self.matrix


def generate_prob_markov(K, lob, upb):
    result = [0.5]
    for _ in range(K - 1):
        c = random.random()
        if c > 0.5:
            x = random.uniform(lob, upb)
            y = random.uniform(1 - upb, 1 - lob)
        else:
            y = random.uniform(lob, upb)
            x = random.uniform(1 - upb, 1 - lob)
        result.append((x, y))
    return result


def generate_markov_chain(N, K, probs):
    chain = np.zeros((N, K))
    chain[:, 0] = np.random.rand(N) < probs[0]
    for k in range(1, K):
        prev_state = chain[:, k - 1]
        p1, p0 = probs[k]
        chain[:, k] = np.where(prev_state == 1, np.random.rand(N) < p1, np.random.rand(N) < p0)
    return chain


def generate_tree(N, K, lob, upb, alternate_prob=True):
    tree = np.zeros((N, K), dtype=int)
    tree[:, 0] = np.random.rand(N) < 0.5
    node_index = 1
    parent_indices = [0]
    while node_index < K:
        next_parent_indices = []
        for parent_idx in parent_indices:
            if node_index >= K:
                break
            level = int(np.log2(parent_idx + 1)) + 1
            if alternate_prob:
                if level % 2 == 1:
                    p1, p0 = random.uniform(lob, upb), random.uniform(1 - upb, 1 - lob)
                else:
                    p1, p0 = random.uniform(1 - upb, 1 - lob), random.uniform(lob, upb)
            else:
                p1, p0 = random.uniform(lob, upb), random.uniform(1 - upb, 1 - lob)
            for _ in range(2):
                if node_index >= K:
                    break
                parent_state = tree[:, parent_idx]
                p = np.where(parent_state == 1, p1, p0)
                tree[:, node_index] = np.random.rand(N) < p
                next_parent_indices.append(node_index)
                node_index += 1
        parent_indices = next_parent_indices
    return tree


def diverse_tree(N, K, bounds):
    p0 = random.uniform(bounds.lob, bounds.upb)
    p1 = random.uniform(bounds.lob2, bounds.upb2)
    p2 = random.uniform(bounds.lob3, bounds.upb3)
    p3 = random.uniform(bounds.lob4, bounds.upb4)
    p4 = random.uniform(bounds.lob5, bounds.upb5)
    diverse_btree = DiverseBTree(N, K, p4, p0, p1, p2, p3)
    _, _, matrix = diverse_btree.build()
    return matrix


def model8(N, K, bounds):
    p0 = random.uniform(bounds.lob, bounds.upb)
    p1 = random.uniform(bounds.lob2, bounds.upb2)
    p2 = random.uniform(bounds.lob3, bounds.upb3)
    p3 = random.uniform(bounds.lob4, bounds.upb4)
    p4 = random.uniform(bounds.lob5, bounds.upb5)
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, 0] = np.random.rand() < 0.5
        matrix[i, 1], matrix[i, 2] = (
            np.random.rand() < p4 if matrix[i, 0] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 0] == 1 else np.random.rand() < p4,
        )
        matrix[i, 3], matrix[i, 4] = (
            np.random.rand() < p4 if matrix[i, 1] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 1] == 1 else np.random.rand() < p4,
        )
        matrix[i, 5], matrix[i, 6] = (
            np.random.rand() < p4 if matrix[i, 2] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 2] == 1 else np.random.rand() < p4,
        )
        matrix[i, 7] = (
            np.random.rand() < p0 if matrix[i, 4] == 1 and matrix[i, 6] == 1 else
            np.random.rand() < p1 if matrix[i, 4] == 1 and matrix[i, 6] == 0 else
            np.random.rand() < p2 if matrix[i, 4] == 0 and matrix[i, 6] == 1 else
            np.random.rand() < p3
        )
    return matrix


def model7(N, K, bounds):
    p0 = random.uniform(bounds.lob, bounds.upb)
    p1 = random.uniform(bounds.lob2, bounds.upb2)
    p2 = random.uniform(bounds.lob3, bounds.upb3)
    p3 = random.uniform(bounds.lob4, bounds.upb4)
    p4 = random.uniform(bounds.lob5, bounds.upb5)
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, 0] = np.random.rand() < 0.5
        matrix[i, 1], matrix[i, 2] = (
            np.random.rand() < p4 if matrix[i, 0] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 0] == 1 else np.random.rand() < p4,
        )
        matrix[i, 3], matrix[i, 4] = (
            np.random.rand() < p0 if matrix[i, 1] == 1 and matrix[i, 2] == 1 else
            np.random.rand() < p1 if matrix[i, 1] == 1 and matrix[i, 2] == 0 else
            np.random.rand() < p2 if matrix[i, 1] == 0 and matrix[i, 2] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 1] == 1 and matrix[i, 2] == 1 else
            np.random.rand() < p1 if matrix[i, 1] == 1 and matrix[i, 2] == 0 else
            np.random.rand() < p2 if matrix[i, 1] == 0 and matrix[i, 2] == 1 else
            np.random.rand() < p3,
        )
        matrix[i, 5], matrix[i, 6] = (
            np.random.rand() < p4 if matrix[i, 3] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 4] == 1 else np.random.rand() < p4,
        )
    return matrix


def model13(N, K, bounds):
    p0 = random.uniform(bounds.lob, bounds.upb)
    p1 = random.uniform(bounds.lob2, bounds.upb2)
    p2 = random.uniform(bounds.lob3, bounds.upb3)
    p3 = random.uniform(bounds.lob4, bounds.upb4)
    p4 = random.uniform(bounds.lob5, bounds.upb5)
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, 0] = np.random.rand() < 0.5
        matrix[i, 1], matrix[i, 2] = (
            np.random.rand() < p4 if matrix[i, 0] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 0] == 1 else np.random.rand() < p4,
        )
        matrix[i, 3], matrix[i, 4] = (
            np.random.rand() < p4 if matrix[i, 1] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 1] == 1 else np.random.rand() < p4,
        )
        matrix[i, 5], matrix[i, 6] = (
            np.random.rand() < p4 if matrix[i, 2] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 2] == 1 else np.random.rand() < p4,
        )
        matrix[i, 7], matrix[i, 8], matrix[i, 9] = (
            np.random.rand() < p0 if matrix[i, 3] == 1 and matrix[i, 4] == 1 else
            np.random.rand() < p1 if matrix[i, 3] == 1 and matrix[i, 4] == 0 else
            np.random.rand() < p2 if matrix[i, 3] == 0 and matrix[i, 4] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 3] == 1 and matrix[i, 4] == 1 else
            np.random.rand() < p1 if matrix[i, 3] == 1 and matrix[i, 4] == 0 else
            np.random.rand() < p2 if matrix[i, 3] == 0 and matrix[i, 4] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 3] == 1 and matrix[i, 4] == 1 else
            np.random.rand() < p1 if matrix[i, 3] == 1 and matrix[i, 4] == 0 else
            np.random.rand() < p2 if matrix[i, 3] == 0 and matrix[i, 4] == 1 else
            np.random.rand() < p3,
        )
        matrix[i, 10], matrix[i, 11], matrix[i, 12] = (
            np.random.rand() < p0 if matrix[i, 5] == 1 and matrix[i, 6] == 1 else
            np.random.rand() < p1 if matrix[i, 5] == 1 and matrix[i, 6] == 0 else
            np.random.rand() < p2 if matrix[i, 5] == 0 and matrix[i, 6] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 5] == 1 and matrix[i, 6] == 1 else
            np.random.rand() < p1 if matrix[i, 5] == 1 and matrix[i, 6] == 0 else
            np.random.rand() < p2 if matrix[i, 5] == 0 and matrix[i, 6] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 5] == 1 and matrix[i, 6] == 1 else
            np.random.rand() < p1 if matrix[i, 5] == 1 and matrix[i, 6] == 0 else
            np.random.rand() < p2 if matrix[i, 5] == 0 and matrix[i, 6] == 1 else
            np.random.rand() < p3,
        )
    return matrix


def model16(N, K, bounds):
    p0 = random.uniform(bounds.lob, bounds.upb)
    p1 = random.uniform(bounds.lob2, bounds.upb2)
    p2 = random.uniform(bounds.lob3, bounds.upb3)
    p3 = random.uniform(bounds.lob4, bounds.upb4)
    p4 = random.uniform(bounds.lob5, bounds.upb5)
    matrix = np.zeros((N, K))
    for i in range(N):
        matrix[i, 0] = np.random.rand() < 0.5
        matrix[i, 1], matrix[i, 2] = (
            np.random.rand() < p4 if matrix[i, 0] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 0] == 1 else np.random.rand() < p4,
        )
        matrix[i, 3] = (
            np.random.rand() < p0 if matrix[i, 1] == 1 and matrix[i, 2] == 1 else
            np.random.rand() < p1 if matrix[i, 1] == 1 and matrix[i, 2] == 0 else
            np.random.rand() < p2 if matrix[i, 1] == 0 and matrix[i, 2] == 1 else
            np.random.rand() < p3
        )
        matrix[i, 4], matrix[i, 5] = (
            np.random.rand() < p4 if matrix[i, 3] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 3] == 1 else np.random.rand() < p4,
        )
        matrix[i, 6], matrix[i, 7], matrix[i, 8], matrix[i, 9] = (
            np.random.rand() < p4 if matrix[i, 4] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 4] == 1 else np.random.rand() < p4,
            np.random.rand() < p4 if matrix[i, 5] == 1 else np.random.rand() < 1 - p4,
            np.random.rand() < 1 - p4 if matrix[i, 5] == 1 else np.random.rand() < p4,
        )
        matrix[i, 10], matrix[i, 11], matrix[i, 12] = (
            np.random.rand() < p0 if matrix[i, 6] == 1 and matrix[i, 7] == 1 else
            np.random.rand() < p1 if matrix[i, 6] == 1 and matrix[i, 7] == 0 else
            np.random.rand() < p2 if matrix[i, 6] == 0 and matrix[i, 7] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 6] == 1 and matrix[i, 7] == 1 else
            np.random.rand() < p1 if matrix[i, 6] == 1 and matrix[i, 7] == 0 else
            np.random.rand() < p2 if matrix[i, 6] == 0 and matrix[i, 7] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 6] == 1 and matrix[i, 7] == 1 else
            np.random.rand() < p1 if matrix[i, 6] == 1 and matrix[i, 7] == 0 else
            np.random.rand() < p2 if matrix[i, 6] == 0 and matrix[i, 7] == 1 else
            np.random.rand() < p3,
        )
        matrix[i, 13], matrix[i, 14], matrix[i, 15] = (
            np.random.rand() < p0 if matrix[i, 8] == 1 and matrix[i, 9] == 1 else
            np.random.rand() < p1 if matrix[i, 8] == 1 and matrix[i, 9] == 0 else
            np.random.rand() < p2 if matrix[i, 8] == 0 and matrix[i, 9] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 8] == 1 and matrix[i, 9] == 1 else
            np.random.rand() < p1 if matrix[i, 8] == 1 and matrix[i, 9] == 0 else
            np.random.rand() < p2 if matrix[i, 8] == 0 and matrix[i, 9] == 1 else
            np.random.rand() < p3,
            np.random.rand() < p0 if matrix[i, 8] == 1 and matrix[i, 9] == 1 else
            np.random.rand() < p1 if matrix[i, 8] == 1 and matrix[i, 9] == 0 else
            np.random.rand() < p2 if matrix[i, 8] == 0 and matrix[i, 9] == 1 else
            np.random.rand() < p3,
        )
    return matrix


def sample_latent_matrix(N, K, dag_type, bounds, alternate_prob=True):
    if dag_type == "Markov":
        probs = generate_prob_markov(K, bounds.lob, bounds.upb)
        return generate_markov_chain(N, K, probs)
    if dag_type == "Tree":
        return generate_tree(N, K, bounds.lob, bounds.upb, alternate_prob=alternate_prob)
    if dag_type == "DiverseTree":
        return diverse_tree(N, K, bounds)
    if dag_type == "Model-16":
        return model16(N, K, bounds)
    if dag_type == "Model-8":
        return model8(N, K, bounds)
    if dag_type == "Model-13":
        return model13(N, K, bounds)
    if dag_type == "Model-7":
        return model7(N, K, bounds)
    raise ValueError(f"Unknown DAG_type: {dag_type}")
