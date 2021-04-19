import json
from pathlib import Path
from typing import Union, Any, List

import numpy as np
from scipy import sparse


def load_sparse(input_fname):
    return sparse.load_npz(input_fname).tocsr()

def load_json(fpath: Union[Path, str]) -> Any:
    with open(fpath) as infile:
        return json.load(infile)


def save_json(obj: Any, fpath: Union[Path, str]):
    with open(fpath, "w") as outfile:
        return json.dump(obj, outfile)


def save_topics(sorted_topics, vocab, fpath, n=100):
    """
    Save topics to disk
    """
    if not isinstance(vocab, np.ndarray):
        vocab = np.array(vocab)
    with open(fpath, "w") as outfile:
        for v in sorted_topics:
            topic = [vocab[i] for i in v]
            outfile.write(" ".join(topic) + "\n")

class NPMI:
    def __init__(self, bin_ref_counts: Union[np.ndarray, sparse.spmatrix]):
        assert bin_ref_counts.max() == 1
        self.bin_ref_counts = bin_ref_counts
        self.npmi_cache = {} # calculating NPMI is somewhat expensive, so we cache results

    def compute_npmi(self, beta: np.ndarray, n: int = 10) -> np.ndarray:
        """
        Compute NPMI for an estimated beta (topic-word distribution) parameter using
        binary co-occurence counts from a reference corpus
        """
        num_docs = self.bin_ref_counts.shape[0]
        sorted_topics = np.flip(beta.argsort(-1), -1)[:, :n]

        npmi_means = []
        for indices in sorted_topics:
            npmi_vals = []
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i+1:]:
                    ij = frozenset([idx_i, idx_j])
                    try:
                        npmi = self.npmi_cache[ij]
                    except KeyError:
                        col_i = self.bin_ref_counts[:, idx_i]
                        col_j = self.bin_ref_counts[:, idx_j]
                        c_i = col_i.sum()
                        c_j = col_j.sum()
                        if sparse.issparse(self.bin_ref_counts):
                            c_ij = col_i.multiply(col_j).sum()
                        else:
                            c_ij = (col_i * col_j).sum()
                        if c_ij == 0:
                            npmi = 0.0
                        else:
                            npmi = (
                                (np.log(num_docs) + np.log(c_ij) - np.log(c_i) - np.log(c_j)) 
                                / (np.log(num_docs) - np.log(c_ij))
                            )
                        self.npmi_cache[ij] = npmi
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))

        return np.array(npmi_means)


def compute_npmi(sorted_topics: np.ndarray, bin_ref_counts: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Compute NPMI for an estimated beta (topic-word distribution) parameter using
    binary co-occurence counts from a reference corpus
    """
    num_docs = bin_ref_counts.shape[0]

    sorted_topics = sorted_topics[:, :n]

    npmi_means = []
    for indices in sorted_topics:
        npmi_vals = []
        for i, index1 in enumerate(indices):
            for index2 in indices[i+1:n]:
                col1 = bin_ref_counts[:, index1]
                col2 = bin_ref_counts[:, index2]
                c1 = col1.sum()
                c2 = col2.sum()
                if sparse.issparse(bin_ref_counts):
                    c12 = col1.multiply(col2).sum()
                else:
                    c12 = (col1 * col2).sum()

                if c12 == 0:
                    npmi = 0.0
                else:
                    npmi = (
                        (np.log(num_docs) + np.log(c12) - np.log(c1) - np.log(c2)) 
                        / (np.log(num_docs) - np.log(c12))
                    )
                npmi_vals.append(npmi)
        npmi_means.append(np.mean(npmi_vals))

    return np.array(npmi_means)


def compute_tu(topics, n=10):
    """
    Topic uniqueness measure from https://www.aclweb.org/anthology/P19-1640.pdf
    """
    tu_results = []
    for topics_i in topics:
        w_counts = 0
        for w in topics_i[:n]:
            w_counts += 1 / np.sum([w in topics_j[:n] for topics_j in topics]) # count(k, l)
        tu_results.append((1 / n) * w_counts)
    return tu_results


def compute_tr(topics, n=10):
    """
    Compute topic redundancy score from 
    https://jmlr.csail.mit.edu/papers/volume20/18-569/18-569.pdf
    """
    tr_results = []
    k = len(topics)
    for i, topics_i in enumerate(topics):
        w_counts = 0
        for w in topics_i[:n]:
            w_counts += np.sum([w in topics_j[:n] for j, topics_j in enumerate(topics) if j != i]) # count(k, l)
        tr_results.append((1 / (k - 1)) * w_counts)
    return tr_results


def compute_topic_exclusivity(beta, n=20):
    """
    Compute topic exclusivity, cited in https://arxiv.org/pdf/2010.12626.pdf
    """
    raise NotImplementedError()


def compute_to(topics, n=10, multiplier=2, return_overlaps=False):
    """
    A sensible overlap / redundancy measure. Words from a topic
    are only counted once per "edge"

    Basic algorithm creates a de-duplicated adjacency matrix:
    for each topic A_i, sorted by total number of overlaps:
        create set of sets S = {S_{ij} = A_i \cap A_j st. j=i+1,...,k}
        sort sets in S by their cardinality in descending order
        initialize a set W = {}
        For each S_{ij}' in S:
            if words are not already part of an edge, i.e., |W \cap S_{ij}'| is 0:
               create an edge between A_i and A_j with weight w = |S_{ij}'|
               augment the list of words used in an edge, W = W \cup S_{ij}'
    
    Then, sum the number of edges, weighted by some function of that number
    """
    k = len(topics)
    overlap_counts = np.zeros((k, k), dtype=int)
    overlap_dedup = np.zeros((k, k), dtype=int)
    overlap_words = {}

    # first count all the overlaps between topics
    for i, topic_i in enumerate(topics):
        for j, topic_j in enumerate(topics[i+1:], start=i+1):
            words_ij = set(topic_i[:n]) & set(topic_j[:n])
            overlap_counts[[i, j], [j, i]] = len(words_ij)
            overlap_words[frozenset([i, j])] = words_ij

    # sort topics by those with most overlaps
    sort_idx = overlap_counts.sum(0).argsort()[::-1]
    overlap_counts = overlap_counts[sort_idx, :][:, sort_idx]
    for i, counts in enumerate(overlap_counts):
        counted_words = set()
        start = i + 1
        for j in (counts[start:].argsort()[::-1] + start):
            words_ij = overlap_words[frozenset([i, j])]
            if overlap_counts[i, j] > 0 and len(counted_words & words_ij) == 0:
                overlap_dedup[i, j] = overlap_counts[i, j]
                counted_words |= words_ij

    # how many 1-word n-topic overlaps are equivalent to an n-word 1-topic overlap?
    increments = np.linspace(1/multiplier, n, num=n)
    redundancy = increments[overlap_dedup[overlap_dedup > 0] - 1].sum() / (n * (k - 1))
    if return_overlaps:
        redundancy = redundancy, overlap_dedup
    return redundancy
