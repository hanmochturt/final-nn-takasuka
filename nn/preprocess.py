# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random


def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    num_pos_seqs = sum(labels)
    num_neg_seqs = len(labels) - num_pos_seqs
    pos_seqs = seqs[:num_pos_seqs]
    neg_seqs = seqs[num_pos_seqs:]

    #adjust sequence lengths by subsampling
    len_neg_seq = len(neg_seqs[0])
    len_pos_seq = len(pos_seqs[0])
    if len_pos_seq != len_neg_seq:
        sub_sampled_neg_seqs = []
        if len_neg_seq > len_pos_seq:
            for neg_seq in neg_seqs:
                sub_sampled_start_index = random.randrange(len_neg_seq-len_pos_seq)
                sub_sampled_neg_seq = neg_seq[
                                      sub_sampled_start_index:sub_sampled_start_index+len_pos_seq]
                sub_sampled_neg_seqs.append(sub_sampled_neg_seq)
            seqs = pos_seqs + sub_sampled_neg_seqs

    # adjust sample sizes to be balanced
    if num_neg_seqs / num_pos_seqs > 2:
        multiple_to_add = int(num_neg_seqs / num_pos_seqs) - 1
        for i in range(multiple_to_add):
            seqs += pos_seqs
            labels += [True] * num_pos_seqs

    # filter out seqs with a different length
    normal_seq_length = len(seqs[0])
    seqs_filtered = []
    labels_filtered = []
    for index, seq in enumerate(seqs):
        if len(seq) == normal_seq_length:
            seqs_filtered.append(seq)
            labels_filtered.append(labels[index])
    return seqs_filtered, labels_filtered


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    seqs_arr_one_hot = []
    encoding_dict = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}
    for seq in seq_arr:
        seq_arr_one_hot = []
        for letter in seq:
            one_hot = encoding_dict[letter]
            seq_arr_one_hot += one_hot
        seqs_arr_one_hot.append(seq_arr_one_hot)
    return seqs_arr_one_hot


def reformat_pos_neg_seqs(pos_seqs: List, neg_seqs: List):
    """
    This function should sample the given sequences to account for class imbalance.
    Consider this a sampling scheme with replacement.

    Args:
        pos_seqs: positive sequences
        neg_seqs: negative sequences

    Returns:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels
    """
    positive_labels = [True]*len(pos_seqs)
    negative_labels = [False]*len(neg_seqs)
    return pos_seqs+neg_seqs, positive_labels+negative_labels
