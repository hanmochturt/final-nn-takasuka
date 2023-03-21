import nn.nn as nn
import nn.io as io
import nn.preprocess as preprocess
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

rap_1_seqs = io.read_text_file("data/rap1-lieb-positives.txt")
negative_seqs = io.read_fasta_file("data/yeast-upstream-1k-negative.fa")
seqs, labels = preprocess.reformat_pos_neg_seqs(rap_1_seqs, negative_seqs)
balanced_seqs, balanced_labels = preprocess.sample_seqs(seqs, labels)
balanced_one_hot_seqs = preprocess.one_hot_encode_seqs(balanced_seqs)

one_layer = [{'input_dim': 3, 'output_dim': 2, 'activation': 'linear'}]
n = nn.NeuralNetwork(one_layer, 0.01, 4, 5, 4, "MAE")

x_train, x_test, y_train, y_test = train_test_split(np.array(balanced_one_hot_seqs), \
                                                             np.array(balanced_labels))

#print(x_train[0])
print(x_train.shape, "x train shape")
print(x_test.shape, "x test shape")
print(y_train.shape, "y train shape")
print(y_test.shape, "y test shape")

layers = [{'input_dim': 68, 'output_dim': 32, 'activation': 'relu'},
          {'input_dim': 32, 'output_dim': 1, 'activation': 'sigmoid'}]

final, cache = n.forward(x_train)
#grad_dict = n.backprop(final, y_train_one_hot, cache)
error_train, error_test = n.fit(x_train, np.reshape(y_train, (-1, 1)), x_test, np.reshape(y_test,
                                                                                         (-1, 1)))
plt.plot(error_train)
plt.figure()
plt.plot(error_test)
plt.show()

