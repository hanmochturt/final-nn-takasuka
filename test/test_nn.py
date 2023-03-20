import numpy as np
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from nn import nn as nn
from nn import preprocess


def test_single_forward():
    num_samples = 3
    b_constant = 5
    one_layer = [{'input_dim': 64, 'output_dim': 2, 'activation': 'relu'}]
    n = nn.NeuralNetwork(one_layer, 0.01, 4, 5, 4, "MAE")
    param_dict = {}

    # Initialize each layer's weight matrices (W) and bias matrices (b)
    for layer in one_layer:
        input_dim = layer['input_dim']
        output_dim = layer['output_dim']
        param_dict['W1'] = np.ones((output_dim, input_dim))
        param_dict['b1'] = np.ones((output_dim, 1))*b_constant
        train = np.ones((num_samples, input_dim))

    _, x = n._single_forward(param_dict['W1'], param_dict['b1'], train.T, 'linear')
    expected = np.ones((output_dim, num_samples))*(input_dim+b_constant)
    assert np.allclose(expected, x)


def test_forward():
    num_samples = 3
    seed = 4
    one_layer = [{'input_dim': num_samples, 'output_dim': 2, 'activation': 'linear'}]
    n = nn.NeuralNetwork(one_layer, 0.01, seed, 5, 4, "MAE")
    param_dict = {}

    # Initialize each layer's weight matrices (W) and bias matrices (b)
    for layer in one_layer:
        input_dim = layer['input_dim']
        output_dim = layer['output_dim']
        np.random.seed(seed)
        param_dict['W1'] = np.random.randn(output_dim, input_dim) * 0.1
        param_dict['b1'] = np.random.randn(output_dim, 1) * 0.1
        train = np.ones((num_samples, input_dim))

    calculated, _ = n.forward(train)
    expected = np.dot(param_dict['W1'], train) + param_dict['b1']
    assert np.allclose(expected, calculated)

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    one_hot_seq = preprocess.one_hot_encode_seqs(['AGA'])
    assert one_hot_seq == [[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
