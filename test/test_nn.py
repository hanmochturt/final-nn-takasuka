from mlxtend.preprocessing import one_hot
import math
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sys
import pathlib

PARENT_PARENT_FOLDER = pathlib.Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PARENT_PARENT_FOLDER))
from nn import nn as nn
from nn import preprocess


def test_single_forward():
    """Test that the single forward applies weights and biases to by dot product to the input
    data"""
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
    """Test that the forward applies weights and biases to by dot product to the input
    data"""
    num_samples = 3
    seed = 4
    one_layer = [{'input_dim': num_samples, 'output_dim': 2, 'activation': 'linear'}]
    n = nn.NeuralNetwork(one_layer, 0.01, seed, 5, seed, "MAE")
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
    """Test that when backprop is applied, gradient matrices of the correct dimensions are
    produced"""
    digits_data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits_data["data"], digits_data["target"])
    layer_dims = [64, 16, 64]
    error_name = "MSE"  # BCE or MSE

    layers = [{'input_dim': 64, 'output_dim': layer_dims[0], 'activation': 'relu'},
              {'input_dim': layer_dims[0], 'output_dim': layer_dims[1],
               'activation': 'relu'},
              {'input_dim': layer_dims[1], 'output_dim': layer_dims[2],
               'activation': 'relu'},
              {'input_dim': layer_dims[2], 'output_dim': 10, 'activation': 'sigmoid'}]

    n = nn.NeuralNetwork(layers, 0.00001, 4, 5, 200, error_name)
    y_train_one_hot = one_hot(y_train)
    y_test_one_hot = one_hot(y_test)
    _ = n.fit(x_train, y_train_one_hot, x_test, y_test_one_hot)

    layer = len(layer_dims)-1
    dimensions = layer_dims[-1]
    num_samples = x_train.shape[0]
    dc_da = np.zeros((num_samples, dimensions))
    dC_dA, dC_dw, dC_db = n._single_backprop(n._param_dict[f"W{layer}"],
                                             n._param_dict[f"b{layer}"],
                                             np.zeros((dimensions, num_samples)),
                                             np.zeros((dimensions, num_samples)),
                                             dc_da,
                                             n.arch[layer - 1]['activation'])
    assert dC_dA.shape == (num_samples, dimensions)
    assert dC_dw.shape == (dimensions,)
    assert dC_db.shape == (dimensions,)


def test_predict():
    """Test that the predict function creates predictions in the same dimensions as the provided
    labels or y data"""
    digits_data = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(digits_data["data"], digits_data["target"])
    layer_dims = [64, 16, 64]
    error_name = "MSE"  # BCE or MSE

    layers = [{'input_dim': 64, 'output_dim': layer_dims[0], 'activation': 'relu'},
              {'input_dim': layer_dims[0], 'output_dim': layer_dims[1],
               'activation': 'relu'},
              {'input_dim': layer_dims[1], 'output_dim': layer_dims[2],
               'activation': 'relu'},
              {'input_dim': layer_dims[2], 'output_dim': 10, 'activation': 'sigmoid'}]

    n = nn.NeuralNetwork(layers, 0.00001, 4, 5, 200, error_name)
    y_train_one_hot = one_hot(y_train)
    y_test_one_hot = one_hot(y_test)
    _ = n.fit(x_train, y_train_one_hot, x_test, y_test_one_hot)

    y_test = n.predict(x_test)
    assert y_test.shape == y_test_one_hot.shape


def test_binary_cross_entropy():
    """Test that the calculation of BCE is performing correctly on a single dimension"""
    one_layer = [{'input_dim': 64, 'output_dim': 2, 'activation': 'relu'}]
    n = nn.NeuralNetwork(one_layer, 0.01, 4, 5, 4, "BCE")
    bce_error = n._binary_cross_entropy(np.array([0.5]), np.array([0.5]))
    true_error = -math.log(0.5)
    assert bce_error == true_error


def test_binary_cross_entropy_backprop():
    """Test that the calculation of BCE backprop is performing correctly on a single dimension"""
    one_layer = [{'input_dim': 64, 'output_dim': 2, 'activation': 'relu'}]
    n = nn.NeuralNetwork(one_layer, 0.01, 4, 5, 4, "BCE")
    backprop = n._binary_cross_entropy_backprop(np.array([0.5]), np.array([0.5]))
    assert backprop == np.array([0])


def test_mean_squared_error():
    """Test that the calculation of MSE is performing correctly on a single dimension"""
    one_layer = [{'input_dim': 64, 'output_dim': 2, 'activation': 'relu'}]
    n = nn.NeuralNetwork(one_layer, 0.01, 4, 5, 4, "BCE")
    calculated_error = n._mean_squared_error(np.array([0.5]), np.array([0.3]))
    true_error = (0.5-0.3)**2
    assert calculated_error == true_error


def test_mean_squared_error_backprop():
    """Test that the calculation of MSE backprop is performing correctly on a single dimension"""
    one_layer = [{'input_dim': 64, 'output_dim': 2, 'activation': 'relu'}]
    n = nn.NeuralNetwork(one_layer, 0.01, 4, 5, 4, "BCE")
    backprop = n._mean_squared_error_backprop(np.array([0.5]), np.array([0.5]))
    assert backprop == np.array([0])


def test_sample_seqs():
    """Test that subsampling of the sequences causes them to be the same length, keep the same
    labels, and be subsampled from the correct sequence"""
    original_seqs = ["AACATGCATGCATG", "ACTGTGCTAGCAAGCATACATGCATGCTCGATC"]
    balanced_seqs, labels = preprocess.sample_seqs(original_seqs, [0, 1])
    assert len(balanced_seqs[0]) == len(balanced_seqs[1])
    assert labels == [0, 1]
    assert balanced_seqs[1] in original_seqs[1]


def test_one_hot_encode_seqs():
    """Test a short sequence with a known one hot encoding output"""
    one_hot_seq = preprocess.one_hot_encode_seqs(['AGA'])
    assert one_hot_seq == [[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]]
