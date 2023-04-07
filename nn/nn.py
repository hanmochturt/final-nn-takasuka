# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

# simpler example https://www.kdnuggets.com/2018/10/simple-neural-network-python.html

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
            self,
            nn_arch: List[Dict[str, Union[int, str]]],
            lr: float,
            seed: int,
            batch_size: int,
            epochs: int,
            loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
            self,
            W_curr: ArrayLike,
            b_curr: ArrayLike,
            A_prev: ArrayLike,
            activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        x = np.dot(W_curr, A_prev).shape
        Z_curr = np.dot(W_curr, A_prev) + b_curr
        if activation == 'linear':
            A_curr = Z_curr
        elif activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        elif activation == 'relu':
            A_curr = self._relu(Z_curr)
        else:
            raise ValueError("Invalid activation function")
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        cache = {}
        activation = X.T
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            activation, z = self._single_forward(self._param_dict[f'W{layer_idx}'],
                                                 self._param_dict[f'b{layer_idx}'],
                                                 activation, layer['activation'])
            cache[f"A{layer_idx}"] = activation
            cache[f"Z{layer_idx}"] = z
        return activation, cache

    def _single_backprop(
            self,
            W_curr: ArrayLike,
            b_curr: ArrayLike,
            Z_curr: ArrayLike,
            A_prev: ArrayLike,
            dA_curr: ArrayLike,
            activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike dC/dA[l]
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike dC/dA[l-1]
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike dC/dw[l]
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        dA_prev = dA_curr  # TODO

        if activation_curr == 'sigmoid':
            dc_dzl = self._sigmoid_backprop(dA_prev, Z_curr)
        elif activation_curr == 'relu':
            dc_dzl = self._relu_backprop(dA_prev, Z_curr)
        dc_dw = A_prev.T.dot(dc_dzl)  # dC/dw[l] = dC/dz[l] * dz[l]/dw[l]
        dW_curr = np.sum(dc_dw, axis=0)
        db_curr = np.sum(dc_dzl * 1, axis=0)  # dC/db[l] = dC/dz[l] * dz[l]/db[l]
        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        grad_dict = {}
        num_layers = int(len(cache) / 2)
        for layer in range(num_layers, 0, -1):
            y = cache[f"A{layer}"].T
            if self._loss_func == 'BCE':
                dc_da = self._binary_cross_entropy_backprop(y, y_hat)
            elif self._loss_func == 'MSE':
                dc_da = self._mean_squared_error_backprop(y, y_hat)
            else:
                raise ValueError("choose a valid error function")
            dC_dA, dC_dw, dC_db = self._single_backprop(self._param_dict[f"W{layer}"],
                                                        self._param_dict[f"b{layer}"],
                                                        cache[f"Z{layer}"], cache[f"A{layer}"],
                                                        dc_da,  # TODO cache A
                                                        # prev layer???
                                                        self.arch[layer - 1]['activation'])
            grad_dict[f"dC/dA{layer}"] = dC_dA
            grad_dict[f"dC/dw{layer}"] = dC_dw
            grad_dict[f"dC/db{layer}"] = dC_db
            y_hat = y_hat.dot(self._param_dict[f"W{layer}"])

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        for layer in range(int(len(grad_dict) / 3)):
            layer = layer + 1
            current_weights = self._param_dict[f"W{layer}"]
            current_b = self._param_dict[f"b{layer}"]
            adjustments_w = (current_weights.T + grad_dict[f"dC/dw{layer}"]*self._lr).T
            adjustments_b = (current_b.T + grad_dict[f"dC/db{layer}"]*self._lr).T
            self._param_dict[f"W{layer}"] = adjustments_w
            self._param_dict[f"b{layer}"] = adjustments_b

    def fit(
            self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        error_train = []
        error_test = []
        for pass_index in range(self._epochs):
            final_activation, cache = self.forward(X_train)
            y_calculated = self.predict(X_train)
            grad_dict = self.backprop(y_calculated, y_train, cache)
            self._update_params(grad_dict)
            y_calculated_test = self.predict(X_val)
            if self._loss_func == 'BCE':
                error_train_i = self._binary_cross_entropy(y_train, y_calculated)
                error_test_i = self._binary_cross_entropy(y_val, y_calculated_test)
            elif self._loss_func == 'MSE':
                error_train_i = self._mean_squared_error(y_train, y_calculated)
                error_test_i = self._mean_squared_error(y_val, y_calculated_test)
            else:
                raise ValueError("choose a valid error function")
            error_train.append(error_train_i)
            error_test.append(error_test_i)
        return error_train, error_test

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        layer_values = X.T
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            layer_values, linear_layer_values = self._single_forward(self._param_dict[f'W{layer_idx}'],
                                            self._param_dict[f'b{layer_idx}'],
                                            layer_values, layer['activation'])
        return layer_values.T

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        output_current_layer = self._sigmoid(Z)
        # https://towardsdatascience.com/understanding-the-derivative-of-the-sigmoid-function-cbfd46fb3716
        da_dz = output_current_layer * (1 - output_current_layer)
        dc_dz = da_dz.dot(dA)
        return dc_dz.T

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        nl_transform = Z
        nl_transform[nl_transform < 0] = 0
        return nl_transform

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """

        da_dz = Z
        da_dz[Z < 0] = 0
        da_dz[Z > 0] = 1
        dc_dz = da_dz.dot(dA)
        return dc_dz

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        y_hat = np.clip(y_hat, 0.000001, 0.999999)
        loss = y * np.log(y_hat) + ((1 - y) * np.log(1 - y_hat))
        average_loss = -np.sum(loss) / len(loss)
        return average_loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """

        dC_dZ = y_hat - y  # https://www.pinecone.io/learn/cross-entropy-loss/
        return dC_dZ

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        error = y - y_hat
        loss = np.sum(error ** 2) / len(error)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike dC/dA
                partial derivative of loss with respect to A matrix.
        """
        dC_dA = y_hat - y
        return dC_dA
