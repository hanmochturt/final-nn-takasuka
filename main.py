from mlxtend.preprocessing import one_hot
import nn.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

digits_data = load_digits()
print(digits_data["data"].shape)

x_train, x_test, y_train, y_test = train_test_split(digits_data["data"], digits_data["target"])

print(x_train.shape, "x train shape")
print(x_test.shape, "x test shape")
print(y_train.shape, "y train shape")
print(y_test.shape, "y test shape")

layers = [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'},
          {'input_dim': 32, 'output_dim': 10, 'activation': 'sigmoid'}]

n = nn.NeuralNetwork(layers, 0.001, 4, 5, 15, "BCE") #BCE or MSE

'''x_1 = n._single_forward(initial_params['W1'], initial_params['b1'], train.T, 'linear')
print(x_1.shape)
x_2 = n._single_forward(initial_params['W2'], initial_params['b2'], x_1, 'linear')'''

y_train_one_hot = one_hot(y_train)
y_test_one_hot = one_hot(y_test)
final, cache = n.forward(x_train)
#grad_dict = n.backprop(final, y_train_one_hot, cache)
error_train, error_test = n.fit(x_train, y_train_one_hot, x_test, y_test_one_hot)
plt.plot(error_train)
plt.figure()
plt.plot(error_test)
plt.show()

