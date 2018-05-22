import mnist_loader
import numpy as np
from code import interact

def ReLU(activation):
  zeros = np.zeros(activation.shape)
  stacked = np.stack([activation, zeros], 2)
  return np.max(stacked, 2)



# parameters
H = 100
batch_size = 64
N = 2  # number of epochs



# load MNIST dataset
x_train, t_train, x_test, t_test = mnist_loader.load()
num_training_set = x_train.shape[0]
num_test_set = x_test.shape[0]
input_shape = x_train[0:batch_size, :].shape
output_shape = [1, 10]

# initialize parameters
w_i_h = np.random.random_sample([input_shape[1], H]) * 2 - 1
b_i_h = np.random.random_sample([1, H]) * 2 - 1
w_h_o = np.random.random_sample([w_i_h.shape[1], output_shape[1]]) * 2 - 1
b_h_o = np.random.random_sample([1, output_shape[1]]) * 2 - 1

input = x_test[0, :]/255
hidden_activation = input.dot(w_i_h) + b_i_h
hidden_state = ReLU(hidden_activation)
output = hidden_state.dot(w_h_o) + b_h_o



interact(local=locals())

# # loop over epochs
# for epoch in range(N):
#   num_minibatches = num_training_set // batch_size
#   # loop over minibatches
#   for i in range(num_minibatches):
#     input = None
#     target = None
#     # output = forward_propagation(input)
#     loss = loss(output, target)
#
#     pass




