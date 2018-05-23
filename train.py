import mnist_loader
import numpy as np
from code import interact

class net:

    def __init__(self, input_size, output_size, hidden_units):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        # initialize parameters
        self.Wih = np.random.random_sample([input_size, hidden_units]) * 2 - 1
        self.Bih = np.zeros([1, hidden_units])
        self.Who = np.random.random_sample([hidden_units, output_size]) * 2 - 1
        self.Bho = np.zeros([1, output_shape[1]])
        self.input = None
        self.hidden_state = None
        self.batch_size = None

        self.d_Bho = None
        self.d_Who = None
        self.d_Bih = None
        self.d_Wih = None

    def forward_propagation(self, input):
        self.batch_size = input.shape[0]
        self.input = input
        hidden_activation = input.dot(self.Wih) + self.Bih
        self.hidden_state = sigmoid(hidden_activation)
        logits = self.hidden_state.dot(self.Who) + self.Bho
        output = softmax(logits)

        return output

    def backpropagate(self, output, target):
        # interact(local=locals())

        d_logits = target - output

        self.d_Bho = np.ones(self.batch_size).dot(d_logits)
        self.d_Who = self.hidden_state.transpose().dot(d_logits)

        d_hidden_state = d_logits.dot(self.Who.transpose())

        d_k = np.multiply(np.multiply(self.hidden_state, 1 - self.hidden_state), d_hidden_state)

        self.d_Bih = np.ones(self.batch_size).dot(d_k)
        self.d_Wih = self.input.transpose().dot(d_k)

    def apply_gradients(self, learning_rate):

        self.Wih += self.d_Wih * learning_rate
        self.Bih += self.d_Bih * learning_rate
        self.Who += self.d_Who * learning_rate
        self.Bho += self.d_Bho * learning_rate




def ReLU(activation):
    zeros = np.zeros(activation.shape)
    stacked = np.stack([activation, zeros], 2)
    return np.max(stacked, 2)


def sigmoid(activation):
    return 1 / (1 + np.exp(-activation))


def softmax(activation):
    max_value = np.max(activation, axis=1)
    max_value = np.repeat(np.expand_dims(max_value, 1), 10, axis=1)
    offset_values = activation - max_value
    exponentiated = np.exp(offset_values)
    return exponentiated/np.sum(exponentiated, axis=1)


def mean_squared_error(target, output):
    return np.linalg.norm(0.5 * (target - output))


def cross_entropy_error(target, output):
    tiny = 1e-40
    log_outputs = np.log(output + tiny)
    loss = target.transpose().dot(log_outputs)
    return loss





batch_size = 1

# load MNIST dataset
x_train, t_train, x_test, t_test = mnist_loader.load()
num_training_set = x_train.shape[0]
num_test_set = x_test.shape[0]
input_shape = x_train[0:batch_size, :].shape
output_shape = [1, 10]

input = x_test[0:1, :]/255
net = net(784, 10, 30)

onehot = np.eye(10)
targets = t_train[0:1]
target = []
for t in targets:
  target.append(onehot[t, :])

target = np.array(target)



for i in range(1000000):
    output = net.forward_propagation(input)
    loss = mean_squared_error(target, output)
    net.backpropagate(output, target)
    net.apply_gradients(0.0001)

    if i % 100 == 0:
        print(loss)


output = net.forward_propagation(input)
target = np.zeros(output_shape)
target[0, 2] = 1
net.backpropagate(output, target)



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




