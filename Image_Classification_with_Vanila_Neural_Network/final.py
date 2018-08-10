import pandas as pd
import numpy as np
import argparse
import sys
import copy
import os
import pickle
import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser(description='Trains the FeedForward Neural Network')
parser.add_argument("--lr", type=float, help="initial learning rate for gradient descent based algorithms")

parser.add_argument("--momentum", type=float, help="momentum to be used by momentum based algorithms")

parser.add_argument("--num_hidden", type=int,
                    help="number of hidden layers - this does not include the 784 dimensional input_x layer\
                     and the 10 dimensional output layer")

parser.add_argument("--sizes", type=str, help="a comma separated list for the size of each hidden layer")

parser.add_argument("--activation", type=str,
                    help="the choice of activation function - valid values are hyperbolicTangent/sigmoid")

parser.add_argument("--loss", type=str,
                    help="possible choices are squared error[sq] or cross entropy loss[ce]")

parser.add_argument("--opt", type=str,
                    help="the optimization algorithm to be used:\
                     gd, momentum, nag, adam - you will be implementing \
                     the mini-batch version of these algorithms")

parser.add_argument("--batch_size", type=int,
                    help="the batch size to be used - valid values are 1 and multiples of 5")

parser.add_argument("--anneal", type=str,
                    help="if true the algorithm should halve the learning rate if at any epoch the \
                    validation loss decreases and then restart that epoch")

parser.add_argument("--save_dir", type=str,
                    help="the directory in which the pickled model should be saved - by model we mean\
                     all the weights and biases of the network")

parser.add_argument("--expt_dir", type=str,
                    help="the directory in which the log files will be saved - see below for a detailed\
                     description of which log files should be generated")

parser.add_argument("--train", type=str, help="path to the Training dataset")

parser.add_argument("--test", type=str, help="path to the Test dataset")

parser.add_argument("--val", type=str, help="path to the Validation dataset")

print("Parsing Arguments...")

args = parser.parse_args()
args.sizes = tuple([int(n) for n in args.sizes.split(',')])

if len(args.sizes) != args.num_hidden:
    if len(args.sizes) > args.num_hidden:
        print("Error: Comma separated list for Sizes of hidden layers has unnecessary more values.")
        sys.exit()
    else:
        print("Error: Comma separated list for Sizes of hidden layers has less number of values.")
        sys.exit()

if args.activation == "tanh" or args.activation == "sigmoid":
    pass
else:
    print("Error: Unidentified activation function.")
    sys.exit()

if args.loss == "sq" or args.loss == "ce":
    pass
else:
    print("Error: Unidentified Loss Metric.")
    sys.exit()

if args.opt == "gd" or args.opt == "momentum" or args.opt == "nag" or args.opt == "adam":
    pass
else:
    print("Error: Unidentified Optimization Algorithm")
    sys.exit()

if args.batch_size == 1 or args.batch_size % 5 == 0:
    pass
else:
    print("Error: Batch size should be 1 or a multiple of 5")
    sys.exit()

if args.anneal == "true" or args.anneal == "false":
    pass
else:
    print("Error: Unidentified value of Anneal parameter.")
    sys.exit()

# Load Data
print("Loading Data...")
train = pd.read_csv(args.train)
test = pd.read_csv(args.test)
val = pd.read_csv(args.val)
train_x = np.array(train.drop(columns=["id", "label"], axis=1))
train_y = np.array(train["label"]).reshape(55000, 1)
val_x = np.array(val.drop(columns=["id", "label"], axis=1))
val_y = np.array(val["label"]).reshape(5000, 1)
test_x = np.array(test.drop(columns=["id"], axis=1))

print("Preparing Data... ")
# Convert to One Hot Encoding
train_y_target = train_y.reshape(-1)
train_y_onehot = np.eye(10)[train_y_target]
val_y_target = val_y.reshape(-1)
val_y_onehot = np.eye(10)[val_y_target]


# Normalizing data
def normalize(x):
    a = 0
    b = 1
    x_max = 255
    x_min = np.amin(x)
    return ((x - x_min) * (b - a)) / (x_max - x_min)


train_x, val_x, test_x = normalize(train_x), normalize(val_x), normalize(test_x)
n_x = 784
n_y = 10


def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * \
                                   np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = W.dot(A) + b
    cache = (A, W, b)

    return Z, cache


def softmax(Z):
    e_x = np.exp(Z - np.max(Z))
    A = e_x / e_x.sum(axis=0)
    cache = Z
    return A, cache


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh_backward(dA, cache):
    Z = cache
    dZ = 1 - np.square(np.tanh(Z))
    assert (dZ.shape == Z.shape)
    return dZ


def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == "tanh":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = tanh(Z)
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, activation_back=args.activation):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                             activation=activation_back)
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)],
                                          activation="softmax")
    caches.append(cache)

    return AL, caches


def compute_loss(AL, Y):
    if args.loss == "ce":
        m = Y.shape[1]
        loss = -1 * np.sum((Y * np.log(AL)))
        loss = np.squeeze(loss)
        return loss
    else:
        # Squared Error loss
        return ((AL - Y) ** 2).mean()


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_backward_output(dZ, cache):
    A_prev, W, b = cache[0]
    m = A_prev.shape[1]

    dW = 1. / m * np.dot(dZ, A_prev.T)
    db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "tanh":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dA_prev, dW, db = linear_backward_output(dA, cache)

    return dA_prev, dW, db


def L_model_backward(AL, Y, caches, activation_back):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    dAL = - (Y - AL)

    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
        linear_activation_backward(dAL, current_cache, activation='softmax')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
            linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation=activation_back)

        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads




def nag_update_parameters(parameters, grads, m, gamma, learning_rate, AL, Y_batch, caches, activation_back):
    L = len(parameters) // 2
    parameters_PV = copy.deepcopy(parameters)

    for l in range(L):

        m["dW" + str(l + 1)] = gamma * m["dW" + str(l + 1)]
        m["db" + str(l + 1)] = gamma * m["db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - m["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - m["db" + str(l + 1)]

        grads = L_model_backward(AL, Y_batch, caches, activation_back)

        for t in range(L):
            m["dW" + str(t + 1)] = gamma * m["dW" + str(t + 1)] + learning_rate * grads["dW" + str(t + 1)]
            m["db" + str(t + 1)] = gamma * m["db" + str(t + 1)] + learning_rate * grads["db" + str(t + 1)]
            parameters["W" + str(t + 1)] = parameters_PV["W" + str(t + 1)] - m["dW" + str(t + 1)]
            parameters["b" + str(t + 1)] = parameters_PV["b" + str(t + 1)] - m["db" + str(t + 1)]
            parameters_PV["W" + str(t + 1)] = copy.deepcopy(parameters["W" + str(t + 1)])
            parameters_PV["W" + str(t + 1)] = copy.deepcopy(parameters["W" + str(t + 1)])
    return parameters, m


def compute_ce_loss(X, Y_onehot, parameters):
    AL, caches = L_model_forward(X, parameters)
    loss = compute_loss(AL, Y_onehot)
    return loss


def compute_sq_loss(X, Y_onehot, parameters):
    AL, caches = L_model_forward(X, parameters)
    loss = compute_loss(AL, Y_onehot)
    return loss


def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.array(np.zeros(shape=parameters["W" + str(l + 1)].shape))
        v["db" + str(l + 1)] = np.array(np.zeros(shape=parameters["b" + str(l + 1)].shape))

    return v


def momentum_update(parameters, grads, m, gamma, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        m["dW" + str(l + 1)] = gamma * m["dW" + str(l + 1)] + learning_rate * grads["dW" + str(l + 1)]
        m["db" + str(l + 1)] = gamma * m["db" + str(l + 1)] + learning_rate * grads["db" + str(l + 1)]

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - m["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - m["db" + str(l + 1)]

    return parameters, m


def nag_update(parameters, grads, m, gamma, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        m["dW" + str(l + 1)] = gamma * m["dW" + str(l + 1)] + (1 - gamma) * grads["dW" + str(l + 1)]
        m["db" + str(l + 1)] = gamma * m["db" + str(l + 1)] + (1 - gamma) * grads["db" + str(l + 1)]

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * m["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * m["db" + str(l + 1)]

    return parameters, v


def initialize_adam(parameters):
    L = len(parameters) // 2
    m = {}
    v = {}

    for l in range(L):
        m["dW" + str(l + 1)] = np.zeros(shape=parameters["W" + str(l + 1)].shape)
        m["db" + str(l + 1)] = np.zeros(shape=parameters["b" + str(l + 1)].shape)
        v["dW" + str(l + 1)] = np.zeros(shape=parameters["W" + str(l + 1)].shape)
        v["db" + str(l + 1)] = np.zeros(shape=parameters["b" + str(l + 1)].shape)

    return m, v


def adam_update(parameters, grads, m, v, t, learning_rate=args.lr,
                beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    m_corrected = {}
    v_corrected = {}

    for l in range(L):
        m["dW" + str(l + 1)] = beta1 * m["dW" + str(l + 1)] + (1 - beta1) * grads["dW" + str(l + 1)]
        m["db" + str(l + 1)] = beta1 * m["db" + str(l + 1)] + (1 - beta1) * grads["db" + str(l + 1)]

        m_corrected["dW" + str(l + 1)] = m["dW" + str(l + 1)] / (1 - beta1 ** t)
        m_corrected["db" + str(l + 1)] = m["db" + str(l + 1)] / (1 - beta1 ** t)

        v["dW" + str(l + 1)] = beta2 * v["dW" + str(l + 1)] + (1 - beta2) * np.power(grads["dW" + str(l + 1)], 2)
        v["db" + str(l + 1)] = beta2 * v["db" + str(l + 1)] + (1 - beta2) * np.power(grads["db" + str(l + 1)], 2)

        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - beta2 ** t)
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - beta2 ** t)

        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * m_corrected[
            "dW" + str(l + 1)] / (np.sqrt(v_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * m_corrected[
            "db" + str(l + 1)] / (np.sqrt(v_corrected["db" + str(l + 1)] + epsilon))

    return parameters, m, v


def gd_update(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def compute_error(AL, Y_Batch):
    y_corr = 1 * (np.multiply(AL, Y_Batch) >= 0.5)
    accu = 100 * np.sum(y_corr) / Y_Batch.shape[1]
    return 100 - accu


def predict(X, y, parameters, loss_type):
    m = X.shape[1]
    probas, caches = L_model_forward(X, parameters)
    p = 1 * (probas >= 0.5)
    y_predict = p.argmax(axis=0).reshape(1, m).T
    percentage_loss = 100 - np.sum(1 * np.equal(y_predict, y)) * 100 / m

    print(loss_type + " Loss: " + str(percentage_loss) + "%")

    return percentage_loss

def ffnetwork(X, Y, train_y, val_X, val_Y, layers_dims, num_iterations, print_cost=False,
              batch_size=args.batch_size):
    learning_rate = args.lr
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)
    activation_back = args.activation

    if args.opt == "momentum":
        m = initialize_velocity(parameters)

    m, v = initialize_adam(parameters)
    gamma = 0.9

    log_file_path = args.expt_dir + "log_train.txt"
    log_file_writer = open(log_file_path, 'w+')

    total_count = 0
    epoch_losses = []
    epoch_errors = []
    train_val_losses = []
    valdata_val_losses = []
    AL = []
    caches = []
    grads = []
    valdata_val_loss = -1

    i = 0
    while i < num_iterations:
        print("Running Epoch", i)
        # prev_parameters, prev_v, prev_s = copy.deepcopy(parameters), copy.deepcopy(m), copy.deepcopy(v)
        # step = 1
        # batch_errors = []
        # batch_losses = []
        # t = 0
        # for j in range(X.shape[0] // batch_size):
        #     # print("\tRunning Batch", j)

        #     X_batch, Y_batch = X[j * batch_size:(j + 1) * batch_size].T, Y[j * batch_size:(j + 1) * batch_size].T

        #     prev_AL = AL
        #     prev_caches = caches
        #     AL, caches = L_model_forward(X_batch, parameters)

        #     loss = compute_loss(AL, Y_batch)
        #     error = compute_error(AL, Y_batch)

        #     batch_errors.append(loss)
        #     batch_losses.append(error)

        #     prev_grads = grads
        #     grads = L_model_backward(AL, Y_batch, caches, activation_back)

        #     prev_parameters = parameters
        #     if args.opt == "gd":
        #         parameters = gd_update(parameters, grads, learning_rate)
        #     elif args.opt == "momentum":
        #         parameters,m = momentum_update(parameters, grads, m, gamma, learning_rate)
        #     elif args.opt == "nag":
        #         parameters, m = nag_update_parameters(parameters, grads, m, gamma, learning_rate, AL, \
        #                                               Y_batch, caches, activation_back)
        #     else:
        #         t += 1
        #         parameters,m, v = adam_update(parameters, grads,m, v, t, learning_rate, beta1, beta2,
        #                                        epsilon)
        #     if step % 100 == 0:
        #         log_file_writer.write(
        #             "Epoch: {}, Step: {}, Loss: {}, Error: {}, lr: {}\n".format(i, step, round(loss, 2),
        #                                                                         round(error, 2), args.lr))

        #     step = step + 1
        #     total_count += batch_size

        # epoch_losses.append(np.mean(batch_losses))
        # epoch_errors.append(np.mean(batch_errors))
        # # train_val_loss = predict(X.T, train_y, parameters, "Training")

        # prev_valdata_loss = valdata_val_loss
        # # valdata_val_loss = predict(val_X.T, val_Y, parameters, "Validation")

        # if args.loss == "ce":
        #     train_val_loss = compute_ce_loss(X.T, Y.T, parameters)
        #     valdata_val_loss = compute_ce_loss(val_x.T, val_y_onehot.T, parameters)
        # else:
        #     train_val_loss = compute_sq_loss(X.T, Y.T, parameters)
        #     valdata_val_loss = compute_sq_loss(val_x.T, val_y_onehot.T, parameters)

        # train_val_losses.append(train_val_loss)
        # valdata_val_losses.append(valdata_val_loss)

        # if i > 2 and args.anneal == "true" and prev_valdata_loss < valdata_val_loss:
        #     args.lr = args.lr / 2.0
        #     parameters, v, s = copy.deepcopy(prev_parameters), copy.deepcopy(prev_v), copy.deepcopy(prev_s)
        #     if args.opt == "adams":
        #         t = t - (X.shape[0] // batch_size)
        #     print("Annealing changed learning rate from %f to %f" % (2 * args.lr, args.lr))
        #     continue
        # else:
        #     train_val_losses.append(train_val_loss)
        #     valdata_val_losses.append(valdata_val_loss)

        i = i + 1

    log_file_writer.close()
    return parameters, train_val_losses, valdata_val_losses


def output(X, parameters, ve_no):
    m = X.shape[1]
    probas, caches = L_model_forward(X, parameters)
    p = 1 * (probas >= 0.5)
    y_predict = p.argmax(axis=0).reshape(1, m).T

    A = np.arange(m).reshape(m, 1)
    tag = ['id', 'label']
    C = np.concatenate((A, y_predict), axis=1)
    p = pd.DataFrame(C, columns=tag)
    p.head()
    p.to_csv('test_submission_v1.' + str(ve_no) + '.csv', sep=',', encoding='utf-8', index=False)
    print("Success")


layers_dims = [n_x]
args.sizes = list(args.sizes)
for i in args.sizes:
    layers_dims.append(i)
layers_dims.append(n_y)
layers_dims = tuple(layers_dims)

parameters, train_val_losses, valdata_val_losses = ffnetwork(train_x, train_y_onehot, train_y, val_x, val_y, layers_dims, 2, print_cost=True)
# pred_test = predict(val_x.T, val_y, parameters, loss_type="Validation")
output(test_x.T, parameters, 2)

file_writer = open(args.save_dir + "opt_losses.txt", "a")
file_writer.write("================================= Summary =================================\
\nOPT = {}\nTrain Loss = {}\nValidation Loss = {}\n========\
===================================================================\n".\
                  format(args.opt, train_val_losses, valdata_val_losses))
file_writer.close()

# plt.plot(np.squeeze(train_val_losses))
# plt.ylabel('Loss')
# plt.xlabel('Iterations')
# plt.title("Train Losses @lr = " + str(args.lr))
# plt.show()
#
# plt.plot(np.squeeze(valdata_val_losses))
# plt.ylabel('Loss')
# plt.xlabel('Iterations')
# plt.title("Validation Losses @lr = " + str(args.lr))
# plt.show()
#
# with open('variables_v1.2.pickle', 'rb') as f:
#     params = pickle.load(f)
