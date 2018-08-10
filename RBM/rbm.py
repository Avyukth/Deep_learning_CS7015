import os
import math
import time
import math
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

from ggplot import *
from pdb import set_trace as bp
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import ImageGrid

parser = argparse.ArgumentParser(description='Restricted Boltzman Machine')
parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate for gradient descent based algorithms")

parser.add_argument("--k", type=int, default=1, help="momentum to be used by momentum based algorithms")


parser.add_argument("--num_hidden", type=int, default=500,
                    help="number of hidden variable - this does not include the 784 dimensional input_x layer\
                     and the 10 dimensional output layer")

parser.add_argument("--batch_size", type=int, default=10,
                    help="the batch size to be used - valid values are 1 and multiples of 5")

parser.add_argument("--epochs", type=int, default=10, help="the no of epochs the model should run")


parser.add_argument("--save_dir", default=os.getcwd() + "/", type=str,
                    help="the directory in which the pickled model should be saved - by model we mean\
                     all the weights and biases of the network")

parser.add_argument("--train", default=os.getcwd() + "/train.csv", type=str, help="path to the Training dataset")

parser.add_argument("--test", default=os.getcwd() + "/test.csv", type=str, help="path to the Test dataset")

print("Parsing Arguments...")

args = parser.parse_args()


def make_matgrid(input_image, file_name):

    fig = plt.figure(1, (20., 20.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(8, 8),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for i in range(input_image.shape[0]):
        im = input_image[i:i + 1].reshape(28, 28)
        grid[i].imshow(im, cmap='gray')  # The AxesGrid object work as a list of axes.

    plt.savefig(file_name)
    plt.close('all')


def tsne_custom(x, y, filename):
    feat_cols = ['pixel' + str(i) for i in range(1, 785)]
    df = pd.DataFrame(x, columns=feat_cols)
    df['label'] = y
    df['label'] = df['label'].apply(lambda i: str(i))

    rndperm = np.random.permutation(df.shape[0])
    n_sne = 10000

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne], feat_cols].values)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    df_tsne = df.loc[rndperm[:n_sne], :].copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]

    chart = ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label')) \
        + geom_point(size=70, alpha=0.5) \
        + ggtitle("tSNE dimensions colored by digit")
    chart.save(filename + ".png")
    plt.close('all')


class RBM(object):
    """docstring for RBM"""

    def __init__(self, input=None, n_V=784, n_H=args.num_hidden, W=None, h_bias=None, v_bias=None, numpy_rng=None,):
        super(RBM, self).__init__()
        self.input = input
        self.visible = n_V
        self.hidden = n_H
        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1)
        if W is None:
            W = numpy_rng.uniform(
                low=-4 * np.sqrt(6. / (n_H + n_V)),
                high=4 * np.sqrt(6. / (n_H + n_V)),
                size=(n_V, n_H)
            )
        if h_bias is None:
            h_bias = np.zeros(n_H)
        if v_bias is None:
            v_bias = np.zeros(n_V)
        self.W = W
        self.numpy_rng = numpy_rng
        self.hbias = h_bias
        self.vbias = v_bias
        self.params = [self.W, self.hbias, self.vbias]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1.0 - y)

    def propagate_up(self, visible):

        pre_sigmoid = np.dot(visible, self.W) + self.hbias
        post_sigmoid = self.sigmoid(pre_sigmoid)
        return [pre_sigmoid, post_sigmoid]

    def h_given_v(self, v0):
        pre_sigmoid_h1, h1_mean = self.propagate_up(v0)
        h1_sample = self.numpy_rng.binomial(n=1, p=h1_mean, size=h1_mean.shape)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propagate_down(self, hidden):
        pre_sigmoid = np.dot(hidden, self.W.T) + self.vbias
        post_sigmoid = self.sigmoid(pre_sigmoid)
        return [pre_sigmoid, post_sigmoid]

    def v_given_h(self, h0):
        pre_sigmoid_v1, v1_mean = self.propagate_down(h0)
        v1_sample = np.random.binomial(n=1, p=v1_mean, size=v1_mean.shape)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def giibs_hvh(self, h0):
        pre_sigmoid_v1, v1_mean, v1_sample = self.v_given_h(h0)
        pre_sigmoid_h1, h1_mean, h1_sample = self.h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def giibs_vhv(self, v0):
        pre_sigmoid_h1, h1_mean, h1_sample = self.h_given_v(v0)
        pre_sigmoid_v1, v1_mean, v1_sample = self.v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def free_energy_loss(self, epsilon=10**(-3)):
        pre_sigmoid_activation_h = np.dot(self.input, self.W) + self.hbias
        sigmoid_activation_h = self.sigmoid(pre_sigmoid_activation_h)

        pre_sigmoid_activation_v = np.dot(sigmoid_activation_h, self.W.T) + self.vbias
        sigmoid_activation_v = self.sigmoid(pre_sigmoid_activation_v)

        cross_entropy = - np.mean(
            np.sum(self.input * np.log(sigmoid_activation_v + epsilon) +
                   (1 - self.input) * np.log(1 + epsilon - sigmoid_activation_v),
                   axis=1))

        return cross_entropy

    def get_update_rbm(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.h_given_v(self.input)

        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        for step in range(k):
            if step == 0:
                pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples = self.giibs_hvh(chain_start)
            else:
                pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples = self.giibs_hvh(nh_samples)

        self.W += lr * (np.dot(self.input.T, ph_mean) - np.dot(nv_samples.T, nh_means))
        self.hbias += lr * np.mean(nh_samples - nh_means, axis=0)
        self.vbias += lr * np.mean(self.input - nv_samples, axis=0)
        chain_end = nv_samples
        cost = self.free_energy_loss()
        return cost, chain_end

    def reconstruct(self, v):

        h = self.sigmoid(np.dot(v, self.W) + self.hbias)
        reconstructed_v = self.sigmoid(np.dot(h, self.W.T) + self.vbias)
        return reconstructed_v


def test_rbm(data, testdata, test_y, learning_rate=args.lr, k=1, training_epochs=1):
    data = data
    rng = np.random.RandomState(123)
    fileout = open(args.save_dir + str(k) + "_out.txt", "a")
    # construct RBM

    # train
    epoch_cost_list = []
    rbm = RBM(input=data)
    step = 0
    storedimage = data[0]
    for epoch in range(training_epochs):
        batch_cost_list = []

        for batch in random_mini_batches(data, mini_batch_size=args.batch_size, seed=0):
            rbm.input = batch
            cost, chain_end = rbm.get_update_rbm(lr=learning_rate, k=k)
            batch_cost_list.append(cost)
            if step % math.floor(args.epochs * (60000 / args.batch_size) / 64) == 0:
                storedimage = np.vstack((storedimage, chain_end[0]))
                # bp()
                if storedimage.shape[0] == 65:
                    make_matgrid(storedimage[1:, :], args.save_dir + str(k) + "hidden" + str(step))

            step += 1
            print("step: {}, stacked length {}".format(step, storedimage.shape))
        batch_cost = (np.mean(np.array(batch_cost_list)))
        print('Training epoch {0}, cost is {1}' .format(epoch, batch_cost), file=fileout)
        print('Training epoch {0}, cost is {1}' .format(epoch, batch_cost))
        epoch_cost_list.append(batch_cost)
    print("{0}".format(epoch_cost_list), file=fileout)
    t = rbm.reconstruct(testdata)
    tsne_custom(t, test_y, args.save_dir + str(k) + "tsne")
    i = 0
    fileout.close()
    for x in random_mini_batches(t, mini_batch_size=64, seed=0):
        make_matgrid(x, args.save_dir + str(k) + "recon" + str(i))
        i += 1
        if i == 5:
            break


def random_mini_batches(X, mini_batch_size=64, seed=0):

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size]
        mini_batch = (mini_batch_X)
        mini_batches.append(mini_batch)

    return mini_batches


print("Loading Data...")
train = pd.read_csv(args.train)
test = pd.read_csv(args.test)

train_x = np.array(train.drop(columns=["label"], axis=1))
train_x = (train_x > 126) * 1
test_x = np.array(test.drop(columns=["label"], axis=1))
test_x = (test_x > 126) * 1
test_y = test["label"]

test_rbm(train_x, testdata=test_x, test_y=test_y, k=args.k, training_epochs=args.epochs)
