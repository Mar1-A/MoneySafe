import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec as Grid
from config import args
import utils

class RBM(object):

    def __init__(self, input_size, output_size, learning_rate, epochs, batchsize):
        # Define hyperparameters
        self._input_size = input_size
        self._output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batchsize = batchsize

        # Initialize weights and biases using zero matrices
        self.w = np.zeros([input_size, output_size], "float")
        self.hb = np.zeros([output_size], "float")
        self.vb = np.zeros([input_size], "float")

    def prob_h_given_v(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.compat.v1.random_uniform(tf.shape(probs))))

    def train(self, X):
        _w = tf.compat.v1.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.compat.v1.placeholder("float", [self._output_size])
        _vb = tf.compat.v1.placeholder("float", [self._input_size])

        prv_w = np.zeros([self._input_size, self._output_size], "float")
        prv_hb = np.zeros([self._output_size], "float")
        prv_vb = np.zeros([self._input_size], "float")

        cur_w = np.zeros([self._input_size, self._output_size], "float")
        cur_hb = np.zeros([self._output_size], "float")
        cur_vb = np.zeros([self._input_size], "float")

        v0 = tf.compat.v1.placeholder("float", [None, self._input_size])
        h0 = self.sample_prob(self.prob_h_given_v(v0, _w, _hb))

        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.compat.v1.to_float(tf.shape(v0)[0])
        update_vb = _vb +  self.learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_hb = _hb +  self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        err = tf.reduce_mean(tf.square(v0 - v1))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for epoch in range(self.epochs):
                for i in range(int(len(X) / self.batchsize)):
                    # print('--{}'.format(i))
                    batch_train_inputs = []
                    batch_train_labels = []
                    filenames = []
                    for file in X[i * self.batchsize:(i + 1) * self.batchsize]:
                        arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
                        filenames.append(file)
                        batch_train_inputs.append(arr[0])
                        batch_train_labels.append(arr[1])
                        # print(arr[0].shape)
                    batch = np.asarray(utils.pad_sequences(batch_train_inputs, args.input_max_len), dtype=np.float32)
                    batch = utils.concatenate_mfccs(batch, 5)
                    cur_w = sess.run(update_w, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_hb = sess.run(update_hb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    cur_vb = sess.run(update_vb, feed_dict={v0: batch, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                    prv_w = cur_w
                    prv_hb = cur_hb
                    prv_vb = cur_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
                print ('Epoch: %d' % epoch,'reconstruction error: %f' % error)
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb


    def rbm_output(self, X):

      input_X = tf.constant(X)
      _w = tf.constant(self.w)
      _hb = tf.constant(self.hb)
      _vb = tf.constant(self.vb)
      out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
      hiddenGen = self.sample_prob(self.prob_h_given_v(input_X, _w, _hb))
      visibleGen = self.sample_prob(self.prob_v_given_h(hiddenGen, _w, _vb))
      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          return sess.run(out), sess.run(visibleGen), sess.run(hiddenGen)

    def show_features(self, shape, suptitle, count=-1):
      maxw = np.amax(self.w.T)
      minw = np.amin(self.w.T)
      count = self._output_size if count == -1 or count > \
              self._output_size else count
      ncols = count if count < 14 else 14
      nrows = count//ncols
      nrows = nrows if nrows > 2 else 3
      fig = plt.figure(figsize=(ncols, nrows), dpi=100)
      grid = Grid(fig, rect=111, nrows_ncols=(nrows, ncols), axes_pad=0.01)

      for i, ax in enumerate(grid):
          x = self.w.T[i] if i<self._input_size else np.zeros(shape)
          x = (x.reshape(1, -1) - minw)/maxw
          ax.imshow(x.reshape(*shape), cmap=mpl.cm.Greys)
          ax.set_axis_off()

      fig.text(0.5,1, suptitle, fontsize=20, horizontalalignment='center')
      fig.tight_layout()
      plt.show()
      return