"""
Author: Marwan Alalloush
discription: This file contains the code for the RBM layers.
"""
import pickle
import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class RBM(object):

    def __init__(self, input_size, output_size, learning_rate):
        # Define hyperparameters
        self._input_size = input_size
        self._output_size = output_size
        self.learning_rate = learning_rate

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
    
    def save_rbm(self, filename):
        rbm_data = {
            'w': self.w,
            'hb': self.hb,
            'vb': self.vb,
            'visible_size': self._input_size,
            'hidden_size': self._output_size
        }

        with open(filename, 'wb') as file:
            pickle.dump(rbm_data, file)
    
    @classmethod
    def load_rbm(cls, filename):
        with open(filename, 'rb') as file:
            rbm_data = pickle.load(file)

        rbm_loaded = cls(
            visible_size=rbm_data['visible_size'],
            hidden_size=rbm_data['hidden_size'],
            learning_rate=0,  # Set to 0 since you are loading a trained RBM
        )
        rbm_loaded.w = rbm_data['w']
        rbm_loaded.hb = rbm_data['hb']
        rbm_loaded.vb = rbm_data['vb']

        return rbm_loaded


    def train(self, X):
        _w = tf.compat.v1.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.compat.v1.placeholder("float", [self._output_size])
        _vb = tf.compat.v1.placeholder("float", [self._input_size])

        prv_w = self.w
        prv_hb = self.hb
        prv_vb = self.vb

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
        with tf.device("/GPU:0"):
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())

                cur_w = sess.run(update_w, feed_dict={v0: X, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                cur_hb = sess.run(update_hb, feed_dict={v0: X, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
                cur_vb = sess.run(update_vb, feed_dict={v0: X, _w: prv_w, _hb: prv_hb, _vb: prv_vb})

                prv_w = cur_w
                prv_hb = cur_hb
                prv_vb = cur_vb

                self.w = prv_w
                self.hb = prv_hb
                self.vb = prv_vb
                error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})

        return error

    def rbm_output(self, X):
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        _vb = tf.constant(self.vb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        hiddenGen = self.sample_prob(self.prob_h_given_v(input_X, _w, _hb))
        visibleGen = self.sample_prob(self.prob_v_given_h(hiddenGen, _w, _vb))
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            return sess.run(out), sess.run(visibleGen), sess.run(hiddenGen)

        
    