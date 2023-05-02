"""
Author: Marwan Alalloush
discription: This file contains the code for the DBN model.
"""
import pickle
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DBN(object):
    def __init__(self, original_input_size, input_size, output_size,
                 learning_rate, rbmOne, rbmTwo, rbmThree, rbmFour):
        # Define hyperparameters
        self._original_input_size = original_input_size
        self._input_size = input_size
        self._output_size = output_size
        self.learning_rate = learning_rate
        self.rbmOne = rbmOne
        self.rbmTwo = rbmTwo
        self.rbmThree = rbmThree
        self.rbmFour = rbmFour

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

        v0 = tf.compat.v1.placeholder("float", [None, self._original_input_size])
        forwardOne = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(v0, self.rbmOne.w) + self.rbmOne.hb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid(tf.matmul(v0, self.rbmOne.w) + self.rbmOne.hb)))))
        forwardTwo = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(forwardOne, self.rbmTwo.w) + self.rbmTwo.hb) - tf.compat.v1.random_uniform( tf.shape(tf.nn.sigmoid(tf.matmul(forwardOne, self.rbmTwo.w) + self.rbmTwo.hb)))))
        forwardThree = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(forwardTwo, self.rbmThree.w) + self.rbmThree.hb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid
                                                                                                                                                (tf.matmul( forwardTwo, self.rbmThree.w) 
                                                                                                                                                + self.rbmThree.hb)))))
        forward = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(forwardThree, self.rbmFour.w) + self.rbmFour.hb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid(tf.matmul(forwardThree, self.rbmFour.w) + self.rbmFour.hb)))))
        h0 = self.sample_prob(self.prob_h_given_v(forward, _w, _hb))
        v1 = self.sample_prob(self.prob_v_given_h(h0, _w, _vb))
        h1 = self.prob_h_given_v(v1, _w, _hb)

        positive_grad = tf.matmul(tf.transpose(forward), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)

        update_w = _w + self.learning_rate * (positive_grad - negative_grad) / tf.compat.v1.to_float(tf.shape(forward)[0])
        update_vb = _vb +  self.learning_rate * tf.reduce_mean(forward - v1, 0)
        update_hb = _hb +  self.learning_rate * tf.reduce_mean(h0 - h1, 0)

        backwardOne = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(v1, self.rbmFour.w.T) + self.rbmFour.vb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid( tf.matmul(v1, self.rbmFour.w.T) + self.rbmFour.vb)))))
        backwardTwo = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(backwardOne, self.rbmThree.w.T) + self.rbmThree.vb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid( tf.matmul(backwardOne, self.rbmThree.w.T) + self.rbmThree.vb)))))
        backwardThree = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(backwardTwo, self.rbmTwo.w.T) + self.rbmTwo.vb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid( tf.matmul(backwardTwo, self.rbmTwo.w.T) + self.rbmTwo.vb)))))
        backward = tf.nn.relu(tf.sign(tf.nn.sigmoid(tf.matmul(backwardThree, self.rbmOne.w.T) + self.rbmOne.vb) - tf.compat.v1.random_uniform(tf.shape(tf.nn.sigmoid( tf.matmul(backwardThree, self.rbmOne.w.T) + self.rbmOne.vb)))))

        err = tf.reduce_mean(tf.square(v0 - backward))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            cur_w = sess.run(update_w, feed_dict={v0: X, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
            cur_hb = sess.run(update_hb, feed_dict={v0: X, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
            cur_vb = sess.run(update_vb, feed_dict={v0: X, _w: prv_w, _hb: prv_hb, _vb: prv_vb})
            prv_w = cur_w
            prv_hb = cur_hb
            prv_vb = cur_vb
            error = sess.run(err, feed_dict={v0: X, _w: cur_w, _vb: cur_vb, _hb: cur_hb})
            self.w = prv_w
            self.hb = prv_hb
            self.vb = prv_vb
        
        return error

    def dbn_output(self, X):

        input_X = tf.constant(X)
        forwardOne = tf.nn.sigmoid(tf.matmul(input_X, self.rbmOne.w) + self.rbmOne.hb)
        forwardTwo = tf.nn.sigmoid(tf.matmul(forwardOne, self.rbmTwo.w) + self.rbmTwo.hb)
        forwardThree = tf.nn.sigmoid(tf.matmul(forwardTwo, self.rbmThree.w) + self.rbmThree.hb)
        forward = tf.nn.sigmoid(tf.matmul(forwardThree, self.rbmFour.w) + self.rbmFour.hb)

        _w = tf.constant(self.w, dtype=tf.float32)
        _hb = tf.constant(self.hb, dtype=tf.float32)
        _vb = tf.constant(self.vb, dtype=tf.float32)

        out = tf.nn.sigmoid(tf.matmul(forward, _w) + _hb)
        hiddenGen = self.sample_prob(self.prob_h_given_v(forward, _w, _hb))
        visibleGen = self.sample_prob(self.prob_v_given_h(hiddenGen, _w, _vb))

        backwardThree = tf.nn.sigmoid(tf.matmul(visibleGen, self.rbmFour.w.T) + self.rbmFour.vb)
        backwardTwo = tf.nn.sigmoid(tf.matmul(backwardThree, self.rbmThree.w.T) + self.rbmThree.vb)
        backwardOne = tf.nn.sigmoid(tf.matmul(backwardTwo, self.rbmTwo.w.T) +self.rbmTwo.vb)
        backward = tf.nn.sigmoid(tf.matmul(backwardOne, self.rbmOne.w.T) + self.rbmOne.vb)

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            return sess.run(out), sess.run(backward)
    
    def save_model(self, filename):
        model_data = {
            'w': self.w,
            'hb': self.hb,
            'vb': self.vb,
            'rbmOne': self.rbmOne,
            'rbmTwo': self.rbmTwo,
            'rbmThree': self.rbmThree,
            'rbmFour': self.rbmFour
        }

        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)
            
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        dbn_loaded = cls(
            original_input_size=model_data['rbmOne'].visible_size,
            input_size=model_data['rbmFour'].hidden_size,
            output_size=model_data['w'].shape[1],
            learning_rate=0,
            rbmOne=model_data['rbmOne'],
            rbmTwo=model_data['rbmTwo'],
            rbmThree=model_data['rbmThree'],
            rbmFour=model_data['rbmFour']
        )
        dbn_loaded.w = model_data['w']
        dbn_loaded.hb = model_data['hb']
        dbn_loaded.vb = model_data['vb']

        return dbn_loaded

