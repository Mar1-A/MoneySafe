"""
Author: Marwan Alalloush
description: This file contains the code for the LSTM model.
"""

import tensorflow as tf
import pickle
import os
from tensorflow.python.ops import array_ops
from tensorflow.keras import regularizers
# from config import args
import utils
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)


class model_modes:
    
    TRAIN = 1
    EVALUATE = 2
    INFER = 3


class SpeechRecognitionModel(object):
    def __init__(self, mode, args, embedding_matrix=None):
      
        self.mode = mode
        self.args = args

        # Define the batch size
        batch_size = args.batch_size
        
        # If the system is in inference (actual recognition), change the batch size to 1 so it can recognize immediately
        if self.mode == model_modes.INFER:
            args.batch_size = 1
        
        self.inputs = tf.compat.v1.placeholder(tf.float32, [None, None, args.num_features], name='inputs')
        self.targets = tf.compat.v1.sparse_placeholder(tf.int32, name = 'targets')

        with tf.compat.v1.variable_scope("Compute_Sequence_Lengths"):
            # Computing the sequence lengths for all examples in the batch.
            seq_lens = utils.compute_seq_lens(self.inputs)

        with tf.compat.v1.variable_scope("LSTM"):
            # Create LSTM cells with dropout
            lstm_cells = [tf.keras.layers.LSTMCell(args.num_hidden, dropout=args.dropout_rate, recurrent_dropout=args.dropout_rate) for _ in range(args.num_layers)]

            # Stacking rnn cells
            stack_lstm = tf.keras.layers.StackedRNNCells(lstm_cells)

            # Use dynamic_rnn instead of bidirectional_dynamic_rnn
            self.outputs, _ = tf.compat.v1.nn.dynamic_rnn(stack_lstm, self.inputs, seq_lens, dtype=tf.float32)

            shape = tf.shape(self.inputs)
            batch_s, max_timesteps = shape[0], shape[1]

            self.outputs = tf.reshape(self.outputs, [-1, args.num_hidden * 2])
            W = tf.Variable(tf.random.truncated_normal([args.num_hidden * 2, args.num_classes], stddev=0.1))
            b = tf.Variable(tf.constant(0., shape=[args.num_classes]))
            self.logits = tf.matmul(self.outputs, W) + b
            self.logits = tf.reshape(self.logits, [batch_s, -1, args.num_classes])
            self.logits = tf.transpose(self.logits, (1, 0, 2))

        if self.mode == model_modes.TRAIN:
            self.loss = tf.compat.v1.nn.ctc_loss(self.targets, self.logits, seq_lens)
            self.cost = tf.reduce_mean(self.loss)
            cost_summary = tf.compat.v1.summary.scalar('cost', self.cost)
            self.summary = tf.compat.v1.summary.merge([cost_summary])
            # self.optimizer = tf.compat.v1.train.MomentumOptimizer(args.lr, momentum=args.momentum).minimize(self.cost)
            self.optimizer = tf.compat.v1.train.GradientDescentOptimizer(args.lr, use_locking=False, name='GradientDescent').minimize(self.cost)
            # self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr).minimize(self.cost)


        if self.mode == model_modes.EVALUATE or self.mode == model_modes.INFER:
            if args.beam_width > 0:
                self.decoded, self.log_prob = tf.compat.v1.nn.ctc_beam_search_decoder(self.logits, seq_lens, beam_width=args.beam_width)
            else:
                self.decoded, self.neg_sum_logits = tf.nn.ctc_greedy_decoder(self.logits, seq_lens)

        if self.mode == model_modes.EVALUATE:
            with tf.compat.v1.variable_scope("LER_evaluation"):
                self.ler = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.targets))
                ler_summary = tf.compat.v1.summary.scalar('label error rate', self.ler)
                self.summary = tf.compat.v1.summary.merge([ler_summary])

        self.saver = tf.compat.v1.train.Saver()
            
    @classmethod
    def load(cls, hparams_path, checkpoint_path, sess, mode):

        # Loading the hyperparameters from the passed filepath (.../config.py)
        with open(hparams_path, 'rb') as f:
            hparams = pickle.load(f)

        # Creating an object of this class with the obtained hparams and the passed mode
        obj = cls(mode, hparams)

        # Restore a saved model from the passed filepath
        obj.saver.restore(sess, checkpoint_path)

        return obj


    # A method for training the model
    def train(self, inputs, targets, sess):
      assert self.mode == model_modes.TRAIN
      # inputs = sess.run(autoencoder(inputs))
      return sess.run([self.cost, self.optimizer, self.summary], feed_dict={self.inputs: inputs, self.targets: targets})


    # A method for evaluating the model
    def eval(self, inputs, targets, sess):

        # Make sure we are in evaluation mode
        assert self.mode == model_modes.EVALUATE
        # inputs = sess.run(autoencoder(inputs))
        # Evaluate the model with the passed inputs and target labels
        return sess.run([self.ler, self.summary], feed_dict={self.inputs: inputs, self.targets: targets})


    # A method for decoding an audio file (inference)
    def infer(self, inputs, sess):

        # Make sure we are in inference mode
        assert self.mode == model_modes.INFER

        # init = tf.global_variables_initializer()
        # sess.run(init)
        return sess.run([self.decoded], feed_dict={self.inputs: inputs})


    # A method for saving the model
    def save(self, path, sess, global_step=None):
        # Enable eager execution mode
        tf.compat.v1.disable_eager_execution()
        # Save the hyperparameters first
        with open(os.path.join(self.args.hparams_path), 'wb') as f:
            pickle.dump(self.args, f)

        # Save the current state of the model
        self.saver.save(sess, path, self.args.global_step)
