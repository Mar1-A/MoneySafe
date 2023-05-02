"""
Author: Marwan Alalloush
discription: This file contains the code for training the model.
"""

import numpy as np
import tqdm
import utils
from config import args
import tensorflow as tf
import os
import time

from BLSTM import SpeechRecognitionModel, model_modes
from DBN import DBN

from tensorflow.python import debug as tf_debug
from comet_ml import Experiment

# Load filenames for training and testing
train_files, test_files = utils.load_filenames()

# Calculate input maximum length
args.input_max_len, _ = utils.calculate_input_max_len()

# Determine the number of features
arr = np.load(args.train_path + '/61-70968-0000.npy', allow_pickle=True)
args.num_features = arr[0].shape[1]

# Reset the default graph
tf.compat.v1.reset_default_graph()


def main(experiment=Experiment(api_key='dummy_key', disabled=True)):
    # Log experiment parameters
    experiment.log_parameters(args)

    # Create logging directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Create graphs for training and evaluation
    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    # Initialize model
    print('\nInitializing model...\n')
    device = '/gpu:0'
    epochs = args.epochs
    batch_size = args.batch_size
    steps_per_checkpoint = args.steps_per_checkpoint
    checkpoints_dirpath = args.checkpoints_dirpath

    # Create training and evaluation sessions
    train_sess = tf.compat.v1.Session(graph=train_graph)
    eval_sess = tf.compat.v1.Session(graph=eval_graph)

    # Create and initialize training model
    with train_graph.as_default():
        with tf.device(device):
            if args.load_from_checkpoint:
                train_model = SpeechRecognitionModel.load(args.hparams_path, args.checkpoints_dirpath + 'checkpoint-{}'.format(args.checkpoint_num), train_sess, model_modes.TRAIN)
            else:
                train_model = SpeechRecognitionModel(model_modes.TRAIN)
                variables_initializer = tf.compat.v1.global_variables_initializer()

    # Create and initialize evaluation model
    with eval_graph.as_default():
        with tf.device(device):
            if args.load_from_checkpoint:
                eval_model = SpeechRecognitionModel.load(args.hparams_path, args.checkpoints_dirpath + '/checkpoint-{}'.format(args.checkpoint_num), eval_sess, model_modes.EVAL)
            else:
                eval_model = SpeechRecognitionModel(model_modes.EVALUATE)

    # Create summary loggers
    with tf.compat.v1.Graph().as_default():
        training_logger = tf.compat.v1.summary.FileWriter(os.path.join(args.log_dir, 'train'), graph=train_graph)
        eval_logger = tf.compat.v1.summary.FileWriter(os.path.join(args.log_dir, 'eval'), graph=eval_graph)

    experiment.set_model_graph(train_graph, overwrite=False)
    experiment.set_model_graph(eval_graph, overwrite=False)

    # Initialize variables if starting from scratch
    if not args.load_from_checkpoint:
        train_sess.run(variables_initializer)

    start_time = time.time()
    train_model.save(checkpoints_dirpath, train_sess, global_step=args.global_step)

    # Load DBN model
    print('loading DBN...')
    dbn = DBN.load_model('dbn_39.pickle')

    print('\n\n     +++ MODEL INITIALIZED +++')

    # Start training
    args.batch_size = 10
    print('Training...')

    try:
        # Train for specified number of epochs
        for ep in tqdm.tqdm(range(args.epochs), desc='Epoch progress'):
            num_batches = int(len(train_files) / batch_size)
            for i in tqdm.tqdm(range(num_batches), desc='Batch progress', leave=False):
                # Prepare batch data
                batch_train_inputs = []
                batch_train_labels = []
                filenames = []

                # Loop through filenames for the current batch
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
                    filenames.append(file)
                    batch_train_inputs.append(arr[0])
                    batch_train_labels.append(arr[1])

                # Pad and convert inputs and labels to required format
                batch_train_inputs = np.asarray(utils.pad_sequence(batch_train_inputs, args.input_max_len), dtype=np.float32)
                batch_train_labels = utils.sparse_tuple_from(np.asarray(batch_train_labels))
                 #creat context window
                batch_train_inputs = utils.context_windows(batch_train_inputs)
                #extract features with DBN
                batch_train_inputs,_ = dbn.dbn_output(batch_train_inputs)
                #reshape data to fit model
                batch_train_inputs = utils.reshap_data(batch_train_inputs)
                # Train model on batch and get cost and summary
                cost, _, summary = train_model.train(batch_train_inputs, batch_train_labels, train_sess)

                # Update global step and log summary
                args.global_step += batch_size
                training_logger.add_summary(summary, global_step=args.global_step)
                experiment.log_metric("Loss", cost.item(), step=args.global_step)

                # Show training progress
                tqdm.tqdm.write('Training (Cost: {:.4f})'.format(cost.item()))

                # Save and evaluate model at checkpoints
                if args.global_step % steps_per_checkpoint == 0:
                    print('Checkpointing... (Global step = {})'.format(args.global_step))

                    # Save model checkpoint
                    current_checkpoint = train_model.saver.save(train_sess, args.checkpoints_dirpath, global_step=args.global_step)

                    # Restore saved model to evaluate it
                    eval_model.saver.restore(eval_sess, current_checkpoint)

                    # Evaluate model and log summary
                    ler, summary = eval_model.eval(batch_train_inputs, batch_train_labels, eval_sess)
                    eval_logger.add_summary(summary, global_step=args.global_step)
                    experiment.log_metric("ler", ler.item(), step=args.global_step)
                    tf.compat.v1.summary.merge_all()

                    print('#####\nEvaluation --- LER: {} %\n#####'.format(ler*100))

    except KeyboardInterrupt:
        print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, args.global_step))
        train_sess.close()
        eval_sess.close()

    print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, args.global_step))
    train_sess.close()
    eval_sess.close()

comet_api_key = "" # add your api key here
project_name = "LibriSpeech Model"
experiment_name = "speechModel"

if comet_api_key:
    experiment = Experiment(api_key=comet_api_key, project_name=project_name, parse_args=False)
    experiment.set_name(experiment_name)
    experiment.display()
else:
    experiment = Experiment(api_key='dummy_key', disabled=True)

main(experiment)
experiment.end()
