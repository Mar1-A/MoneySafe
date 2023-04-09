import utils
from config import args
import tensorflow as tf
import os

from tensorboard import default as tb_default
from tensorboard import program as tb_program
from LSTM import SpeechRecognitionModel, model_modes

from tensorflow.python import debug as tf_debug
from comet_ml import Experiment

train_files, test_files = utils.load_filenames()
print(' -> Filenames --- LOADED SUCCESSFULLY')

args.input_max_len, _ = utils.calculate_input_max_len()
import numpy as np
# Defining the number of frequency bins
# Obtained by reading the shape of a random file from the dataset, since all of them are preporcessed in the same way and have the same frequency resolution.
arr = np.load(args.train_path + '/103-1240-0000.npy', allow_pickle=True)
args.num_features = arr[0].shape[1]
print(arr[0].shape)

print(' -> Number of features --- LOADED SUCCESSFULLY')




# Removes nodes from the default graph and reset it
tf.compat.v1.reset_default_graph()    
#############################
###  GENERAL PREPARATION  ###
#############################
def main(experiment=Experiment(api_key='dummy_key', disabled=True)):
    experiment.log_parameters(args)
    # Creating a new logging directory
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # Creating the graph structures for training and evaluation
    train_graph = tf.Graph()
    eval_graph = tf.Graph()

    def print_model_summary(variables):
        print("Layer Name\tVariable Shape")
        print("="*35)
        for var in variables:
            print("{}\t{}".format(var.name, var.shape))



    ##################################
    ###   INITIALIZING THE MODEL   ###
    ##################################
    import time 
    import pickle

    print('\nInitializing model...\n')

    # Specifying the processor on which the model will be trained
    device = '/gpu:0'

    # Pulling some hyper-parameters from the config file
    epochs = args.epochs
    batch_size = args.batch_size
    steps_per_checkpoint = args.steps_per_checkpoint
    checkpoints_dirpath = args.checkpoints_dirpath
    checkpoint_num = args.checkpoint_num

    # If the checkpoint path doesn't exist - creating it.
    if not os.path.exists(checkpoints_dirpath):
        os.makedirs(checkpoints_dirpath)

    # Creating training and evaluation sessions
    train_sess =  tf.compat.v1.Session(graph=train_graph)

    eval_sess =  tf.compat.v1.Session(graph=eval_graph)

    # checkpoints_path = os.path.join(checkpoints_dirpath, 'model')


    # Creating a training model object
    with train_graph.as_default():
        with tf.device(device):
            # If training is resumed from a checkpoint
            if args.load_from_checkpoint == True:
                train_model = SpeechRecognitionModel.load(args.hparams_path, args.checkpoints_dirpath + 'checkpoint-{}'.format(args.checkpoint_num), train_sess, model_modes.TRAIN)
            
            # If training starts from scratch
            else:
                train_model = SpeechRecognitionModel(model_modes.TRAIN)
                variables_initializer = tf.compat.v1.global_variables_initializer()


    # Creating an evaluation model object
    with eval_graph.as_default():
        with tf.device(device):
            # If training is resumed from a checkpoint
            if args.load_from_checkpoint == True:
                eval_model = SpeechRecognitionModel.load(args.hparams_path, args.checkpoints_dirpath + '/checkpoint-{}'.format(args.checkpoint_num), eval_sess, model_modes.EVAL)
            
            # If training starts from scratch
            else:
                eval_model = SpeechRecognitionModel(model_modes.EVALUATE)

    # Creating summary logging file writers ---> will be populated later with methods like .add_summary()
    with tf.compat.v1.Graph().as_default():
        training_logger = tf.compat.v1.summary.FileWriter(os.path.join(args.log_dir, 'train'), graph=train_graph)
        eval_logger = tf.compat.v1.summary.FileWriter(os.path.join(args.log_dir, 'eval'), graph=eval_graph)


    # Initializing all variables in the model if training starts from scratch
    if args.load_from_checkpoint == False:
        train_sess.run(variables_initializer)
        variables = tf.compat.v1.trainable_variables()
        print_model_summary(variables)

    # Setting the global step and getting current time when training starts
    
    start_time = time.time()

    # Saving the initial state of the model ---> saving all hyper-parameters and variables from the model
    # tf.compat.v1.enable_eager_execution()
    train_model.save(checkpoints_dirpath, train_sess, global_step=args.global_step)


    print('\n\n     +++ MODEL INITIALIZED +++')

    import matplotlib.pyplot as plt
    #################################
    ####    TRAINING EXECUTION    ####
    ##################################
    args.batch_size = 10
    print('Training...')

    # Run until interrupted by the keyboard (user)
    try:

        # Looping for as many times as epochs are specified
        for ep in range(args.epochs):

            # Defininf the number of the current epoch
            current_epoch = ep

            # Dividng the training data into batches and looping through them
            for i in range(int(len(train_files)/batch_size)):
                print('--{}'.format(i))
                curr_time = time.time()

                # Defining placeholders
                batch_train_inputs = []
                batch_train_labels = []
                # batch_train_seq_len = []
                filenames = []

                # Looping through the filenames for the current batch
                for file in train_files[i*batch_size:(i+1)*batch_size]:
                    # Reading in the .npy file for the training example
                    arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
                    filenames.append(file)
                    # Appending the audio and transcription to the batch arrays
                    batch_train_inputs.append(arr[0])
                    batch_train_labels.append(arr[1])

                batch_train_inputs = np.asarray(utils.pad_sequences(batch_train_inputs, args.input_max_len), dtype=np.float32)
                batch_train_labels = utils.sparse_tuple_from(np.asarray(batch_train_labels))
                # print(batch_train_inputs.shape)
                # print(batch_train_seq_len)
                # Converting to sparse representation so as to to feed SparseTensor input
                # tf.compat.v1.enable_eager_execution()
                # batch_train_inputs = autoencoder(batch_train_inputs).numpy()
                
                # print(ler)
                # Run the training method from the model class. Returns the cost value and the summary.
                cost, _, summary= train_model.train(batch_train_inputs, batch_train_labels, train_sess)

                # Updating the global step
                args.global_step += batch_size

                # Adding summary to the training logs
                training_logger.add_summary(summary, global_step=args.global_step)
                experiment.log_metric("Loss", cost.item(), step = args.global_step)
                # Calculating time for the console output
                tot = time.time() - start_time
                h = int(tot/3600)
                m = int((tot/3600-h)*60)
                s = int((((tot/3600-h)*60)-m)*60)
                if h < 10: h = '0{}'.format(h)
                if m < 10: m = '0{}'.format(m)
                if s < 10: s = '0{}'.format(s)
                time_tot = '{}:{}:{}'.format(h, m, s)
                print('~~~ \nEpoch: {} \nGlobal Step: {} \nCost: {} \nTime: {} s\nTime total: {} \nFilenames: {}\n~~~'.format(current_epoch, args.global_step, cost, time.time() - curr_time, time_tot, filenames))


                # If the global step is a multiple of steps_per_checkpoint ---> if it is time for a checkpoint
                if args.global_step % steps_per_checkpoint == 0:

                    print('Checkpointing... (Global step = {})'.format(args.global_step))

                    # Saving a checkpoint after a certain number of iterations
                    current_checkpoint = train_model.saver.save(train_sess, args.checkpoints_dirpath, global_step=args.global_step)
                        
                    # Immediately restoring the saved model to evaluate it!
                    eval_model.saver.restore(eval_sess, current_checkpoint)
                
                    # EVALUATING THE MODEL AT CHECKPOINT
                    ler, summary = eval_model.eval(batch_train_inputs, batch_train_labels, eval_sess)
                    # Adding summary to the evaluation logs
                    eval_logger.add_summary(summary, global_step=args.global_step)
                    experiment.log_metric("ler", ler.item(), step = args.global_step)
                    # Mergin all summaries
                    tf.compat.v1.summary.merge_all()

                    print('#####\nEvaluation --- LER: {} %\n#####'.format(ler*100))

    except KeyboardInterrupt:

        print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, args.global_step, time_tot))
        train_sess.close()
        eval_sess.close()

    print('===========\nTotal iterations: {}\nAudio files processed: {}\nTime taken: {}\n==========='.format(i, args.global_step, time_tot))
    train_sess.close()
    eval_sess.close()



comet_api_key = "Uj8SFBUzFMMGwax7pJyk3RfAC" # add your api key here
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