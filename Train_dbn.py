"""
Author: Marwan Alalloush
discription: This file contains the code for training the RBM layers and DBN model.
"""
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import utils
from config import args

from RBM import RBM
from DBN import DBN

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

args.lr = 0.0001

train_files, test_files = utils.load_filenames()
print(' -> Filenames --- LOADED SUCCESSFULLY')

#initialize the RBM
rbm_list = []
rbm_list.append(RBM(195,1024,args.lr,10,))
rbm_list.append(RBM(1024,512,args.lr,10,437000))
rbm_list.append(RBM(512,256,args.lr,10,437000))
rbm_list.append(RBM(256,39,args.lr,10,437000))

# Train the RBM
tf.compat.v1.disable_eager_execution()

print('Training first RBM')
for ep in range(args.epochs):
    for i in range(int(len(train_files) / args.batch_size)):
        print('--{}'.format(i))
        batch_train_inputs = []
        filenames = []

        for file in train_files[i * args.batch_size:(i + 1) * args.batch_size]:
            arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
            filenames.append(file)
            batch_train_inputs.append(arr[0])

        batch_train_inputs = pad_sequences(batch_train_inputs, dtype=np.float32)
        batch_train_inputs = utils.create_context_windows(batch_train_inputs, 5)
        print(batch_train_inputs.shape)
        error = rbm_list[0].train(batch_train_inputs)

    print(f'Epoch: {ep}, reconstruction error: {error}')

layer_name = f'rbm_layer_{rbm_list[0]._output_size}'
rbm_list[0].save_rbm(layer_name)
print(f'Saved {layer_name}.')

print('Training second RBM')
for ep in range(args.epochs):
    for i in range(int(len(train_files) / args.batch_size)):
        print('--{}'.format(i))
        batch_train_inputs = []
        filenames = []

        for file in train_files[i * args.batch_size:(i + 1) * args.batch_size]:
            arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
            filenames.append(file)
            batch_train_inputs.append(arr[0])

        batch_train_inputs = pad_sequences(batch_train_inputs, dtype=np.float32)
        batch_train_inputs = utils.create_context_windows(batch_train_inputs, 5)
        batch_train_inputs, _, _ = rbm_list[0].rbm_output(batch_train_inputs)
        print(batch_train_inputs.shape)
        error = rbm_list[1].train(batch_train_inputs)

    print(f'Epoch: {ep}, reconstruction error: {error}')

layer_name = f'rbm_layer_{rbm_list[1]._output_size}'
rbm_list[1].save_rbm(layer_name)
print(f'Saved {layer_name}.')

print('Training third RBM')
for ep in range(args.epochs):
    for i in range(int(len(train_files) / args.batch_size)):
        print('--{}'.format(i))
        batch_train_inputs = []
        filenames = []

        for file in train_files[i * args.batch_size:(i + 1) * args.batch_size]:
            arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
            filenames.append(file)
            batch_train_inputs.append(arr[0])

        batch_train_inputs = pad_sequences(batch_train_inputs, dtype=np.float32)
        batch_train_inputs = utils.create_context_windows(batch_train_inputs, 5)
        batch_train_inputs, _, _ = rbm_list[0].rbm_output(batch_train_inputs)
        batch_train_inputs, _, _ = rbm_list[1].rbm_output(batch_train_inputs)
        print(batch_train_inputs.shape)
        error = rbm_list[2].train(batch_train_inputs)

    print(f'Epoch: {ep}, reconstruction error: {error}')

layer_name = f'rbm_layer_{rbm_list[2]._output_size}'
rbm_list[1].save_rbm(layer_name)
print(f'Saved {layer_name}.')

print('Training forth RBM')
for ep in range(args.epochs):
    for i in range(int(len(train_files) / args.batch_size)):
        print('--{}'.format(i))
        batch_train_inputs = []
        filenames = []

        for file in train_files[i * args.batch_size:(i + 1) * args.batch_size]:
            arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
            filenames.append(file)
            batch_train_inputs.append(arr[0])

        batch_train_inputs = pad_sequences(batch_train_inputs, dtype=np.float32)
        batch_train_inputs = utils.create_context_windows(batch_train_inputs, 5)
        batch_train_inputs, _, _ = rbm_list[0].rbm_output(batch_train_inputs)
        batch_train_inputs, _, _ = rbm_list[1].rbm_output(batch_train_inputs)
        batch_train_inputs, _, _ = rbm_list[2].rbm_output(batch_train_inputs)
        print(batch_train_inputs.shape)
        error = rbm_list[3].train(batch_train_inputs)

    print(f'Epoch: {ep}, reconstruction error: {error}')

layer_name = f'rbm_layer_{rbm_list[3]._output_size}'
rbm_list[3].save_rbm(layer_name)
print(f'Saved {layer_name}.')


#initialize the DBN
dbn = DBN(original_input_size=195, input_size=39, output_size=39,
           learning_rate=args.lr, rbmOne=rbm_list[0], rbmTwo=rbm_list[1], rbmThree=rbm_list[2], rbmFour=rbm_list[3])

print('Training the DBN')
for ep in range(args.epochs):
    for i in range(int(len(train_files) / args.batch_size)):
        print('--{}'.format(i))
        batch_train_inputs = []
        filenames = []

        for file in train_files[i * args.batch_size:(i + 1) * args.batch_size]:
            arr = np.load(args.train_path + '/{}'.format(file), allow_pickle=True)
            filenames.append(file)
            batch_train_inputs.append(arr[0])

        batch_train_inputs = pad_sequences(batch_train_inputs, dtype=np.float32)
        batch_train_inputs = utils.create_context_windows(batch_train_inputs, 5)
        print(batch_train_inputs.shape)
        error = dbn.train(batch_train_inputs)

    print(f'Epoch: {ep}, reconstruction error: {error}')

model_name = f'dbn_{dbn._output_size}'
dbn.save_model(model_name)
print(f'Saved {model_name}.')