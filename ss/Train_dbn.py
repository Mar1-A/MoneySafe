import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import DBN
import utils
from config import args
from RBM import RBM


def pad_mfccs(mfcc, max_frames):
    
  num_frames, num_features = mfcc.shape
  padding_length = max_frames - num_frames
  padding = np.zeros((padding_length, num_features))
  padded_mfcc = np.vstack((mfcc, padding))

  return padded_mfcc
def concatenate_mfccs_single_observation(data, num_frames):
    total_frames, num_features = data.shape
    concatenated_data = np.zeros(((total_frames - num_frames + 1), num_frames * num_features))
    index = 0
    for j in range(total_frames - num_frames + 1):
        concatenated_data[index] = data[j : j + num_frames].flatten()
        index += 1
    return concatenated_data

train_files, test_files = utils.load_filenames()
print(' -> Filenames --- LOADED SUCCESSFULLY')
args.input_max_len, _= utils.calculate_input_max_len()
print(args.input_max_len)

train_inputs = []
# # Load MFCCs and transcripts from the same .npy files
for file in test_files:
  arr = np.load(args.test_path + '/{}'.format(file), allow_pickle=True)
#   mfccs = pad_mfccs(arr[0], args.input_max_len)
#   mfccs = concatenate_mfccs_single_observation(mfccs,num_frames=5)
  train_inputs.append(arr[0])
#   print(mfccs.shape)
  # batch_train_labels.append(arr[1])

train_inputs = pad_sequences(train_inputs, dtype=np.float32)
train_inputs = utils.concatenate_mfccs(train_inputs, num_frames=5)
print(train_inputs.shape)
#initialize the RBM
rbm_list = []
rbm_list.append(RBM(195,1024,1.0,100,200))
rbm_list.append(RBM(1024,512,1.0,100,200))
rbm_list.append(RBM(512,256,1.0,100,200))
rbm_list.append(RBM(256,39,1.0,100,200))

# Train the RBM
tf.compat.v1.disable_eager_execution()
outputList = []
error_list = []

#For each RBM in our list
for i in range(0,len(rbm_list)):
    print('RBM', i+1)
    #Train a new one
    rbm = rbm_list[i]
    err = rbm.train(train_files, train_inputs)
    error_list.append(err)
    #Return the output layer

    outputX, reconstructedX, hiddenX = rbm.rbm_output(train_inputs)
    outputList.append(outputX)
    inputX = hiddenX

#initialize the DBN
dbn = DBN(original_input_size=195, input_size=39, output_size=2,
           learning_rate=0.1, epochs=10, batchsize=100, rbmOne=rbm_list[0],
             rbmTwo=rbm_list[1], rbmThree=rbm_list[2], rbmFour=rbm_list[3])
dbn.train(train_inputs)