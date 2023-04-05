import os
import tensorflow as tf
import utils
from config import args
import RBM
import DBN
# Load MFCCs and transcripts from the same .npy files
train_files, test_files = utils.load_filenames()
print(' -> Filenames --- LOADED SUCCESSFULLY')
args.input_max_len= 3493
print(args.input_max_len)


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
    err = rbm.train(train_files)
    error_list.append(err)
    #Return the output layer
    outputX, reconstructedX, hiddenX = rbm.rbm_output(train_files)
    outputList.append(outputX)
    inputX = hiddenX

#initialize the DBN
dbn = DBN(original_input_size=195, input_size=39, output_size=2,
           learning_rate=0.1, epochs=10, batchsize=100, rbmOne=rbm_list[0],
             rbmTwo=rbm_list[1], rbmThree=rbm_list[2], rbmFour=rbm_list[3])
dbn.train(train_files)