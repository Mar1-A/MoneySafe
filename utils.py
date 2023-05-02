"""
Author: Marwan Alalloush
discription: This file contains the code for helper functions.
"""
import os
import tensorflow as tf
import numpy as np
from config import args
import matplotlib.pyplot as plt
import librosa
import os
import random

def load_filenames():
    train_files = []
    test_files = []

    # Looping through files in the training directory
    for file in os.listdir(args.train_path):
        if file not in ['.', '..', '.DS_Store']:
            # Appending each file to the train_files array
            train_files.append(file)

    # Looping through file in the testing directory
    for file in os.listdir(args.test_path):
        if file not in ['.', '..', '.DS_Store']:
            # Appending each file to the test_files array
            test_files.append(file)

    # Shuffle the train_files and test_files lists
    random.shuffle(train_files)
    random.shuffle(test_files)

    return train_files, test_files

# Computing the actual sequence lengths without the padding for all examples in a batch
def compute_seq_lens(input):

	# Keep only the highest frequency measurement for each time step --> flattening the 3rd dimension; convert to 1s and 0s
	used = tf.sign(tf.reduce_max(tf.abs(input), 2))

	# Sum all time steps where there is some frequency info (value is 1, not 0) --> flattening the 2nd dimension and giving an array of sequence lengths for all examples in the batch.
	lengths = tf.reduce_sum(used, 1)
	lengths = tf.cast(lengths, tf.int32)

	return lengths

# Creating a sparse tensor for the labels (returning the 3 dense tensors that make up the sparse one)
def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


# Padding all sequences in a batch, so they are all equal length (equal to the longest one)
def pad_sequence(sequences, max_len):

	padded_sequences = []

	for seq in sequences:
		# Creating zero-filled vectors of length n_features to make all sequences equal in length
		padding = [np.zeros(len(seq[1])) for _ in range(max_len - len(seq))]

		# If padding is needed, concatenate it to the sequence and append it to the return array
		if len(padding) > 0:
			padded_sequences.append(np.concatenate((seq, padding), axis=0))
		else:
			padded_sequences.append(seq)

	# Will return a tensor of shape (batch_size, max_len, n_features)
	return padded_sequences 


def calculate_input_max_len():
    max_len = 0
    filename = ''
    for file in os.listdir(args.train_path):
        if file not in ['.','..','.DS_Store']:
            if file.endswith('.npy'):
                train_arr = np.load(os.path.join(args.train_path, file), allow_pickle=True);
                train_audio_len = train_arr[0].shape[0]
                if train_audio_len > max_len:
                    max_len = train_audio_len
                    filename = file

    return max_len, filename


def create_context_window(mfccs, context_frames=5):
    padding = np.zeros((mfccs.shape[0], (context_frames - 1) // 2))
    padded_mfccs = np.hstack((padding, mfccs, padding))
    windows = []

    for i in range(padded_mfccs.shape[1] - context_frames + 1):
        window = padded_mfccs[:, i:i + context_frames]
        windows.append(window.flatten())

    return np.array(windows)

def create_context_windows(batch_mfccs, context_frames):
    batch_windows = []

    # Iterate through each input array (MFCC) in the batch
    for mfccs in batch_mfccs:
        # Create padding (zeros) for the left and right side of each MFCC array
        padding = np.zeros((mfccs.shape[0], (context_frames - 1) // 2))
        padded_mfccs = np.hstack((padding, mfccs, padding))

        # Iterate through the padded MFCC array, creating context windows
        for i in range(padded_mfccs.shape[1] - context_frames + 1):
            # Extract the window with a specified number of context frames
            window = padded_mfccs[:, i:i + context_frames]

            # Flatten the window and add it to the batch
            batch_windows.append(window.flatten())

    return np.array(batch_windows)



def compute_mfcc(audio_data, sample_rate, num_mfcc = 13, n_fft = 2048, n_mels = 26, frame_size = 0.016, frame_stride = 0.008):
    
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])

    # Frame creation
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate
    signal_length = len(emphasized_audio)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_audio, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming window
    frames *= np.hamming(frame_length)

      # Fast Fourier Transform
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))

    # MEL-scaled spectrogram
    mel_spec = librosa.feature.melspectrogram(S=mag_frames.T, sr=sample_rate, n_fft=n_fft, n_mels=n_mels)

    # Logarithmic compression
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Discrete Cosine Transform
    mfcc = np.zeros((num_mfcc, num_frames))
    for m in range(num_mfcc):
        for k in range(n_mels):
            mfcc[m, :] += log_mel_spec[k, :] * np.cos(np.pi * m / num_mfcc * (k + 0.5))

    # Mean and standard deviation normalization
    mfcc -= np.mean(mfcc, axis=1, keepdims=True)
    mfcc /= np.std(mfcc, axis=1, keepdims=True)

    # Add the first and second derivatives
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Stack the feature vectors
    mfcc_feat = np.vstack((mfcc, delta_mfcc, delta2_mfcc))

    return mfcc_feat.T

def reshap_data(dbn_features):
    # Calculate the number of frames per sample
    frames_per_sample = dbn_features.shape[0] // args.batch_size

    # Reshape the DBN output to match the input shape expected by the LSTM
    lstm_input = dbn_features.reshape(args.batch_size, frames_per_sample, -1)

    return lstm_input
