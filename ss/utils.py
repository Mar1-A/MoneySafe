import os
import tensorflow as tf
import numpy as np
from config import args

# Loading the names for all files in the passed data set
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

	return train_files, test_files
# Computing the actual sequence lengths without the padding for all examples in a batch
def compute_seq_lens(input):

	# Keep only the highest frequency measurement for each time step --> flattening the 3rd dimension; convert to 1s and 0s
	used = tf.sign(tf.reduce_max(tf.abs(input), 2))

	# Sum all time steps where there is some frequency info (value is 1, not 0) --> flattening the 2nd dimension and giving an array of sequence lengths for all examples in the batch.
	lengths = tf.reduce_sum(used, 1)
	lengths = tf.cast(lengths, tf.int32)

	return lengths


# Padding all sequences in a batch, so they are all equal length (equal to the longest one)
def pad_sequences(sequences, max_len):

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
# def pad_sequences(sequences, maxlen):
#     padded_sequences = []
#     for seq in sequences:
#         if len(seq) < maxlen:
#             padding = np.zeros((maxlen - len(seq), seq.shape[-1]))  # Ensure proper padding shape
#             padded_seq = np.concatenate((seq, padding), axis=0)
#             padded_sequences.append(padded_seq)
#         else:
#             padded_sequences.append(seq[:maxlen])
#     return padded_sequences


def calculate_input_max_len():
    max_len = 0
    filename = ''
    for file in os.listdir(args.test_path):
        if file not in ['.','..','.DS_Store']:
            if file.endswith('.npy'):
                train_arr = np.load(os.path.join(args.test_path, file), allow_pickle=True);
                train_audio_len = train_arr[0].shape[0]
                if train_audio_len > max_len:
                    max_len = train_audio_len
                    filename = file

    return max_len, filename


def concatenate_mfccs(data, num_frames):
    num_samples, total_frames, num_features = data.shape
    concatenated_data = np.zeros((num_samples * (total_frames - num_frames + 1), num_frames * num_features))
    index = 0
    for i in range(num_samples):
        for j in range(total_frames - num_frames + 1):
            concatenated_data[index] = data[i, j : j + num_frames].flatten()
            index += 1
    return concatenated_data

def concatenate_mfccs_with_labels(data, labels, num_frames):
	num_samples, total_frames, num_features = data.shape
	concatenated_data = np.zeros((num_samples * (total_frames - num_frames + 1), num_frames * num_features))
	concatenated_labels = np.zeros(num_samples * (total_frames - num_frames + 1))
	index = 0
	for i in range(num_samples):
		for j in range(total_frames - num_frames + 1):
			concatenated_data[index] = data[i, j : j + num_frames].flatten()
			concatenated_labels[index] = labels[i]
			index += 1
	return concatenated_data, concatenated_labels


import numpy as np
import librosa


def compute_mfcc(audio_data, sample_rate, num_mfcc, frame_size, frame_stride, n_fft, n_mels):
    # Pre-emphasis filter
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

def normalize_mfcc(mfcc_features):
    return (mfcc_features - np.mean(mfcc_features, axis=1, keepdims=True)) / np.std(mfcc_features, axis=1, keepdims=True)