# Author: Nikola Tonev
# Date: March 2019

import os
import time
import utils
import argparse
import numpy as np
import soundfile as sf
import tensorflow as tf 
import librosa
from config import args
from BLSTM import SpeechRecognitionModel, model_modes
from lables_preprocess import TextTransform

# tf.compat.v1.logging.set_verbosity(tf.logging.WARN)

# from config import args
# from lstm import SpeechRecognitionModel, model_modes
args.input_max_len,_= utils.calculate_input_max_len()
print(args.input_max_len)
tf.compat.v1.reset_default_graph()  
def infer():

    # # Creating a command line argument parser
    # arg_p = argparse.ArgumentParser()

    # # Adding an expected argument
    # arg_p.add_argument('-af', '--audiofile', required=True, type=str)

    # # Parsing the arguments passed by the user
    # args = arg_p.parse_args()

    # Creating an inference session
    infer_sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()

    # Creating an inference model
    model = SpeechRecognitionModel.load('/content/drive/MyDrive/project/model1/hparams', '/content/drive/MyDrive/project/model1/checkpoints/checkpoint-2848', infer_sess, mode = model_modes.INFER)

    # Loading the output mapping (in the form of a dictionary ---> e.g. {'a': 0, 'b': 1, 'c': 2})
    # output_mapping = utils.load_output_mapping(hparams.dataset)

    # Reading the audio file from the path passed as a command line argument
    # audio, sr = sf.read()
    audio, sr = librosa.load('/content/drive/MyDrive/project/librispeech_data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac', sr = None)

    # Converting the audio to a spectrogram (feature representation)
    features = utils.compute_mfcc2(audio_data=audio, sample_rate=sr, num_mfcc=13, n_fft = 2048, frame_size = 0.016, frame_stride = 0.008, n_mels = 26)
    # features = features.T
    print(features.shape)
    

    # Padding the sequence so that it is the same length as all sequences on which the model was trained
    # Passing features in [] in order to make it three-dimensional (2D --> 3D)
    features_padded = np.asarray(utils.pad_sequences([features], args.input_max_len), dtype=np.float32)
    print(features_padded.shape)

    # Setting the start time for decoding
    start_time = time.time()

    # Performing decoding ---> returns a ...
    decoded = model.infer(features_padded, infer_sess)

    print(decoded)
    print('###')

    # Converting transcription IDs ---> Text
    text_prediction = TextTransform().int_to_text(decoded[0][0].values)
    print('---> Text predicition length: {}'.format(len(text_prediction)))


    print('\n\n###############\nTranscription: {}\nTook {} seconds.'.format(text_prediction, time.time()-start_time))



if __name__ == '__main__':

    infer()
# # !pip install pyaudio
# import pyaudio

# def microphone_continuous_infer(chunk_duration=1.0, overlap=0.25):
#     # Load the model and create the session
#     infer_sess = tf.compat.v1.Session()
#     tf.compat.v1.disable_eager_execution()
#     model = SpeechRecognitionModel.load('/content/drive/MyDrive/project/model1/hparams', '/content/drive/MyDrive/project/model1/checkpoints/checkpoint-2848', infer_sess, mode=model_modes.INFER)

#     # Initialize PyAudio
#     audio_format = pyaudio.paInt16
#     channels = 1
#     rate = 16000  # Sample rate of 16 kHz
#     chunk_size = int(chunk_duration * rate)
#     overlap_size = int(overlap * chunk_size)

#     audio = pyaudio.PyAudio()

#     # Start the audio stream
#     stream = audio.open(format=audio_format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk_size)

#     # Initialize buffers and transcription
#     audio_buffer = np.zeros(2 * chunk_size, dtype=np.int16)
#     print("Start speaking...")

#     try:
#         while True:
#             # Read audio data from the microphone
#             audio_data = stream.read(chunk_size)
#             new_samples = np.frombuffer(audio_data, dtype=np.int16)

#             # Add new samples to the buffer and remove old samples
#             audio_buffer[:-chunk_size] = audio_buffer[chunk_size:]
#             audio_buffer[-chunk_size:] = new_samples

#             # Extract features
#             features = compute_mfcc2(y=audio_buffer.astype(np.float32), sr=rate, n_mfcc=args.num_features)
#             # features = features.T

#             # Pad the features
#             features_padded = np.asarray(pad_sequences([features], args.input_max_len), dtype=np.float32)

#             # Perform decoding
#             decoded = model.infer(features_padded, infer_sess)
#             text_prediction = TextTransform().int_to_text(decoded[0][0].values)

#             print('\r', text_prediction, end='', flush=True)

#     except KeyboardInterrupt:
#         print("\nStopped")

#     finally:
#         # Clean up the audio stream
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()

# if __name__ == '__main__':
#     microphone_continuous_infer()
