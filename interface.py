"""
Author: Marwan Alalloush
"""

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
from DBN import DBN
from lables_preprocess import TextTransform

args.input_max_len = 965
print(args.input_max_len)
tf.compat.v1.reset_default_graph()

def infer():
    infer_sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()
    
    #load dbn model
    dbn = DBN.load_model('dbn_39.pickle')

    # Load the model
    model = SpeechRecognitionModel.load('Newfolder/hparams', 'Newfolder/checkpoints-344160', infer_sess, mode=model_modes.INFER)

    # Load the audio file and compute MFCC features
    audio, sr = librosa.load('audio.wav', sr=22050)
    features = utils.compute_mfcc(audio_data=audio, sample_rate=sr)

    # Pad the input features
    features_padded = np.asarray(utils.pad_sequence([features], args.input_max_len), dtype=np.float32)
    
    #creat context window
    features = utils.context_windows(features_padded)
    #extract features with dbn
    features,_ = dbn.dbn_output(features)
    #reshape data to fit model
    features = utils.reshap_data(features)
    # Perform inference
    decoded = model.infer(features, infer_sess)

    orginal_text = 'this is a test'
    text_prediction = TextTransform().int_to_text(decoded[0][0].values)

    # Print results
    print('\n\n###############\nActual Text: {}\nPredicted Text: {}\nTook {} seconds.'.format(orginal_text, text_prediction, time.time()-start_time))

if __name__ == '__main__':
    infer()

# # # !pip install pyaudio
# import pyaudio

# def microphone_continuous_infer(chunk_duration=1.0, overlap=0.25):
#     # Load the model and create the session
#     infer_sess = tf.compat.v1.Session()
#     tf.compat.v1.disable_eager_execution()
#     model = SpeechRecognitionModel.load('Newfolder/hparams', 'Newfolder/checkpoints-344160', infer_sess, mode = model_modes.INFER)

#     # Initialize PyAudio
#     audio_format = pyaudio.paInt16
#     channels = 1
#     rate = 22050  # Sample rate of 16 kHz
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
#             features = utils.compute_mfcc(audio_data=audio_buffer.astype(np.float32), sample_rate=rate)
#             # features = features.T

#             # Pad the features
#             features_padded = np.asarray(utils.pad_sequence([features], args.input_max_len), dtype=np.float32)

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


