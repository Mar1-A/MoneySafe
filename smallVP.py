"""
Author: Marwan Alalloush
description: This file contains the code for preprocessing for the small vocabulary dataset.
"""

#importing requierd libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# import tensorflow_io as tfio
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np
import soundfile as sf
from lables_preprocess import TextTransform
import librosa
import utils
import pandas as pd



def preprocess_librispeech(path):
    """
    the output of this function is a npy file for each audio file, the npy file will hold an array of mfcc, and a list of numbers which represent the transcript for that file
    """

    # Creating a directory for the processed data if it doesn't exist yet.
    if not os.path.exists('Data1/librispeech_processed'):
        os.makedirs('Data1/librispeech_processed')

    # Creating subdirectories for the different parts of the data set if they don't exist yet.
    if not os.path.exists('Data1/librispeech_processed/smallvoc_processed'):
        os.makedirs('Data1/librispeech_processed/smallvoc_processed')

    print('\n ####  PREPROCESSING small vocabulary CORPUS  ####\n')
    df = pd.read_csv(path+ '/metadata.csv')
    # Go through all files and folders under 'data/librispeech'
    for i, row in df.iterrows():
        print(f"Processing file: {row[0]}")  # Add this line

        try:
            # Read the audio file
            audio, sr = librosa.load(os.path.join(path, row[0]), sr=None)
            
            # Resample the audio to a higher rate
            sr_resampled = 22050
            
            # Compute its MFCCs
            features = utils.compute_mfcc(audio_data=audio, sample_rate=sr_resampled)
            text = row[1].replace('-', '')

            transcription = TextTransform().text_to_int(text.lower())

            # Saving all the info from the dictionaries above to the according folder
            save_path = f'Data1/librispeech_processed/smallvoc_processed/{row[0][:-4]}.npy'
            if not os.path.exists(save_path):
                data_to_save = np.array([features, transcription], dtype=object)  # Use dtype=object
                np.save(save_path, data_to_save, allow_pickle=True)

        except Exception as e:
            print(f"Error processing file: {row[0]}, skipping. Error: {e}")  # Add this line



if __name__ == '__main__':
	path = './Data1/generatedData'
	# download_and_process_datasets(path, LIBRI_SPEECH_URLS)
	preprocess_librispeech(path)