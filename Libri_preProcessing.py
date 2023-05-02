"""
Author: Marwan Alalloush
description: This file contains the code for preprocessing the LibriSpeech dataset.
"""
#importing requierd libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import urllib.request
# import tensorflow_io as tfio
import librosa.display
import matplotlib.pyplot as plt 
import numpy as np
import sys
import tarfile
import tempfile
import logging
import soundfile as sf
from lables_preprocess import TextTransform
from tqdm import tqdm
import librosa
import utils

LIBRI_SPEECH_URLS = {
    "train-clean-100":
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "test-clean":
        "http://www.openslr.org/resources/12/test-clean.tar.gz"
}

def download_and_extract(directory, url):
  """Download and extract the given split of dataset.
  Args:
    directory: the directory where to extract the tarball.
    url: the url to download the data file.
  """

#   directory = os.path.join(directory, "data")

  if not tf.io.gfile.exists(directory):
    tf.io.gfile.makedirs(directory)

  _, tar_filepath = tempfile.mkstemp(suffix=".tar.gz")

  try:
    logging.info("Downloading %s to %s" % (url, tar_filepath))

    def _progress(count, block_size, total_size):
      sys.stdout.write("\r>> Downloading {} {:.1f}%".format(
          tar_filepath, 100.0 * count * block_size / total_size))
      sys.stdout.flush()

    urllib.request.urlretrieve(url, tar_filepath, _progress)
    print()
    statinfo = os.stat(tar_filepath)
    logging.info(
        "Successfully downloaded %s, size(bytes): %d" % (url, statinfo.st_size))
    with tarfile.open(tar_filepath, "r") as tar:
      tar.extractall(directory)
  finally:
    tf.io.gfile.remove(tar_filepath)


def preprocess_librispeech(path):
    """
    the output of this function is a npy file for each audio file, the npy file will hold an array of mfcc, and a list of numbers which represent the transcript for that file
    """

    # Creating a directory for the processed data if it doesn't exist yet.
    if not os.path.exists('Data1/librispeech_processed'):
        os.makedirs('Data1/librispeech_processed')

    # Creating subdirectories for the different parts of the data set if they don't exist yet.
    dirpaths = ['test-clean', 'train-clean-100']
    for dirpath in dirpaths:
        if not os.path.exists('Data1/librispeech_processed/{}'.format(dirpath)):
            os.makedirs('Data1/librispeech_processed/{}'.format(dirpath))

    # Placeholders
    audio_feats = {}
    transcripts = {}

    print('\n ####  PREPROCESSING LIBRISPEECH CORPUS  ####\n')
    # Go through all files and folders under 'data/librispeech'
    for root, dirs, files in tqdm(list(os.walk(path))):
        # If files exist and the path is not to the master directory... 
        if files and root != path:
            # Determine the dataset type (train or test) based on the root directory path
            dataset_type = None
            for dirpath in dirpaths:
                if dirpath in root:
                    dataset_type = dirpath
                    break

            # Loop through all files in the directory
            for file in files:
                if file[-5:] == '.flac': # If it is an audio file...
                    # Read the audio file
                    audio, sr = librosa.load(os.path.join(root, file), sr = None)
                    # Compute its MFCCs
                    features = utils.compute_mfcc(audio_data=audio, sample_rate=sr)
                    # Append the Mfcca to the audio_feats dictionary
                    audio_feats[file[:-5]] = features.T
                elif file[-4:] == '.txt': # If it is a text file (transcription)...
                    with open(os.path.join(root, file)) as f:
                        # Loop through all lines and slice them up to get the ids and transcriptions
                        for line in f.readlines():
                            audio_file_id = line.split(' ', 1)[0]
                            transcription = line.split(' ', 1)[1].strip('\n').lower()
                            transcription_mapped = TextTransform().text_to_int(transcription)
                            
                            # Append the transcription to the transcripts dictionary
                            transcripts[audio_file_id] = transcription_mapped

            # Saving all the info from the dictionaries above to the according folder
            for key in audio_feats.keys():
                save_path = f'Data1/librispeech_processed/{dataset_type}/{key}.npy'
                if not os.path.exists(save_path):
                    if key in transcripts:
                        np.save(save_path, [audio_feats[key], transcripts[key]])
                    else:
                        print(f"Key {key} not found in transcripts.")

            # Clearing the dictionaries to prepare them for the next iteration of the loop
            audio_feats.clear()
            transcripts.clear()



def download_and_process_datasets(directory, datasets):
  """Download and pre-process the specified list of LibriSpeech dataset.
  Args:
    directory: the directory to put all the downloaded and preprocessed data.
    datasets: list of dataset names that will be downloaded and processed.
  """

  logging.info("Preparing LibriSpeech dataset: {}".format(
      ",".join(datasets)))
  for dataset in datasets:
    logging.info("Preparing dataset %s", dataset)
    # dataset_dir = os.path.join(directory, dataset)
    download_and_extract(directory, LIBRI_SPEECH_URLS[dataset])
    preprocess_librispeech(directory)

if __name__ == '__main__':
	path = './Data'
	download_and_process_datasets(path, LIBRI_SPEECH_URLS)
	# preprocess_librispeech(path)

