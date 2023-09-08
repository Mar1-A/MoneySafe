Telephone Fraud Detection using LSTM-based Speech Recognition: Final Year Project

Overview
This repository houses the code and resources related to my final year project, which centers around the use of deep learning to detect and counteract telephone-based fraud. In particular, the project addresses the challenges posed by scam calls in the UK that seek to extract sensitive bank card information. The project employs an LSTM-based neural network model to recognize and transcribe particular key phrases from real-time conversational speech, aiming to identify potential fraudulent activities.

Key Features
Datasets:
Large-vocabulary dataset (LibriSpeech corpus): Used to ensure the model's adaptability across diverse accents and speaking styles.
Small-vocabulary dataset (personalized speech data): Specialized data set tailored to the specific needs of fraud detection.
Model Architecture:
Feature Extraction: Mel Frequency Cepstral Coefficients (MFCCs) combined with a Deep Belief Network (DBN) that comprises 4 Restricted Boltzmann Machine (RBM) layers.
The main model relies on an LSTM-based Recurrent Neural Network (RNN) for effective sequence modeling.
Results
The model achieved a Label Error Rate (LER) of 55.5% on the large-vocabulary dataset.
A notable LER of 7.9% was observed on the small-vocabulary dataset, underscoring the model's efficacy.
Project Background
This project was undertaken as part of my final year dissertation. It stands as a testament to my skills in deep learning, speech processing, and security applications. The overarching goal was to harness cutting-edge AI techniques to address a pressing real-world issue, and this repository provides an in-depth look into the methodologies and results of this endeavor.
