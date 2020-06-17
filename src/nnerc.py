#!usr/bin/python3

'''
Fits model and makes predictions. 
'''

from nnerc_learner import learn
from nnerc_classifier import predict
from pathlib import Path

if __name__ == "__main__":

    # Learn

    # modelname = learn(
    #     traindir = Path('./data/Train'), validationdir = Path('./data/Devel'), version=54
    # )

    # Predict

    predict('LSTM-CRF_54', Path('./data/Test-NER'), 'task9.1_LSTM_NER.txt')


