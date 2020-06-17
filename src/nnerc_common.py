#!bin/usr/python3

from pathlib import Path
import json

traindata_file  = Path('./data/xml2json/traindata.json')
traindir        = Path('./data/Train')

valdata_file    = Path('./data/xml2json/develdata.json')
validationdir   = Path('./data/Devel')

testdata_file   = Path('./data/xml2json/testdata.json')
testdir         = Path('./data/Test')

# glove_file = './data/glove.6B/glove.6B.100d.txt'
glove_file = Path('./data/glove.840B.300d.txt')

class NetConfig:
    def __init__(self, version):
        self.config_f = Path('./config/config_%i.json' % version)

        with self.config_f.open('r') as cf:
            self.config = json.load(cf)

        # ************************************************
        # extract network's hyperparameters from json file   

        # architectural parameters

        self.pre_trained =       self.config['arch']['pre_trained']
        self.w_embedding =       self.config['arch']['w_embedding']
        self.c_embedding =       self.config['arch']['c_embedding']
        self.lstm_char_units =   self.config['arch']['lstm_char_units']
        self.lstm_main_units =   self.config['arch']['lstm_main_units']
        self.dense_units =       self.config['arch']['dense_units']
        self.return_sequences =  self.config['arch']['return_sequences']
        self.mask_zero =         self.config['arch']['mask_zero']
        self.activation =        self.config['arch']['activation']

        #training parameters

        self.batch =        self.config['training']['batch']
        self.dropout =      self.config['training']['dropout']
        self.rcrr_dropout=  self.config['training']['rcrr_dropout']
        self.epochs =       self.config['training']['epochs']
        self.optimizer =    self.config['training']['optimizer']
        self.loss =         self.config['training']['loss']
        self.metrics =      self.config['training']['metrics']

        #********************************************************
