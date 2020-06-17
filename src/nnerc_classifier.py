#!usr/bin/python3
from nnerc_utils import load_data, encode_words, j_dump
from nnerc_common import NetConfig, valdata_file, testdata_file
from keras.models import load_model
from keras_contrib.layers import CRF
from collections import OrderedDict
from pathlib import Path
import numpy as np
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_accuracy
import json
import os

def predict (modelname, testdir, outfilename):

    '''
    Loads a NN model from file  `modelname` and uses it to extract drugs
    in datadir . Saves results to `outfile` in the appropriate format.

    Parameters:
    -----------
    modelname: str

    testdir: pathlib.Path

    outfilename: str

    Returns:
    --------
    None
    '''
    
    version = int(modelname.split('_')[1])
    name = outfilename.split('.txt')[0]
    fmt = 'txt'
    outfilename = ('%s_%i.%s' % (name, version, fmt))

    # path to output file
    # test data is the validation data

    out_file = Path('./results/%s' % outfilename)

    # load model and associated encoding data

    model, idx = load_model_and_indexs(modelname)


    # ** UNCOMMENT THIS SECTION TO LOAD DATA AND STORE IT **

    # load data to annotate and save it in .json file
    # testdata = load_data(testdir)

    # ******************************************************

    # encode dataset

    X = encode_words(testdata_file, idx)

    # one-hot prediction

    Y = model.predict(X)
    
    # map one-hot prediction to tags prediction

    idx2tags = {v:k for k,v in idx['tags'].items()}
    
    Y = [[idx2tags[np.argmax(c)] for c in s] for s in Y]

    # extract entities and dump them to output file

    output_entities(testdata_file, Y, out_file)

    # evaluate using official evaluator .

    evaluation(testdir.__str__(), out_file.__str__())

    

def load_model_and_indexs(filename):
    '''
    Loads a model from filename.nn, and the indexs from
    filename.idx. Returns the loaded model and indexs
    
    Parameters:
    -----------
    filename: str

    Returns:
    --------    
    model: neural network

    idx: dict
    '''
    # filenames
    
    model_file = './models/%s.nn' % filename
    idx_file = './models/%s.idx' % filename

    # load the model

    model = load_model(model_file, custom_objects={'CRF':CRF, 
                                                   'crf_loss':crf_loss,
                                                   'crf_accuracy':crf_accuracy})

    # load the indexes
                                                      
    with Path(idx_file).open('r') as f:
        idx = json.load(f)
    
    return model, idx


def output_entities(testdata_file, Y, out_file):
    '''
    Receives a dataset produced by load data, and the
    corresponding tags predicted by the model. Prints the 
    detected entities in file outfilename in the appropriate 
    format for the evaluator: one line per entity.
        
    Parameters:
    -----------
    testdata_file: pathlib.Path

    Y: np.array
    
    outfilename: str

    Returns:
    --------
    None
    '''
    # extract data from .json file
    # load data as an ordered dict to get always same ordering

    with testdata_file.open('r') as f:
        testdata = json.load(f, object_pairs_hook=OrderedDict)

    # get the indexes of the entities in each sentence
     
    w_idx = [{i:value for i,value in enumerate(s_tags) if value not in ['<PAD>','O','<UNK>']} 
                for s_tags in Y ]

    with out_file.open('w') as f:
        for (s_id, v), e_d in zip(testdata.items(), w_idx):

            # get the entities in the sentence

            for index,bio_pred in e_d.items():

                entity = np.array(v)[index]
                pred = bio_pred.split('-')[-1]
                word = entity[0]
                offset = '%s-%s' % (str(entity[1]), str(entity[2]))
                # tag = entity[3] #real tag
                f.write('%s|%s|%s|%s' % (s_id,offset,word,pred))
                f.write('\n')



def evaluation (datadir, outfile):
    '''
    Receives a directory with ground truth data, and a file with
    entities extracted by the model. Runs the official evaluator and gets the results.
        
    Parameters:
    -----------
    datadir: str
    outfile: str

    Returns:
    --------
    '''

    java = Path('./eval/evaluateNER.jar')
    java = java.__str__()

    os.system("java -jar %s %s %s" % (java, datadir, outfile))

