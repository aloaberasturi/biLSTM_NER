#!usr/bin/python3
import json
import numpy as np
from pathlib import Path
from nnerc_utils import load_data, encode_words, j_dump, load_glove, classify_token, embedding_matrix, what_capital
from nnerc_common import traindata_file, valdata_file, NetConfig
from keras.preprocessing.text import Tokenizer
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils
from keras_contrib.metrics import crf_accuracy
from keras.activations import softmax
from keras_contrib.losses import crf_loss
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Nadam
from collections import OrderedDict
from nltk import pos_tag
from matplotlib import pyplot
from ast import literal_eval

def learn(traindir, validationdir, version):
    '''
    Learns a NN model using traindir as training data , and validationdir
    as validation data . Saves learnt model in a file named modelname
    
    Parameters:
    -----------
    traindir: pathlib.Path
        Path to train directory
    
    validationdir: pathlib.Path
        Path to validation directory

    version: int

    Returns:
    --------
    modelname: str
    '''

    modelname = "LSTM-CRF_%i" % version
    config = NetConfig(version)

    # ******** UNCOMMENT THIS SECTION TO LOAD DATA AND STORE IT *******

    # load the data from .xml files. This also stores the training and 
    # validation data in a permanent .json files

    # load_data(traindir)
    # load_data(validationdir)

    # *****************************************************************
    
    # create indexes from training data

    idx = create_indexs(traindata_file, max_len_sentences=75, max_len_words=50)

    # build network

    model = build_network(idx, config)

    # write on file containing the summary

    dump_summary(model, modelname)

    # encode datasets

    Xtrain = encode_words(traindata_file, idx)
    Ytrain = encode_tags(traindata_file, idx)
    Xval = encode_words(valdata_file, idx)
    Yval = encode_tags(valdata_file, idx)

    # train model and save it


    history = model.fit(Xtrain, Ytrain, validation_data=(Xval, Yval),
            verbose=1, 
            batch_size=config.batch, 
            epochs=config.epochs
            # callbacks=[EarlyStopping(
            #                     monitor='val_loss', 
            #                     patience=3, mode='min', 
            #                     restore_best_weights=True
            #                     )
                    # ]
            )
    
    # plot train and validation loss

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()

    # save indexes for later use in prediction

    save_model_and_indexs(model, idx, modelname)

    return modelname

def create_indexs(j_file, max_len_sentences, max_len_words):
    '''
    Receives a dataset produced by load data, and the maximum
    length in a sentence.
    Creates a set of words seen in the data and a set of BIO
    tags. Enumerates those sets, assigning a unique integer to each
    element. Returns these mappings in a single dictionary, with an
    additional entry for the given max length value.
    Other embeddings explored: lowercased words, non-lowercased words, 
    PoS tags and char embedding. This last one has a shape, for each 
    sentence in the set, of 

    Parameters:
    -----------
    j_file: pathlib.Path
        Path to .json file containing train data in dict form 

    max_len_sentences: int
        Maximum sentence length
    
    max_len_words: int
        Maximum word length


    Returns:
    --------
    dict
    '''
    # load data as an ordered dict to get always the same ordering

    # with j_file.open('r') as f:
    #     my_dict = json.load(f, object_pairs_hook=OrderedDict)
    with j_file.open('r') as f:
        my_dict = json.load(f)

    sentences = [[w[0] for w in s] for s in my_dict.values()]

    pos_list = list( set( [ pos[1] for s in sentences for pos in pos_tag(s) ] ) )
    w_list = list(set([w[0] for v in my_dict.values() for w in v]))
    chars_list = list(set([c for w in w_list for c in w ]))
    case_list = list(set([what_capital(w) for w in w_list]))
    type_list = list(set([classify_token(w) for w in w_list]))
    t_list = list(set([w[3] for v in my_dict.values() for w in v ]))    

    pos_d = {j: i + 2 for i,j in enumerate(pos_list)}
    pos_d['<PAD>'] = 0
    pos_d['<UNK>'] = 1

    words_d = {j: i + 2 for i,j in enumerate(w_list)}
    words_d['<PAD>'] = 0
    words_d['<UNK>'] = 1

    case_d = {j: i + 1 for i,j in enumerate(case_list)}# unknown is already considered
    case_d['<PAD>'] = 0

    type_d = {j : i + 1 for i,j in enumerate(type_list)} # unknown is already considered
    type_d['<PAD>'] = 0

    chars_d = {j : i + 1 for i,j in enumerate(chars_list)}
    chars_d['<PAD>'] = 0
    chars_d['<UNK>'] = 1

    tags_d  = {j: i + 1 for i,j in enumerate(t_list)}
    tags_d['<PAD>'] = 0

    return {'chars' : chars_d,
            'words' : words_d,
            'case'  : case_d,
            'pos'   : pos_d,
            'type'  : type_d,
            'tags'  : tags_d, 
            'max_len_sentences' : max_len_sentences, 
            'max_len_words'     : max_len_words}


def build_network(idx, config):
    '''
    Builds the nn. Receives the index dictionary with the encondings 
    of words and tags , and the maximum length of sentences

    Parameters:
    -----------
    idx: dict

    config: NetConfig instance
        Contains configuration of the neural network
        
    Returns:
    --------    
    model: neural network
    '''

    # sizes

    n_pos =             len(idx['pos'])             # UNK & PAD considered
    n_case =            len(idx['case'])            # PAD considered
    n_type =            len(idx['type'])            # PAD considered
    n_chars =           len(idx['chars'])           # UNK & PAD considered
    n_words =           len(idx['words'])           # UNK & PAD considered
    n_tags =            len(idx['tags'])            # PAD considered
    max_len_sentences = idx['max_len_sentences']   
    max_len_words =     idx['max_len_words'] 

    # ************************************************

    # architectural parameters

    pre_trained =       config.pre_trained
    w_embedding =       config.w_embedding
    c_embedding =       config.c_embedding
    lstm_char_units =   config.lstm_char_units
    lstm_main_units =   config.lstm_main_units
    dense_units =       config.dense_units
    return_sequences =  config.return_sequences
    mask_zero =         config.mask_zero
    activation =        config.activation

    #training parameters

    dropout =       config.dropout
    rcrr_dropout =  config.rcrr_dropout
    optimizer =     config.optimizer
    loss =          config.loss
    metrics =       config.metrics

    #********************************************************

    # create network layers

    # type embedding
    #---------------#

    type_inp = Input(shape=(max_len_sentences,))
    type_emb = Embedding(
        input_dim=n_type,
        output_dim=w_embedding,
        input_length=max_len_sentences,
        mask_zero=mask_zero)(type_inp)

    # pos embedding
    #--------------#

    pos_inp = Input(shape=(max_len_sentences,))
    pos_emb = Embedding(
        input_dim=n_pos,
        output_dim=w_embedding,
        input_length=max_len_sentences,
        mask_zero=mask_zero)(pos_inp)

    # capitalization words embedding
    #--------------------------#    

    case_inp = Input(shape=(max_len_sentences,))
    case_emb = Embedding(
        input_dim=n_case,
        output_dim=w_embedding,
        input_length=max_len_sentences,
        mask_zero=mask_zero)(case_inp)

    # word embedding
    # --------------#

    word_inp = Input(shape=(max_len_sentences,))
      
    if pre_trained: 

        #  word embedding option (1): load pre-trained embeddings 
        # and create the customized weights matrix according to our dataset

        word_emb = Embedding(
            input_dim=n_words, 
            output_dim=w_embedding,
            weights=[embedding_matrix(idx, n_words, w_embedding)], 
            trainable=False)(word_inp)

    else:

        # word embedding option (2): random embedding

        word_emb = Embedding(
        input_dim=n_words, 
        output_dim=w_embedding,
        input_length=max_len_sentences, 
        mask_zero=mask_zero)(word_inp)        


    #char embedding + char biLSTM
    #----------------------------

    char_inp = Input(shape=(max_len_sentences, max_len_words)) 

    char_emb = TimeDistributed(
                    Embedding(
                        input_dim=n_chars,
                        output_dim=c_embedding,
                        input_length=max_len_words,
                        mask_zero=mask_zero)
                    )(char_inp)  

    char_biLSTM = TimeDistributed(
                    Bidirectional(LSTM(
                    units=lstm_char_units, 
                    return_sequences=False,
                    recurrent_dropout=rcrr_dropout, 
                    dropout=dropout))
                    )(char_emb) 
    
    # main LSTM
    #---------#

    model = concatenate([
        word_emb, 
        char_biLSTM,
        case_emb,
        # pos_emb, 
        # type_emb
        ]
    )


    # model = Dropout(dropout)(model)

    model = Bidirectional(LSTM(units=lstm_main_units, return_sequences=return_sequences,
                recurrent_dropout=rcrr_dropout, dropout=dropout))(model)

    model = TimeDistributed(Dense(units=dense_units, activation=activation))(model) 

    model = TimeDistributed(Dense(units=n_tags, activation=activation))(model) 
    
    # model = Dropout(dropout)(model)

    # CRF layer
    #----------

    crf = CRF(n_tags)

    out = crf(model)               
    
    # create and compile model

    model = Model([
        word_inp, 
        char_inp,
        case_inp,
        # pos_inp, 
        # type_inp, 
        
        ], out)

    
    if str.lower(optimizer) == 'nadam': 
        optimizer = Nadam()

    model.compile(optimizer=optimizer, loss=crf_loss, metrics=[crf_accuracy])


    return model

def dump_summary(model, modelname):
    '''
    Receives a model and its name. Calls keras' model.summary() and 
    dumps it into a txt file.

    Parameters:
    -----------
    model: neural network

    modelname: str

    Returns:
    --------
    None
    '''
    
    with open('./models/%s_summary.txt' % modelname,'w') as mf:
    
        # Pass the file handle in as a lambda function to make it callable    

        model.summary(print_fn=lambda s: mf.write(s + '\n'))

def encode_tags(j_file, idx):
    '''
    Receives a dataset produced by load data, and the index
    dictionary produced by create indexs. Returns the dataset 
    as a list of sentences. Each sentence is a list of integers,
    corresponding to the code of the BIO tag for each word. 
    If the sentence is shorter than max len it is padded with the
    code for <PAD>.

    Parameters:
    -----------
    j_file: pathlib.Path
        path to .json file containing the train data
    
    idx: dict
        Dictionary with the indexes returned by create_indexs

    Returns:
    --------
    np.array: list of encoded sentences
    '''

    # load data as an ordered dict to get always same ordering

    with j_file.open('r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    # convert the tags to integers   
    Y = [
            [
            idx['tags'][e[3]] for e in entities
            ] for entities in data.values()
        ]

    # if the sentence is shorter than max_len, we pad it 

    Y = pad_sequences(maxlen=idx['max_len_sentences'], sequences=Y, 
                        padding='post', value=idx['tags']['<PAD>'])

    # one-hot encoding
    
    return np.array([to_categorical(i, num_classes=len(idx['tags'])) for i in Y])


def save_model_and_indexs(model, idx, filename):
    '''
    Receives a trained model, an index dictionary, and a string.
    Stores the model in a file named filename.nn, and the
    indexs in a file named filename.idx

    Parameters:
    -----------
    model: Trained neural network

    idx: dict

    filename: str

    Returns:
    --------
    None
    '''

    path_to_model = Path('./models/%s.nn' % filename)
    path_to_indxs = Path('./models/%s.idx'% filename)

    # save model
    model.save(path_to_model.__str__())

    # save indexes
    with path_to_indxs.open('w') as f:
        json.dump(idx,f)
