#!usr/bin/python3

import xml.etree.ElementTree as ET
import json
import spacy
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from nltk import pos_tag
from nnerc_common import traindata_file, valdata_file, testdata_file, traindir, validationdir, glove_file
from collections import OrderedDict

tokenizer = TreebankWordTokenizer()

def embedding_matrix(idx, n_words, w_embedding):
    """
    Creates custom weight matrix using pre-trained GloVe vectos

    Parameters:
    -----------
    idx : dict

    n_words : int

    word_emb : int

    Returns:
    --------
    embedding_matrix : np.array
    """

    embeddings_index = load_glove()
    embedding_matrix = np.zeros((n_words, w_embedding)) 
    for word, i in idx['words'].items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:                
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def classify_token(w):
    """
    Function for feature augmentation. Improves the network's results by generating
    the information that will be fed into the feature-augmentend embedding layer

    Parameters
    ----------
    w : str
        Word
    
    Returns
    -------
    str
        Class to which a token pertains by basic feature extraction
    """

    suffixes = ["azole", "idine", "amine", "mycin", "xacin", "ostol", "adiol"]
    suffixes_drug = ["ine", "cin", "ium", "ide"]
    suffixes_brand = ["gen"]
    suffixes_group = ["etic","tics", "acid", "tors", "cids"]
    pref_group = ['anta','thia','ace','diur']

    # if w.isupper() or str.lower(w[-3:]) in suffixes_brand: 
    #     return "brand"
    # elif str.lower(w[-5:]) in suffixes or str.lower(w[-3:]) in suffixes_drug: 
    #     return "drug"
    if str.lower(w[-4:]) in suffixes_group or "agent" in str.lower(w) or str.lower(w[:4]) in pref_group:
        return "group"
    else : 
        return "other"

def what_capital(w):
    """
    Receives a word and returns its capitalization. If it's numerical,
    returns a "contains_digit" flag.

    Parameters:
    -----------
    w : str
        Token
    
    Returns:
    --------
    str
        Flag
    """

    if ( True in [c.isdigit() for c in w] and not w.isdigit() ): 
        return 'contains_numeric'

    elif w.isupper(): 
        return 'all_upper'

    elif w.islower(): 
        return 'all_lower'

    elif w.istitle(): 
        return 'initial_upper'

    elif w.isdigit(): 
        return 'numeric'

    else: 
        return 'other'


def bio_tags(off, entity_info):
    """
    Creates list with BIO tags in a sentence.

    Parameters:
    -----------
    xml_sent: sentence in xml format

    Returns:
    --------
    list of BIO tags
    """

    for (span, e_type) in entity_info :

        e_start, e_end = span[0], span[1]

        if off[0] == e_start and off[1] <= e_end: 
            return "B-%s" % e_type

        elif off[0] >= e_start and off[1] <= e_end: 
            return "I-%s" % e_type

    return "O"


def load_data(datadir):
    '''
    Receives a directory containing XML files.
    Parses XML files in given directory, tokenizes each
    sentence, extracts ground truth BIO tags for each token, and
    returns the dataset as a dictionary. Dictionary keys are the
    sentence id, and values are the list of token tuples (word, start,
    end, ground truth).

    Parameters:
    -----------
    datadir: pathlib.Path
        Directory with xml files

    Returns:
    --------
    ner_d : dict
        Dictionary with sentence id as keys and
        (token, begin, end, BIO tag) as values
    '''
    ner_d = {}
    for file in datadir.iterdir():
        
        group_suff = []
        drug_suff = []
        drug_n_suff = []
        brand_suff = []
        tree = ET.parse(file.__str__())
        root = tree.getroot()

        for sent in root.iter('sentence'):

            sentence = sent.get('text')
            s_id = sent.get('id')
            entities_info = []

            for entity in sent.iter('entity'):

                offset = [int(o) for o in entity.get('charOffset').split(";")[0].split('-')]
                
                entities_info.append((offset, entity.get('type')))

            # TreebankWordTokenizer().span_tokenize returns the beginning of the word 
            # and its length. We have to substract one unit to the length to get the
            # position of the word ending.

            tokens = tokenizer.tokenize(sentence)
            offsets = tokenizer.span_tokenize(sentence)
            offsets = [(i[0],i[1] - 1) for i in offsets]
            b_tags = [bio_tags(off, entities_info) for off in offsets]

            for i,j,k in zip(tokens, offsets, b_tags):
                values = (i,j[0],j[1],k) 
                if s_id not in ner_d.keys():
                    ner_d[s_id] = []
                ner_d[s_id].append(values)   

    j_dump(ner_d, 'train' if datadir == traindir else 'devel')  
    
    return ner_d    
                                  
def encode_words(j_file, idx):
    '''Receives a dataset produced by load_data, and the index
    dictionary produced by create_indexs. Returns the dataset as a list 
    of sentences. Each sentence is a list of integers, corresponding to
    the code of the BIO tag for each word. If the sentence is shorter
    than max len it is padded with the code for <PAD>.

    Parameters:
    -----------
    dataset: pathlib.Path
        path to .json file containing the train data
    
    idx: dict
        Dictionary with the indexes returned by create_indexs

    Returns:
    --------
    sentences: list of lists
    '''
    # load data as an ordered dict to get always same ordering
    # this is important in nnerc_classifier.py because we 
    # load data twice: once here and after in `output_entities()`

    with j_file.open('r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    
    sentences = [[w[0] for w in s] for s in data.values()]

    # *********************** EMBEDDINGS **************************#

    # PoS embedding

    X_pos = [
        [
        idx['pos'][p[1]] if p[1] in idx['pos'].keys()
        else idx['pos']['<UNK>'] for p in pos_tag(s)
        ] for s in sentences
    ]

    # convert the sentences of words to lists of integers  

    X_words = [
        [
        idx['words'][e[0]]  if e[0] in idx['words'].keys() 
        else idx['words']['<UNK>'] for e in entities
        ] for entities in data.values()
    ]

    # feature augmentation
        # 1) capitalization
 
    X_capitalization = [
        [
            idx['case'][what_capital(e[0])] for e in entities
        ] for entities in data.values()
    ]
     
        # 2) token type embedding 

    X_type = [
        [
        idx['type'][classify_token(e[0])] for e in entities
        ] for entities in data.values()
    ]

    # char embedding

    temp = [
                [
                    [
                        idx['chars'][c]  if c in idx['chars'].keys() 
                        else idx['chars']['<UNK>'] for c in word[0]
                    ] for word in entities 
                ]for entities in data.values()
            ]

    # *************************** PADDINGS ******************************# 
            
    X_type = pad_sequences(maxlen=idx['max_len_sentences'], sequences=X_type, 
                        padding='post', value=idx['type']['<PAD>'])

    X_pos = pad_sequences(maxlen=idx['max_len_sentences'], sequences=X_pos, 
                        padding='post', value=idx['pos']['<PAD>'])

    X_words = pad_sequences(maxlen=idx['max_len_sentences'], sequences=X_words, 
                        padding='post', value=idx['words']['<PAD>'])

    X_capitalization = pad_sequences(maxlen=idx['max_len_sentences'], sequences=X_capitalization, 
                        padding='post', value=idx['case']['<PAD>'])

    # chars require special embedding: 
    # X_chars.shape = (# dataset_len, # max_len_sentences, # max_len_words)

    X_chars = np.zeros((len(temp), idx['max_len_sentences'],idx['max_len_words'])).astype(np.dtype)

    for i, sent in enumerate(temp):
        for j, word in enumerate(sent):
            if j < idx['max_len_words']:
                X_chars[i, j, :len(word)] = word

    # return embeddings 

    return[X_words,
        X_chars,        
        X_capitalization,
        # X_pos,
        # X_type,         
        ]


def j_dump(data, dataset): 
    '''
    Stores datadict into .json file.

    Parameters:
    -----------
    data: dict
    dataset: str

    Returns:
    --------
    None
    '''

    # assign json_path 

    if dataset == 'train':
        json_path = traindata_file

    elif dataset == 'devel':
        json_path = valdata_file

    elif dataset == 'test':
        json_path = testdata_file

    with json_path.open('w') as f:

        #dump into file

        json.dump(data, f)


def load_glove():

    """Loads GloVe vectors in numpy array.
    Args:
        file (str): a path to a glove file.
    Return:
        dict: a dict of numpy arrays.
    """
    model = {}
    with open(glove_file.__str__(), 'r', encoding='utf') as f:
        for line in f:
            line = line.split(' ')
            word = line[0]
            vector = np.asarray(line[1:], dtype='float32')
            model[word] = vector

    return model