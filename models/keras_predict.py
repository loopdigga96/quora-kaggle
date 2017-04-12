
# coding: utf-8

# In[1]:

# fit word2vec on full/test questions
# fit tokenizer on full/test questions


# In[2]:

import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from nltk.corpus import stopwords
import gensim, logging
import json

import os.path

MAX_NUM_WORDS = 125


# In[3]:

def submit(y_pred, test, filename):
    sub = pd.DataFrame()
    sub = pd.DataFrame()
    sub['test_id'] = test['test_id']
    sub['is_duplicate'] = y_pred
    sub.to_csv(filename, index=False)

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def correct_dataset(dataset):
    dataset.loc[(dataset['question1'] == dataset['question2']), 'is_duplicate'] = 1
    return dataset

def process_dataset(dataset, correct_dataset=False):
    dataset['question1'].fillna(' ', inplace=True)
    dataset['question2'].fillna(' ', inplace=True)
    
    #delete punctuation
    dataset['question1'] = dataset['question1'].str.replace('[^\w\s]','')
    dataset['question2'] = dataset['question2'].str.replace('[^\w\s]','')

    #lower questions
    dataset['question1'] = dataset['question1'].str.lower()
    dataset['question2'] = dataset['question2'].str.lower()

    #union questions
    dataset['union'] = pd.Series(dataset['question1']).str.cat(dataset['question2'], sep=' ')

    if correct_dataset:
        return correct_dataset(dataset)
    else:
        return dataset

def split_and_rem_stop_words(line):
    cachedStopWords = stopwords.words("english")
    return [word for word in line.split() if word not in cachedStopWords]

def create_word_to_vec(sentences, embedding_path, verbose=0, save=1, **params_for_w2v):
    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec(sentences, **params_for_w2v)
    
    if save:
        model.save(embedding_path)
    
    return model
    

def create_embeddings(sentences, embeddings_path='embeddings/embedding.npz',
                      verbose=0, **params):
    """
    Generate embeddings from a batch of text
    :param embeddings_path: where to save the embeddings
    :param vocab_path: where to save the word-index map
    """

    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec(sentences, **params)
    weights = model.wv.syn0
    np.save(open(embeddings_path, 'wb'), weights)


def load_vocab(vocab_path):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def get_word2vec_embedding_layer(embeddings_path):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """

    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights],
                     trainable=False)
    return layer


# In[4]:

#Load train
print 'Loading datasets'


if os.path.isfile('dataframes/train.h5'):
    train = pd.read_pickle('dataframes/train.h5')
else:
    train = pd.read_csv('../datasets/train.csv')
    train = process_dataset(train)
    train['union_splitted'] = train['union'].apply(lambda sentence: split_and_rem_stop_words(sentence))
    train.to_pickle('dataframes/train.h5')


# In[5]:

# Load test

if all([os.path.isfile('dataframes/test_0.h5'), os.path.isfile('dataframes/test_1.h5'),
        os.path.isfile('dataframes/test_2.h5'), os.path.isfile('dataframes/test_3.h5')]):
    
    test = pd.read_csv('../datasets/test.csv')
    test = process_dataset(test)
    
#     test_0 = pd.read_pickle('dataframes/test_0.h5')
#     test_1 = pd.read_pickle('dataframes/test_1.h5')
#     test_2 = pd.read_pickle('dataframes/test_2.h5')
#     test_3 = pd.read_pickle('dataframes/test_3.h5')

#     test_0.columns = ['union_splitted']
#     test_1.columns = ['union_splitted']
#     test_2.columns = ['union_splitted']
#     test_3.columns = ['union_splitted']

#     test_full_splitted = test_0.append(
#                          test_1.append(
#                          test_2.append(
#                          test_3)))

#     test['union_splitted'] = test_full_splitted['union_splitted'].values
else:
    print 'Not enough files for test'


# In[ ]:

print 'Tokenizing'
#Tokenize test

tokenizer = Tokenizer(nb_words=MAX_NUM_WORDS, split=' ')
tokenizer.fit_on_texts(train['union'])
sequences = tokenizer.texts_to_sequences(test['union'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_test = pad_sequences(sequences, maxlen=MAX_NUM_WORDS)

print('Shape of data tensor:', X_test.shape)


# In[9]:

#Load model

model = load_model('keras_models/new_model_6_epochs.h5')


# In[ ]:

#predict

y_preds = model.predict(X_test, batch_size=128, verbose=1)

submit(y_preds, test, '../submissions/keras_6_epochs_1_dropout.csv')