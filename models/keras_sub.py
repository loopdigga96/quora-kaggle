
# coding: utf-8

# In[1]:

def submit(y_pred, test, filename):
    sub = pd.DataFrame()
    sub = pd.DataFrame()
    sub['test_id'] = test['test_id']
    sub['is_duplicate'] = y_test
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from nltk.corpus import stopwords
import gensim, logging
import json


# In[3]:

def create_word_to_vec(sentences, embedding_path, verbose=0, save=1, **params_for_w2v):
    if verbose:
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec(sentences, **params_for_w2v)
    
    if save:
        model.save(embedding_path)
    
    return model
    

def create_embeddings(sentences, embeddings_path='embeddings/embedding.npz', vocab_path='vocabs/map.json',
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

    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    with open(vocab_path, 'w') as f:
        f.write(json.dumps(vocab))

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

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

train = process_dataset(train)
train['union_splitted'] = train['union'].apply(lambda sentence: split_and_rem_stop_words(sentence))

#test = process_dataset(test)


# In[5]:

w2v = create_word_to_vec(sentences=train['union_splitted'], embedding_path='embeddings/w2v')


# In[6]:

max_num_words = train['union_splitted'].map(len).max()
len_x = len(train['union_splitted'])


# In[7]:

y_train = train.is_duplicate.tolist()


# In[8]:

tokenizer = Tokenizer(nb_words=max_num_words, split=' ')
tokenizer.fit_on_texts(train['union'])
sequences = tokenizer.texts_to_sequences(train['union'])

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X_train = pad_sequences(sequences, maxlen=max_num_words)

print('Shape of data tensor:', X_train.shape)


# In[12]:

weights = np.load(open('embeddings/embedding.npz', 'rb'))

embedding_layer = Embedding(input_dim=weights.shape[0], output_dim=100, weights=[weights], 
                            input_length=max_num_words, trainable=False)
# # embedding_layer = get_word2vec_embedding_layer('embeddings/embedding.npz')

model = Sequential()
model.add(embedding_layer)

model.add(Conv1D(16, 2, activation='relu'))
# model.add(MaxPooling1D(5))

model.add(Conv1D(32, 3, activation='relu'))
model.add(MaxPooling1D(2))

model.add(Conv1D(64, 4, activation='relu'))
model.add(MaxPooling1D(5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(X_train[:400000], y_train[:400000], batch_size=128, nb_epoch=1, 
          validation_data=(X_train[400000:], y_train[400000:]))


# In[10]:

X_train.shape


# In[ ]:



