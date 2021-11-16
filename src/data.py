import os
import pickle as pkl
import pandas as pd
import numpy as np
import gluonnlp as nlp
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
import nltk
nltk.download('punkt')

class SwDA:

  def __init__(self, path='swda/', number_conversations=30):

    self.C = [] #conversations
    self.Y = [] #list of list of tags
    self.Callers = [] #list of list of callers
    self.tags = set()

    count = 0
    for path_conv in os.listdir(path):
      p = '/'.join((path, path_conv))
      if os.path.isdir(p):
        for csv in os.listdir(p):
          if count < number_conversations:
            if 'utt.csv' in csv:
              pp = '/'.join((path, path_conv, csv))
              df = pd.read_csv(pp)[['act_tag', 'caller', 'text']]
              tags = pd.unique(df['act_tag'])
              for tag in tags:
                self.tags.add(tag)
              self.Y.append(df['act_tag'].values)
              self.C.append(df['text'].values)
              self.Callers.append(df['caller'].values)
          count += 1
    
  def serialize_conversations(self, T=5**2):
    #serialize the raw data with a window of length T and clean the utterances 

    self.C_serialized = [] 
    self.Y_serialized = []
    self.Callers_serialized = []
    self.words = set()

    for i in range(len(self.C)):
      conv = []
      tags = []
      callers = []


      for t in range(T, len(self.C[i])):
        seq = self.C[i][t - T: t]
        seq_clean = []
        for utt in seq:
          words_utt = word_tokenize(utt)[-1]
          seq_clean.append(words_utt)
          for word in words_utt:
            self.words.add(word)
        conv.append(seq_clean)
        callers.append(self.Callers[i][t - T: t])
        tags.append(self.Y[i][t - T: t])

      self.C_serialized += conv
      self.Y_serialized += tags
      self.Callers_serialized += callers
      
    self.C_serialized = np.asarray(self.C_serialized, dtype=object)
    self.Y_serialized = np.asarray(self.Y_serialized, dtype=object)
    self.Callers_serialized = np.asarray(self.Callers_serialized, dtype=object)
  
  def embedding_pad(self, T=5**2, max_length=20):
    #pad or make a troncature to have max_length words in each sequence
    #embedding with the fasttext embeddings layer
    # tab : array (number_sequence, T, max_length, 300)

    #set fasttext
    counter = nlp.data.count_tokens([word for word in self.words])
    self.vocab = nlp.Vocab(counter)
    fasttext_simple = nlp.embedding.create('fasttext', source='wiki.simple')
    self.vocab.set_embedding(fasttext_simple)

    tab = np.zeros((self.C_serialized.shape[0], T, max_length, 300))
    for i in range(self.C_serialized.shape[0]):
      for j in range(T):
        for k in range(len(self.C_serialized[i][j])):
          tab[i, j, k, :] = self.vocab.embedding[self.C_serialized[i][j][k]].asnumpy()
        tab[i, j, k+1:, :] = self.vocab.embedding['<pad>'].asnumpy()
    
    self.C_serialized = tab
  
  def one_hot_tags(self, T=5**2):
    #one hot encoding of tags
    
    tab = np.zeros((self.Y_serialized.shape[0], T, len(self.tags)))
    self.tags_dict = dict((tag, i) for (i, tag) in enumerate(self.tags))

    for i in range(self.Y_serialized.shape[0]):
      for j in range(T):
        tab[i, j, self.tags_dict[self.Y_serialized[i][j]]] = 1.0
    self.Y_serialized = tab

  def callers_process(self):
    #code by 1 and 0 callers 1 and B 

    self.Callers_serialized = np.where(self.Callers_serialized == 'A', 1.0, 0.0)

  def preprocess(self, T=5**2, max_length=20):
    self.serialize_conversations(T)
    self.embedding_pad(T, max_length)
    self.one_hot_tags(T)
    self.callers_process()

    #shuffle and split

    self.callers_train, self.callers_test, self.conv_train, self.conv_test, self.y_train, self.y_test = train_test_split(self.Callers_serialized, 
                                                                                                                         self.C_serialized, self.Y_serialized, test_size=.5, shuffle=True)
  
  def get_raw_data(self):
    #return the raw data

    return self.Callers, self.C, self.Y
  
  def get_data(self):
    #return serialized utterances : callers for each utterance, utterances, tags

    return self.callers_train, self.callers_test, self.conv_train, self.conv_test, self.y_train, self.y_test
  
  def save(self, path):
    with open(path, 'wb') as f:
      pkl.dump((self.callers_train, self.callers_test, self.conv_train, self.conv_test, self.y_train, self.y_test), f, protocol=4)
    f.close()
  
  def load(self, path):
    with open(path, 'rb') as f:
      self.callers_train, self.callers_test, self.conv_train, self.conv_test, self.y_train, self.y_test = pkl.load(f)
    f.close()