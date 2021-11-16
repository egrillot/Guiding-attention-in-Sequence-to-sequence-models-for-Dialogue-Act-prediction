import keras
import tensorflow as tf
from keras import Model
from keras.layers import GRU, Dense, Concatenate, Reshape, Lambda
from keras.regularizers import l2
from keras.activations import softmax


#Vanilla decoder

class VGRUd(Model):

  def __init__(self, encoder):
    super(VGRUd, self).__init__()

    self.encoder = encoder #type Model
    self.gru_decoder = GRU(64, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5))
    self.dense_softmax = Dense(78, activation='softmax', kernel_regularizer=l2(1e-5)) #78 tags
    self.concatenate = Concatenate(axis=1)

  def call(self, x):

    seq, H = self.encoder(x)
    input = Reshape((1, 128))(x[1][:, 0, 0, :128])
    state = Reshape((64, ))(H)
    outputs = []
    for t in range(5**2):
      input, state = self.gru_decoder(input, initial_state=state)
      outputs.append(input)
    outputs = self.concatenate(outputs)

    return self.dense_softmax(outputs)

#Decoders with attention

#Vanilla attention

class VGRUatt(Model):

  def __init__(self, encoder):
    super(VGRUatt, self).__init__()
  
    self.encoder = encoder 
    self.seqs = Lambda( lambda x: tf.split(x, [1 for i in range(5**2)], axis=1))
    self.gru_decoder = GRU(128, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5))
    self.dense_softmax = Dense(78, activation='softmax', kernel_regularizer=l2(1e-5)) #78 tags
    self.concatenate = Concatenate(axis=1)
    self.dense_attention = Dense(128, activation=None, kernel_regularizer=l2(1e-5)) 
  
  def call(self, x):

    seq, H = self.encoder(x)

    hs = self.seqs(seq)
    input = Reshape((1, 128))(x[1][:, 0, 0, :128])
    state = Reshape((128, ))(H)
    outputs = []
    for t in range(5**2):
      input, state = self.gru_decoder(input, initial_state=state)
      alphas = []
      for i in range(5**2):
        score = self.dense_attention(hs[i])
        alphas.append(tf.math.reduce_sum(tf.multiply(score, input), axis=2))
      alphas = self.concatenate(alphas)
      alphas = softmax(alphas, axis=-1)
      alphas = self.seqs(alphas)
      state = tf.multiply(Reshape((1, 1))(alphas[0]), hs[0])
      for i in range(1, 5**2):
        state += tf.multiply(Reshape((1, 1))(alphas[i]), hs[i])
      state = Reshape((128, ))(state)
      outputs.append(input)
    outputs = self.concatenate(outputs)

    return self.dense_softmax(outputs)

#Hard guided attention

class VGRUhga(Model):

  def __init__(self, encoder):
    super(VGRUhga, self).__init__()

    self.encoder = encoder 
    self.seqs = Lambda( lambda x: tf.split(x, [1 for i in range(5**2)], axis=1))
    self.gru_decoder = GRU(128, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5))
    self.dense_softmax = Dense(78, activation='softmax', kernel_regularizer=l2(1e-5)) #78 tags
    self.concatenate = Concatenate(axis=1)

  def call(self, x):

    seq, H = self.encoder(x)

    hs = self.seqs(seq)
    input = Reshape((1, 128))(x[1][:, 0, 0, :128])
    state = Reshape((128, ))(H)
    input, state = self.gru_decoder(input, initial_state=state)
    outputs = [input]
    for t in range(1, 5**2):
      input, state = self.gru_decoder(input, initial_state=state)
      state = Reshape((128, ))(hs[t])
      outputs.append(input)
    outputs = self.concatenate(outputs)

    return self.dense_softmax(outputs)
  
#Soft guided attention

class VGRUsga(Model):

  def __init__(self, encoder):
    super(VGRUsga, self).__init__()
  
    self.encoder = encoder 
    self.seqs = Lambda( lambda x: tf.split(x, [1 for i in range(5**2)], axis=1))
    self.gru_decoder = GRU(128, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5))
    self.dense_softmax = Dense(78, activation='softmax', kernel_regularizer=l2(1e-5)) #78 tags
    self.concatenate = Concatenate(axis=1)
    self.dense_attention = Dense(128, activation=None, kernel_regularizer=l2(1e-5)) 
  
  def call(self, x):

    seq, H = self.encoder(x)

    hs = self.seqs(seq)
    input = Reshape((1, 128))(x[1][:, 0, 0, :128])
    state = Reshape((128, ))(H)
    outputs = []
    for t in range(5**2):
      input, state = self.gru_decoder(input, initial_state=state)
      alphas = []
      for i in range(5**2):
        score = self.dense_attention(hs[i])
        a = tf.math.reduce_sum(tf.multiply(score, input), axis=2)
        if i == t:
          alphas.append(tf.add(1.0, a))
        else:
          alphas.append(a)
      alphas = self.concatenate(alphas)
      alphas = softmax(alphas, axis=-1)
      alphas = self.seqs(alphas)
      state = tf.multiply(Reshape((1, 1))(alphas[0]), hs[0])
      for i in range(1, 5**2):
        state += tf.multiply(Reshape((1, 1))(alphas[i]), hs[i])
      state = Reshape((128, ))(state)
      outputs.append(input)
    outputs = self.concatenate(outputs)

    return self.dense_softmax(outputs)