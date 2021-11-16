import keras
import tensorflow as tf
from keras import Model
from keras.layers import Bidirectional, GRU, Lambda, Concatenate, Reshape
from tensorflow.keras.backend import squeeze, repeat_elements
from keras.regularizers import l2

#Vanilla RNN encoder

class VGRUe(Model):

  def __init__(self):
    super(VGRUe, self).__init__()
    
    self.mean_layer = Lambda( lambda x: tf.math.reduce_mean(x, axis=3))
    self.concatenate = Concatenate(axis=1)
    self.biGRU_1 = Bidirectional(GRU(128, return_sequences=True, dropout=0.2, kernel_regularizer=l2(1e-5)))
    self.biGRU_2 = Bidirectional(GRU(64, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5)))

  def call(self, x): # x is a list of two elements: Callers and seq of Utterances

    x = x[1]
    embedding_means = self.mean_layer(x)
    seq = self.biGRU_1(embedding_means)
    seq, state_h, state_c = self.biGRU_2(seq)

    return seq, self.concatenate([state_h, state_c])


#Hierarchical encoders

class HGRU(Model):

  def __init__(self):
    super(HGRU, self).__init__()

    self.word_seqs = Lambda( lambda x: tf.split(x, [1 for i in range(5**2)], axis=1))
    self.concatenate = Concatenate(axis=1)
    self.reshape = Reshape((1, 256))

    self.biGRU_words = Bidirectional(GRU(128, return_sequences=False, return_state=False, dropout=0.2, kernel_regularizer=l2(1e-5)))
    self.biGRU_utt = Bidirectional(GRU(64, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5)))
  
  def call(self, x):

    x = x[1]
    utts = self.word_seqs(x)
    words_dependencies = []
    for utt in utts:
      h = self.biGRU_words(squeeze(utt, axis=1))
      words_dependencies.append(self.reshape(h))
    words_dependencies = self.concatenate(words_dependencies)
    seq, state_h, state_c = self.biGRU_utt(words_dependencies)

    return seq, self.concatenate([state_h, state_c])


#Persona hierarchical encoders

class PersoHGRU(Model):

  def __init__(self):
    super(PersoHGRU, self).__init__()

    self.word_seqs = Lambda( lambda x: tf.split(x, [1 for i in range(5**2)], axis=1))
    self.speaker_differentes_right = Lambda( lambda x: tf.math.abs(tf.subtract(x, tf.roll(x, shift=1, axis=1))))
    self.speaker_differentes_left = Lambda( lambda x: tf.math.abs(tf.subtract(x, tf.roll(x, shift=24, axis=1))))
    self.concatenate = Concatenate(axis=1)
    self.reshape = Reshape((1, 256))

    self.biGRU_words = Bidirectional(GRU(128, return_sequences=False, return_state=False, dropout=0.2, kernel_regularizer=l2(1e-5)))
    self.GRU_persona_left = GRU(128, return_sequences=False, return_state=False, dropout=0.2, kernel_regularizer=l2(1e-5))
    self.GRU_persona_right = GRU(128, return_sequences=False, return_state=False, dropout=0.2, kernel_regularizer=l2(1e-5))
    self.biGRU_utt = Bidirectional(GRU(64, return_sequences=True, return_state=True, dropout=0.2, kernel_regularizer=l2(1e-5)))
  
  def call(self, x):

    c = x[0]
    c_right = repeat_elements(Reshape((25, 1))(self.speaker_differentes_right(c)), rep=256, axis=-1)
    c_left = repeat_elements(Reshape((25, 1))(self.speaker_differentes_left(c)), rep=256, axis=-1)
    u = x[1]
    utts = self.word_seqs(u)
    words_dependencies = []
    for utt in utts:
      h = self.biGRU_words(squeeze(utt, axis=1))
      words_dependencies.append(self.reshape(h))
    words_dependencies = self.concatenate(words_dependencies)
    persona_left = c_left * words_dependencies
    persona_right = c_right * words_dependencies
    p_left = self.word_seqs(persona_left)
    p_right = self.word_seqs(persona_right)
    persona = []
    for t in range(5**2):
      l = self.GRU_persona_left(p_left[t])
      r = self.GRU_persona_right(p_right[t])
      persona.append(Reshape((1, 256))(self.concatenate([l, r])))
    persona = self.concatenate(persona)
    seq, state_h, state_c = self.biGRU_utt(persona)

    return seq, self.concatenate([state_h, state_c])