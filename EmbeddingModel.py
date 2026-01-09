################################################
## Helper functies voor het embedding model   ##
## Floris Menninga                            ##  
## Datum: 09-01-2026                          ##  
################################################


import io
import re
import string
import tqdm

import numpy as np
import requests
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow as tf
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
SEED = 0

class EmbeddingModel:


    def __init__(self, sequence_length=100, vocab_size=4096):
        self.sequence_length = sequence_length
        self.vocab_sizevocab_size = vocab_size


    def load_dataset(self):
        pass


    def create_vocab(self):
        pass


    def vectorize(self):
        pass


    def word2vec(self):
        pass


    def generate_trainingdata(self):
        pass

    def loss_function(self, x_logit, y_true):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

    def fit(self):
        pass

    def save2file(self):
        pass