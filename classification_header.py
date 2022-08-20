import pandas as pd
import string
import re
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from nltk.tokenize import sent_tokenize
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from tensorflow.keras import activations, regularizers, initializers, constraints
from keras.layers import Dense, Embedding, Bidirectional, add, Layer, Flatten, GlobalMaxPool1D, Convolution1D, \
GlobalAveragePooling1D,GlobalMaxPooling1D, Input, Conv1D, SpatialDropout1D, GRU, concatenate, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras import backend as K
import warnings
from gensim.models import Word2Vec

from collections import Counter
from imblearn.under_sampling import NearMiss
warnings.filterwarnings('ignore')

from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from sklearn.utils import class_weight

# Function for plotting confusion matrix and classification report
def plot_conf(ytest, ypred, name = "Confuction Matrix"):
    conf_mat = confusion_matrix(ytest, ypred)
    plt.figure(figsize=(17,4))
    # display number of false positives, false negatives, true positives, and true negatives respectively
    a = sns.heatmap(conf_mat, cmap="cividis", annot=True, annot_kws={"size": 9}, fmt=".0f")
    a.set_xlabel("prediction", fontsize = 15)
    a.set_ylabel("actual", fontsize = 15)
    a.set_title(name)

def vectorizor_util(xtrain, xtest, vectorizer):
    vectorizer.fit(list(xtrain) + list(xtest))
    return (vectorizer.transform(xtrain), vectorizer.transform(xtest))

def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x

def print_labelmapping(Y, Y_inv):
    label_mapping = {} # Store each unique mappings with key as string and value as encoded label
    for i in range(len(Y)):
        if Y_inv[i] in label_mapping: continue
        label_mapping[Y_inv[i]] = Y[i]
    label_mapping = sorted(label_mapping.items(), key = lambda item: item[1]) # Sort the label in ascending order
    for k, v in label_mapping: print("{}: {}".format(v, k), end = ' ')

# Common English Stop Words
stop_words = stopwords.words('english')

# Contraction and Greek Symbol mapping
corrections = {"aren't": "are not","can't": "cannot", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", 
               "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "isn't": "is not", "it's": "it is", 
               "°c" : "degree", "° c" : "degree",  "°": "degree", "needn't": "need not","should've": "should have", "shouldn't": 
               "should not", "that's": "that is", "there's": "there is", "they'll": "they will", "they're": "they are", "they've": 
               "they have", "wasn't": "was not", "weren't": "were not", "what're": "what are",  "what's": "what is", "where's": 
               "where is", "where've": "where have", "who'll": "who will", "who's": "who is", "why's": "why is", "will've":"will have",
               "won't": "will not", "would've": "would have", "wouldn't": "would not",  "α" : "alpha", "ν" : "nu",  "ξ" : "xi", "β" : 
               "beta", "γ" : "gamma", "δ" : "delta", "ε" : "epsilon", "ο" : "omicron", "π" : "pi", "ρ" : "rho", "ζ" : "zeta", "ς" : 
               "sigma", "σ" : "sigma", "η" : "eta", "τ" : "tau", "θ" : "theta", "υ" : "upsilon", "ι" : "iota", "φ" : "phi", "κ" : 
               "kappa", "χ" : "chi","λ" : "lambda", "ψ" : "psi", "ω" : "omega", "μ" : "mu"}

def build_mat(word_index):
    vocab_size = len(word_index) + 1
    embed = Word2Vec.load("mat2vec-master/mat2vec/training/models/pretrained_embeddings").wv.key_to_index
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embed.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape = (input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape = (input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    
def optimal_params(model, grid, name, xtrain, ytrain, xtest, ytest):
    cv_res = RandomizedSearchCV(model, param_distributions = grid, n_jobs = -1, cv = 5,
                                return_train_score = True, scoring = 'accuracy')
    cv_res.fit(xtrain, ytrain)
    print("optimal parameters for {}: {}".format(name, cv_res.best_params_))
    print("cross validation accuracy and validation set accuracy for {} respectively: {} and {}"
        .format(name, round(cv_res.best_score_, 3), round(accuracy_score(ytest, cv_res.predict(xtest)), 3)))
    return cv_res

def undersample_cv(clf, name, xtrain, ytrain, xtest, ytest):
    label_distribution = Counter(ytrain)
    tot = label_distribution[4]
    best_acc, opt_y_pred, perc, opt_perc = 0, None, 0.9, 0
    while perc > 0:
        label_distribution[4] = int(tot * perc)
        x_sam, y_sam = NearMiss(sampling_strategy = label_distribution,
                        n_jobs=-1, version=2, n_neighbors = 5).fit_resample(xtrain, ytrain)
        clf.fit(x_sam, y_sam)
        y_pred = clf.predict(xtest)
        cur_acc = round(accuracy_score(ytest, y_pred), 3)
        if not best_acc or cur_acc > best_acc:
            best_acc, opt_y_pred, opt_perc = cur_acc, y_pred, perc
        perc -= 0.1
    print("Best accuracy and best percentage for {} respectively: {} and {}".format(name, best_acc, opt_perc))
    return opt_y_pred

def plotnn_train_val(train_acc, val_acc):
    plt.figure()
    plt.title('Neural Network accuracy')
    plt.plot(train_acc, 'g*-', label = 'train')
    plt.plot(val_acc, 'ko-', label = 'validation')
    plt.legend()
    plt.show()