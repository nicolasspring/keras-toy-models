#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################
# Imports
########################################################################

from scoring_relation import evaluate

import copy
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Conv1D, GlobalMaxPooling1D,\
                         Concatenate, Dropout
from keras.callbacks import EarlyStopping

########################################################################
# Constants
########################################################################

NUM_CLASSES = 19

########################################################################
# Data Class
########################################################################

class Data(object):

    def __init__(self):
        # DataFrames
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()

        # the actual data
        self._x_train = np.array([])
        self._x_dev = None
        self._x_test = np.array([])
        self._y_train = np.array([])
        self._y_dev = None
        self._train_ids = []
        self._dev_ids = None
        self._test_ids = []

        # parameters influencing the creation of the data
        self._dev_frac = None
        self._max_len = None
        self._shuffle = None
        self._label_smooth_frac = None

        # Tokenizer and LabelEncoder instances
        self._tokenizer = Tokenizer()
        self._label_encoder = LabelEncoder()

    def prepare_data(self, train_tsv, test_tsv,
                     dev_frac=None,
                     max_len=200,
                     shuffle=True,
                     label_smooth_frac=None,
                     tokenizer=None):
        '''Prepares training and test data by converting sentences into
        sequences of numbers and labels into categorical representations.

        Args:
            train_tsv:          path to a tsv file containing training
                                    data
            test_tsv:           path to a tsv file containing test data
            dev_frac:           fraction of the training data split off
                                    and used as the development set
            max_len:            maximal sentence lenght all sentences
                                    are padded to
            shuffle:            whether or not to shuffle the training
                                    data
            label_smooth_frac:  fraction of probability space distributed
                                    from the true example to the others
            tokenizer:          an instance of 
                                    keras.preprocessing.text.Tokenizer
        '''
        self._dev_frac = dev_frac
        self._max_len = max_len
        self._shuffle = shuffle
        self._label_smooth_frac = label_smooth_frac
        if tokenizer:
            self._tokenizer = tokenizer

        self._df_train = self._read_train_tsv(train_tsv, shuffle=self._shuffle)
        x_train, train_ids = self._prepare_data_x_train(self._df_train)
        y_train = self._prepare_data_y(self._df_train)
        self._split_save_train_dev(x_train, y_train, train_ids)

        self._df_test = self._read_test_tsv(test_tsv)
        x_test, test_ids = self._prepare_data_x_test(self._df_test)
        self._save_test(x_test, test_ids)

        return True

    def _read_train_tsv(self, train_tsv, shuffle=True):
        '''reads the tsv containing the training data into a DataFrame'''
        df = pd.read_csv(train_tsv,
                         delimiter='\t',
                         names=['id', 'sentence', 'label'],
                         converters={'id': lambda x: str(x)})
        if shuffle:
            df = df.sample(frac=1)
        return df

    def _read_test_tsv(self, test_tsv):
        '''reads the tsv containing the test data into a DataFrame'''
        df = pd.read_csv(test_tsv,
                         delimiter='\t',
                         names=['id', 'sentence'],
                         converters={'id': lambda x: str(x)})
        return df

    def _prepare_data_x_train(self, df):
        '''fits the Tokenizer, converts sentences into sequences of 
        equal lenght and returns them as well as their ids'''
        ids = [row['id'] for _, row in df.iterrows()]
        x_sents = [row['sentence'] for _, row in df.iterrows()]
        self._tokenizer.fit_on_texts(x_sents)
        x_sents = [sequence
                   for sequence
                   in self._tokenizer.texts_to_sequences_generator(x_sents)]
        return pad_sequences(x_sents, maxlen=self._max_len), ids

    def _prepare_data_x_test(self, df):
        '''converts sentences into sequences of equal lenght and returns
        them as well as their ids'''
        ids = [row['id'] for _, row in df.iterrows()]
        x_sents = [row['sentence'] for _, row in df.iterrows()]
        x_sents = [sequence
                   for sequence
                   in self._tokenizer.texts_to_sequences_generator(x_sents)]
        return pad_sequences(x_sents, maxlen=self._max_len), ids

    def _prepare_data_y(self, df):
        '''fits the LabelEncoder and converts the labels into categorical
        representations adding label smoothing'''
        y_labels = [row['label'] for _, row in df.iterrows()]
        self._label_encoder.fit(y_labels)
        y_labels = self._label_encoder.transform(y_labels)
        return self._label_smoothing(to_categorical(y_labels),
                                                    self._label_smooth_frac)

    def _label_smoothing(self, array, label_smooth_frac):
        '''smoothes the numbers in a categorical numpy array row-wise'''
        if label_smooth_frac:
            return (array*(1-label_smooth_frac)
                    + (1-array)*label_smooth_frac/(array.shape[1] - 1))
        else:
            return array

    def _split_save_train_dev(self, x_train, y_train, train_ids):
        '''splits the train data into a train and dev set and saves it'''
        if self._dev_frac:
            self._x_train = x_train[int(len(x_train)*self._dev_frac):]
            self._x_dev = x_train[:int(len(x_train)*self._dev_frac)]
            self._y_train = y_train[int(len(y_train)*self._dev_frac):]
            self._y_dev = y_train[:int(len(y_train)*self._dev_frac)]
            self._train_ids = train_ids[int(len(train_ids)*self._dev_frac):]
            self._dev_ids = train_ids[:int(len(train_ids)*self._dev_frac)]
        else:
            self._x_train = x_train
            self._y_train = y_train
            self._train_ids = train_ids
        return True

    def _save_test(self, x_test, test_ids):
        '''saves the test set'''
        self._x_test = x_test
        self._test_ids = test_ids
        return True

    def get_df_train(self):
        '''returns the DataFrame read from the train tsv'''
        return self._df_train

    def get_df_test(self):
        '''returns the DataFrame read from the test tsv'''
        return self._df_test

    def get_train_data(self):
        '''returns the training and development data'''
        return ((self._x_train, self._y_train, self._train_ids),
                (self._x_dev, self._y_dev, self._dev_ids))

    def get_test_data(self):
        '''returns the test data'''
        return (self._x_test, self._test_ids)

    def get_positions_e1(self):
        '''returns an array with the position of every token relative to
        entity one. 0 is reserved for padding tokens, 1 is reserved for
        entity one.
        '''
        pos_e1_train = []
        pos_e1_dev = []
        pos_e1_test = []
        for sent in self._x_train:
            sent = list(sent)
            beg_index = sent.index(4)
            end_index = sent.index(5)
            new = []
            for i in range(len(sent)):
                if sent[i] == 0:
                    new.append(sent[i])
                elif beg_index <= i <= end_index:
                    new.append(1)
                elif i < beg_index:
                    new.append(i - beg_index)
                elif i > end_index:
                    new.append(i - end_index + 1)
            pos_e1_train.append(new)
        for sent in self._x_test:
            sent = list(sent)
            beg_index = sent.index(4)
            end_index = sent.index(5)
            new = []
            for i in range(len(sent)):
                if sent[i] == 0:
                    new.append(sent[i])
                elif beg_index <= i <= end_index:
                    new.append(1)
                elif i < beg_index:
                    new.append(i - beg_index)
                elif i > end_index:
                    new.append(i - end_index + 1)
            pos_e1_test.append(new)
        if not self._dev_frac:
            return (pad_sequences(pos_e1_train, maxlen=self._max_len),
                    None,
                    pad_sequences(pos_e1_test, maxlen=self._max_len))

        for sent in self._x_dev:
            sent = list(sent)
            beg_index = sent.index(4)
            end_index = sent.index(5)
            new = []
            for i in range(len(sent)):
                if sent[i] == 0:
                    new.append(sent[i])
                elif beg_index <= i <= end_index:
                    new.append(1)
                elif i < beg_index:
                    new.append(i - beg_index)
                elif i > end_index:
                    new.append(i - end_index + 1)
            pos_e1_dev.append(new)
        return (pad_sequences(pos_e1_train, maxlen=self._max_len),
                pad_sequences(pos_e1_dev, maxlen=self._max_len),
                pad_sequences(pos_e1_test, maxlen=self._max_len))

    def get_positions_e2(self):
        '''returns an array with the position of every token relative to
        entity two. 0 is reserved for padding tokens, 1 is reserved for
        entity two.
        '''
        pos_e2_train = []
        pos_e2_dev = []
        pos_e2_test = []
        for sent in self._x_train:
            sent = list(sent)
            beg_index = sent.index(6)
            end_index = sent.index(7)
            new = []
            for i in range(len(sent)):
                if sent[i] == 0:
                    new.append(sent[i])
                elif beg_index <= i <= end_index:
                    new.append(1)
                elif i < beg_index:
                    new.append(i - beg_index)
                elif i > end_index:
                    new.append(i - end_index + 1)
            pos_e2_train.append(new)
        for sent in self._x_test:
            sent = list(sent)
            beg_index = sent.index(6)
            end_index = sent.index(7)
            new = []
            for i in range(len(sent)):
                if sent[i] == 0:
                    new.append(sent[i])
                elif beg_index <= i <= end_index:
                    new.append(1)
                elif i < beg_index:
                    new.append(i - beg_index)
                elif i > end_index:
                    new.append(i - end_index + 1)
            pos_e2_test.append(new)
        if not self._dev_frac:
            return (pad_sequences(pos_e2_train, maxlen=self._max_len),
                    None,
                    pad_sequences(pos_e2_test, maxlen=self._max_len))

        for sent in self._x_dev:
            sent = list(sent)
            beg_index = sent.index(6)
            end_index = sent.index(7)
            new = []
            for i in range(len(sent)):
                if sent[i] == 0:
                    new.append(sent[i])
                elif beg_index <= i <= end_index:
                    new.append(1)
                elif i < beg_index:
                    new.append(i - beg_index)
                elif i > end_index:
                    new.append(i - end_index + 1)
            pos_e2_dev.append(new)
        return (pad_sequences(pos_e2_train, maxlen=self._max_len),
                pad_sequences(pos_e2_dev, maxlen=self._max_len),
                pad_sequences(pos_e2_test, maxlen=self._max_len))

    def get_entities_and_context(self):
        '''returns an array with only the entities and the context in
        between them. entity markers were removed.
        '''
        context_train = []
        context_dev = []
        context_test = []
        for sent in self._x_train:
            sent = list(sent)
            beg_index = sent.index(4)
            end_index = sent.index(7)
            new = []
            for i in range(len(sent)):
                if beg_index < i < end_index and sent[i] not in [5, 6]:
                    new.append(sent[i])
            context_train.append(new)
        for sent in self._x_test:
            sent = list(sent)
            beg_index = sent.index(4)
            end_index = sent.index(7)
            new = []
            for i in range(len(sent)):
                if beg_index < i < end_index and sent[i] not in [5, 6]:
                    new.append(sent[i])
            context_test.append(new)
        if not self._dev_frac:
            return (pad_sequences(context_train, maxlen=self._max_len),
                    None,
                    pad_sequences(context_test, maxlen=self._max_len))

        for sent in self._x_dev:
            sent = list(sent)
            beg_index = sent.index(4)
            end_index = sent.index(7)
            new = []
            for i in range(len(sent)):
                if beg_index < i < end_index and sent[i] not in [5, 6]:
                    new.append(sent[i])
            context_dev.append(new)
        return (pad_sequences(context_train, maxlen=self._max_len),
                pad_sequences(context_dev, maxlen=self._max_len),
                pad_sequences(context_test, maxlen=self._max_len))

    def get_dev_frac(self):
        '''returns the fraction of the training data split off for the
        development data'''
        return self._dev_frac

    def get_max_len(self):
        '''returns the maximal sequence length'''
        return self._max_len

    def get_shuffle(self):
        '''returns if the training data have been shuffled'''
        return self._shuffle

    def get_label_smooth_frac(self):
        '''returns the fraction of the probability space distributed
        from the true example to the others'''
        return self._label_smooth_frac

    def get_tokenizer(self):
        '''returns the Tokenizer'''
        return self._tokenizer

    def get_label_encoder(self):
        '''returns the LabelEncoder'''
        return self._label_encoder

########################################################################
# EnsemblePredictor Class
########################################################################

class EnsemblePredictor(object):

    def __init__(self, models, data_dict):
        self._models = models
        self._data_dict = data_dict
        self._x_train = data_dict['x_train']
        self._y_train = data_dict['y_train']
        self._x_dev = data_dict['x_dev']
        self._y_dev = data_dict['y_dev']

    def fit_models(self):
        '''Fits all models stored in self._models with their respective 
        params and the data provided
        '''
        for model, params in self._models:
            training_data = {'x':self._x_train,
                             'y':self._y_train,
                             'validation_data':[self._x_dev, self._y_dev]}
            params = {**training_data, **params}
            if self._x_dev is None or self._y_dev is None:
                params.pop('validation_data', None)
            model.fit(**params)
        return True

    def hard_voting_prediction(self, x):
        '''Perform an ensemble prediction with hard voting and all models
        in self._models

        Args:
            x:  x values for prediction
        '''
        all_predictions = np.zeros((x['whole_sent_input'].shape[0], NUM_CLASSES))
        for model, _ in self._models:
            model_prediction = model.predict(x)
            all_predictions += np.array([np.eye(NUM_CLASSES)[n]
                                         for n
                                         in np.argmax(model_prediction, axis=1)])
        joint_prediction = np.argmax(all_predictions, axis=1)
        return joint_prediction

    def soft_voting_prediction(self, x):
        '''Perform an ensemble prediction with soft voting and all models
        in self._models

        Args:
            x:  x values for prediction
        '''
        all_predictions = np.zeros((x['whole_sent_input'].shape[0], NUM_CLASSES))
        for model, _ in self._models:
            model_prediction = model.predict(x)
            all_predictions += model_prediction
        joint_prediction = np.argmax(all_predictions, axis=1)
        return joint_prediction

    def get_models(self):
        '''returns the models and params in self._models'''
        return self._models

    def get_data_dict(self):
        '''returns the data in self._data_dict'''
        return self._data_dict

    def get_x_train(self):
        '''returns the data in self._x_train'''
        return self._x_train

    def get_y_train(self):
        '''returns the data in self._y_train'''
        return self._y_train

    def get_x_dev(self):
        '''returns the data in self._x_dev'''
        return self._x_dev

    def get_y_dev(self):
        '''returns the data in self._y_dev'''
        return self._y_dev

########################################################################
# KFoldValidator Class
########################################################################

class KFoldValidator(object):

    def __init__(self, model, data_dict, label_encoder):
        self._data_x = data_dict['data_x']
        self._data_y = data_dict['data_y']
        self._model = model
        self._label_encoder = label_encoder

    def kfold_validation(self, k=10):
        '''perform a kfold validation of the model and params in self._model
        
        Args:
            k:  The number of folds        
        '''
        macro_f1_sum = 0
        for x_train, y_train, x_dev, y_dev in self._kfold_datasets(k):
            model, params = copy.deepcopy(self._model)
            training_data = {'x':x_train,
                             'y':y_train,
                             'validation_data':[x_dev, y_dev]}
            params = {**training_data, **params}
            model.fit(**params)
            prediction = self._reverse_label_encoding(np.argmax(model.predict(x_dev),
                                                                axis=1))
            true_labels = self._reverse_label_encoding(np.argmax(y_dev['primary_out'],
                                                                 axis=1))
            macro_f1_sum += evaluate(true_labels, prediction)
        mean_macro_f1 = macro_f1_sum/k
        return mean_macro_f1

    def _kfold_datasets(self, k):
        '''split the data in k chunks and return one chunk as dev and the rest
        as train k times

        Args:
            k:  The number of folds
        '''
        chunks_x = {input_key:np.array_split(data, k)
                    for input_key, data 
                    in self._data_x.items()}
        chunks_y = {output_key:np.array_split(data, k)
                    for output_key, data
                    in self._data_y.items()}
        for fold in range(k):
            x_train = None
            y_train = None
            x_dev = {input_key:chunks[fold]
                     for input_key, chunks
                     in chunks_x.items()}
            y_dev = {output_key:chunks[fold]
                     for output_key, chunks
                     in chunks_y.items()}
            for i in range(k):
                if i != fold:
                    if x_train is None and y_train is None:
                        x_train = {input_key:chunks[i]
                                   for input_key, chunks
                                   in chunks_x.items()}
                        y_train = {output_key:chunks[i]
                                   for output_key, chunks
                                   in chunks_y.items()}
                    else:
                        x_train = {input_key:np.vstack((x_train[input_key], chunks[i]))
                                   for input_key, chunks
                                   in chunks_x.items()}
                        y_train = {output_key:np.vstack((y_train[output_key], chunks[i]))
                                   for output_key, chunks
                                   in chunks_y.items()}
            yield x_train, y_train, x_dev, y_dev

    def _reverse_label_encoding(self, array):
        '''reverses label encoding using a specific instance of LabelEncoder'''
        string_labels = list(self._label_encoder.inverse_transform(array))
        return string_labels

    def get_data_x(self):
        '''returns the features for the data'''
        return self._data_x

    def get_data_y(self):
        '''returns the labels for the data'''
        return self._data_y

    def get_model(self):
        '''returns model'''
        return self._model

    def get_label_encoder(self):
        '''returns the LabelEncoder'''
        return self._label_encoder

########################################################################
# Miscellaneous Functions
########################################################################

def reverse_label_encoding(array, label_encoder):
    '''reverses label encoding using a specific instance of LabelEncoder'''
    string_labels = list(label_encoder.inverse_transform(array))
    return string_labels

def write_prediction_file(outpath, ids, labels):
    '''writes ids and labels to a prediction file'''
    pred_array = np.array([[pred_id, pred_label]
                           for pred_id, pred_label
                           in zip(ids, labels)])
    df = pd.DataFrame(pred_array, columns=['id', 'label'])
    df.sort_values(by=['id'], inplace=True)
    df.to_csv(outpath, sep='\t', index=False, header=False)
    return True

########################################################################
# Relation Classification Models and Params
########################################################################

def relation_model(emb_num, emb_words, activation, loss, optimizer):
    '''A model for relation classification'''
    # three main inputs
    whole_sent_input = Input(shape=(200,), name='whole_sent_input')
    e1_input = Input(shape=(200,), name='e1_input')
    e2_input = Input(shape=(200,), name='e2_input')

    # the two inputs with distances to the entities share an embedding layer
    distance_embeddings = Embedding(input_dim=400,
                                    output_dim=emb_num,
                                    input_length=200)
    emb_e1 = distance_embeddings(e1_input)
    emb_e2 = distance_embeddings(e2_input)

    # embedding layer for the whole sentence
    word_embeddings = Embedding(input_dim=5000,
                                output_dim=emb_words,
                                input_length=200)
    emb_whole = word_embeddings(whole_sent_input)

    # combining the embeddings of the three main inputs
    combined = Concatenate()([emb_whole, emb_e1, emb_e2])

    # convolution and pooling over the data with different filter sizes
    conv_three = Conv1D(filters=100,
                        kernel_size=3,
                        activation=activation,
                        padding='same')(combined)
    conv_five = Conv1D(filters=100,
                       kernel_size=5,
                       activation=activation,
                       padding='same')(combined)
    conv_seven = Conv1D(filters=100,
                        kernel_size=7,
                        activation=activation,
                        padding='same')(combined)
    combined_conv = Concatenate()([conv_three, conv_five, conv_seven])
    pool = GlobalMaxPooling1D()(combined_conv)

    # dropout layer
    droput = Dropout(rate=0.2)(pool)

    # softmax output layer
    out = Dense(units=NUM_CLASSES, activation='softmax', name='primary_out')(pool)

    # defining and compiling the model
    model = Model(inputs=(whole_sent_input, e1_input, e2_input), outputs=out)
    model.compile(loss=loss, optimizer=optimizer, metrics=['acc'])

    return model

def relation_params(batch_size, class_weight=None):
    '''fitting params for a model'''
    callbacks = [EarlyStopping(monitor='val_acc',
                               patience=5,
                               verbose=1,
                               restore_best_weights=True,
                               mode='max')]
    params = {'epochs':300,
              'batch_size':batch_size,
              'class_weight':class_weight,
              'callbacks':callbacks}
    return params

########################################################################
# Main Function (Training and K-Fold-Validation)
########################################################################

def main():
    # preparing data
    my_data = Data()
    tokenizer = Tokenizer(filters='\t\n', num_words=5000, oov_token=1)
    train_path = '../../../data/relation_classification/relation_classification_small_bpe.train'
    test_path = '../../../data/relation_classification/relation_classification_small_bpe.test'
    my_data.prepare_data(train_path, test_path,
                         dev_frac=0.1,
                         max_len=200,
                         tokenizer=tokenizer,
                         label_smooth_frac=0.1)
    (x_train, y_train, train_ids), (x_dev, y_dev, dev_ids) = my_data.get_train_data()
    (x_test, test_ids) = my_data.get_test_data()
    (pos_e1_train, pos_e1_dev, pos_e1_test) = my_data.get_positions_e1()
    (pos_e2_train, pos_e2_dev, pos_e2_test) = my_data.get_positions_e2()
    train_data = {'x_train':{'whole_sent_input':x_train,
                             'e1_input':pos_e1_train,
                             'e2_input':pos_e2_train},
                  'y_train':{'primary_out':y_train},
                  'x_dev':{'whole_sent_input':x_dev,
                           'e1_input':pos_e1_dev,
                           'e2_input':pos_e2_dev},
                  'y_dev':{'primary_out':y_dev}}

    # models and fitting params
    class_weight = {cl:1-list(np.argmax(y_train, axis=1)).count(cl)/len(y_train)
                    for cl
                    in set(np.argmax(y_train, axis=1))}
    models_function_params = [
        ({'emb_num':35, 'emb_words':300, 'activation':'elu', 'loss':'poisson', 'optimizer':'rmsprop'},
            {'batch_size':880, 'class_weight':None}),
        ({'emb_num':25, 'emb_words':450, 'activation':'selu', 'loss':'categorical_crossentropy', 'optimizer':'nadam'},
            {'batch_size':60, 'class_weight':None}),
        ({'emb_num':30, 'emb_words':250, 'activation':'elu', 'loss':'poisson', 'optimizer':'adam'},
            {'batch_size':40, 'class_weight':None}),
        ({'emb_num':5, 'emb_words':400, 'activation':'tanh', 'loss':'kullback_leibler_divergence', 'optimizer':'rmsprop'},
            {'batch_size':180, 'class_weight':None}),
        ({'emb_num':30, 'emb_words':400, 'activation':'selu', 'loss':'kullback_leibler_divergence', 'optimizer':'nadam'},
            {'batch_size':300, 'class_weight':class_weight}),
        ({'emb_num':25, 'emb_words':250, 'activation':'selu', 'loss':'kullback_leibler_divergence', 'optimizer':'adagrad'},
            {'batch_size':580, 'class_weight':class_weight}),
        ({'emb_num':20, 'emb_words':450, 'activation':'softplus', 'loss':'poisson', 'optimizer':'adagrad'},
            {'batch_size':500, 'class_weight':class_weight}),
        ({'emb_num':10, 'emb_words':300, 'activation':'relu', 'loss':'kullback_leibler_divergence', 'optimizer':'adam'},
            {'batch_size':60, 'class_weight':class_weight}),
        ({'emb_num':20, 'emb_words':300, 'activation':'selu', 'loss':'poisson', 'optimizer':'adam'},
            {'batch_size':700, 'class_weight':None}),
        ({'emb_num':10, 'emb_words':250, 'activation':'relu', 'loss':'categorical_crossentropy', 'optimizer':'nadam'},
            {'batch_size':260, 'class_weight':None}),
        ({'emb_num':25, 'emb_words':350, 'activation':'linear', 'loss':'kullback_leibler_divergence', 'optimizer':'adamax'},
            {'batch_size':940, 'class_weight':None}),
        ({'emb_num':25, 'emb_words':400, 'activation':'softplus', 'loss':'mean_absolute_error', 'optimizer':'adamax'},
            {'batch_size':240, 'class_weight':class_weight}),
        ({'emb_num':20, 'emb_words':100, 'activation':'elu', 'loss':'categorical_crossentropy', 'optimizer':'rmsprop'},
            {'batch_size':460, 'class_weight':class_weight}),
        ({'emb_num':20, 'emb_words':450, 'activation':'exponential', 'loss':'kullback_leibler_divergence', 'optimizer':'rmsprop'},
            {'batch_size':800, 'class_weight':None}),
        ({'emb_num':45, 'emb_words':250, 'activation':'selu', 'loss':'categorical_crossentropy', 'optimizer':'adagrad'},
            {'batch_size':600, 'class_weight':class_weight})
        ]
    models = [(relation_model(**model_p), relation_params(**params_p))
              for model_p, params_p
              in models_function_params]

    # ensemble predictions
    predictor = EnsemblePredictor(models, train_data)
    predictor.fit_models()
    y_dev_pred = predictor.soft_voting_prediction({'whole_sent_input':x_dev,
                                                   'e1_input':pos_e1_dev,
                                                   'e2_input':pos_e2_dev})
    y_test_pred = predictor.soft_voting_prediction({'whole_sent_input':x_test,
                                                    'e1_input':pos_e1_test,
                                                    'e2_input':pos_e2_test})

    # reversing to_categorical()
    y_dev_num = np.argmax(y_dev, axis=1)

    # reversing label encoding
    label_encoder = my_data.get_label_encoder()
    y_dev_pred_strings = reverse_label_encoding(y_dev_pred, label_encoder)
    y_dev_strings = reverse_label_encoding(y_dev_num, label_encoder)
    y_test_pred_strings = reverse_label_encoding(y_test_pred, label_encoder)

    # evaluation of the performance on the dev set using the scoring script provided
    print(evaluate(y_dev_strings, y_dev_pred_strings))

    # writing test prediction file
    outfile = './outfile.txt'
    write_prediction_file(outfile, test_ids, y_test_pred_strings)


    # # preparing data for kfold validation
    # my_data = Data()
    # tokenizer = Tokenizer(filters='\t\n', num_words=5000, oov_token=1)
    # train_path = '../../../data/relation_classification/relation_classification_small_bpe.train'
    # test_path = '../../../data/relation_classification/relation_classification_small_bpe.test'
    # my_data.prepare_data(train_path, test_path,
    #                      dev_frac=None,
    #                      max_len=200,
    #                      tokenizer=tokenizer,
    #                      label_smooth_frac=0.1)
    # (x_train, y_train, _), (_, _, _) = my_data.get_train_data()
    # (pos_e1_train, _, _) = my_data.get_positions_e1()
    # (pos_e2_train, _, _) = my_data.get_positions_e2()
    # kfold_data = {'data_x':{'whole_sent_input':x_train,
    #                         'e1_input':pos_e1_train,
    #                         'e2_input':pos_e2_train},
    #               'data_y':{'primary_out':y_train}}
    # class_weight = {cl:1-list(np.argmax(y_train, axis=1)).count(cl)/len(y_train)
    #                 for cl
    #                 in set(np.argmax(y_train, axis=1))}

    # # klfold validation of a model
    # test_parms = ({'emb_num':35, 'emb_words':300, 'activation':'elu', 'loss':'poisson', 'optimizer':'rmsprop'},
    #               {'batch_size':880, 'class_weight':None})
    # validator = KFoldValidator((relation_model(**test_parms[0]), 
    #                             relation_params(**test_parms[1])),
    #                            kfold_data,
    #                            my_data.get_label_encoder())
    # macro_f1 = validator.kfold_validation(10)
    # print(f'Mean Macro-F1:\t{macro_f1}')

if __name__ == '__main__':
    main()
