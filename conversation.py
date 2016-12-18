# -*- coding: utf-8 -*-
import uniout
import pickle
import os
import jieba
from itertools import tee

from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense,Activation,Dropout
from keras.models import Sequential
from seq2seq.models import AttentionSeq2seq,SimpleSeq2seq
from seq2seq.layers.decoders import AttentionDecoder
from keras.layers import RepeatVector,TimeDistributed,Dense
import numpy as np

from config import *
from model import *
from preprocess import *

def _batch(tokenized_sentences,batch_size,start_pos):
    size = len(tokenized_sentences)
    if start_pos > size:
        start_pos = 0
    end_pos = start_pos + batch_size
    if end_pos > size:
        end_pos = size

    batch = []
    for line in tokenized_sentences[start_pos:end_pos]:
        batch.append(line)
    return batch
    #yield []

def get_training_batch(w2v_model,x_tokenized_sentences,y_tokenized_sentences,y_token_to_index,start_pos):

    token_voc_size = len(y_token_to_index)

    print "y_token_voc_size: %d" % token_voc_size
    
    X = np.zeros((SAMPLES_BATCH_SIZE,INPUT_SEQUENCE_LENGTH,TOKEN_REPRESENTATION_SIZE))
    Y = np.zeros((SAMPLES_BATCH_SIZE,ANSWER_MAX_TOKEN_LENGTH,token_voc_size))
    
    #for sent_batch in _batch(tokenized_sentences,SAMPLES_BATCH_SIZE)
    x_sent_batch = _batch(x_tokenized_sentences,SAMPLES_BATCH_SIZE,start_pos)
    y_sent_batch = _batch(y_tokenized_sentences,SAMPLES_BATCH_SIZE,start_pos)
    sen_idx = 0
    
    for s_idx in xrange(0,len(x_sent_batch),1):
            
        for t_idx,token in enumerate(x_sent_batch[s_idx][:INPUT_SEQUENCE_LENGTH]):
            X[s_idx,t_idx] = get_token_vector(token,w2v_model)

        for t_idx,token in enumerate(y_sent_batch[s_idx][:ANSWER_MAX_TOKEN_LENGTH]):
            Y[s_idx,t_idx,y_token_to_index[token]] = 1
            
        #yield X,Y
    return X,Y

def train_model(nn_model,w2v_model,x_tokenized_sentences,x_index_to_token,y_tokenized_sentences,y_index_to_token):
    print 'training nn model...'
    #print ('Original input:',tokenized_sentences[0])
    #print ('Expected output:',tokenized_sentences[1])
    token_to_index = dict(zip(y_index_to_token.values(),y_index_to_token.keys()))
  
    start_pos = 0;
    
    for full_data_pass_num in xrange(1,FULL_LEARN_ITER_NUM+1):
        
         x_train,y_train = get_training_batch(w2v_model,x_tokenized_sentences,y_tokenized_sentences,token_to_index,start_pos)
         start_pos = start_pos + SAMPLES_BATCH_SIZE
        #for x_train,y_train in get_training_batch(w2v_model,tokenized_sentences,token_to_index):
        #    nn_model.fit(x_train,y_train,batch_size=TRAIN_BATCH_SIZE,nb_epoch=3,verbose=1)
         nn_model.fit(x_train,y_train,batch_size=TRAIN_BATCH_SIZE,nb_epoch=10,verbose=1)
         '''predictions = nn_model.predict(x_train);
         for i_idx, predict in enumerate(predictions):
             predict_sequence = []
             for predict_vector in predict:
                 next_index = np.argmax(predict_vector)
                 next_token = index_to_token[next_index]
                 predict_sequence.append(next_token)
             print ('Predict output:',predict_sequence)'''
    nn_model.save_weights(NN_MODEL_PATH,overwrite=True)
    print 'end'

'''
    learn and verify nn model
'''

def predict_sentence(sentence,nn_model,w2v_model,index_to_token):
    input_sentence = tokenize(sentence)[:INPUT_SEQUENCE_LENGTH]
    #print 'input_sentence:%s' % str(input_sentence)
    X = np.zeros((TRAIN_BATCH_SIZE,INPUT_SEQUENCE_LENGTH,TOKEN_REPRESENTATION_SIZE))
    #print input_sentence
    for t, token in enumerate(input_sentence):
        X[0,t] = get_token_vector(token,w2v_model)
        #print X[0,t]
       
    predictions = nn_model.predict(X,verbose=1)[0]
    predicted_sequence = []

    for prediction_vector in predictions:
        next_index = np.argmax(prediction_vector)
        #print 'next_index is %d' % next_index
        next_token = index_to_token[next_index]
        predicted_sequence.append(next_token)
    
    predicted_sequence = ''.join(predicted_sequence)

    return predicted_sequence

def learn_and_verify():
  
   sentences = read_corpus(TRAIN_SET_PATH)
   x_sentences = sentences[::2]
   y_sentences = sentences[1::2]

   x_processed_sentences,x_index_to_token = process_corpus(x_sentences)
   y_processed_sentences,y_index_to_token = process_corpus(y_sentences)

   w2v_model = get_language_model(CORPUS_FILE_PATH)
   nn_model = get_nn_model(token_dict_size=len(y_index_to_token))

   train_model(nn_model,w2v_model,x_processed_sentences,x_index_to_token,y_processed_sentences,y_index_to_token)
   
   print 'verfiy nn model...'
   iterable_test_sentences = get_iterable_sentences(TEST_SET_PATH)
   for test_sent in iterable_test_sentences:
       predicted_answer = predict_sentence(test_sent,nn_model,w2v_model,y_index_to_token)
       print test_sent,predicted_answer
   print 'end'


learn_and_verify()
