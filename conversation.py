# -*- coding: utf-8 -*-
import uniout
import pickle
import codecs
import os
import jieba
from gensim.models import Word2Vec
from itertools import tee
from collections import Counter
from itertools import tee
import numpy as np

from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense,Activation,Dropout
from keras.models import Sequential
from seq2seq.models import AttentionSeq2seq,SimpleSeq2seq
from seq2seq.layers.decoders import AttentionDecoder
from keras.layers import RepeatVector,TimeDistributed,Dense

CORPUS_FILE_PATH = "./data/vectors.bin.skipgram.mergenew.2.3"
DATA_SET_PATH = "./data/data.txt"
TRAIN_SET_PATH = "./data/train/train.txt"
TEST_SET_PATH = "./data/test/test.txt"
TOKEN_REPRESENTATION_SIZE = 300
INPUT_SEQUENCE_LENGTH = 20
ANSWER_MAX_TOKEN_LENGTH = 16
VOCAB_MAX_SIZE = 20000
HIDDEN_LAYER_DIMENSION = 512
FULL_LEARN_ITER_NUM = 500
SAMPLES_BATCH_SIZE = 100
EOS_SYMBOL = '$$$'
EMPTY_SYMBOL= '###'
TRAIN_BATCH_SIZE = 20 
NN_MODEL_PATH = './model/model'

'''
    preprocess for sentences

'''

def get_language_model(full_file_path):
    #model = Word2Vec.load(full_file_path)
    print 'getting language model...'
    model = Word2Vec.load_word2vec_format(full_file_path,binary=True,unicode_errors='ignore')
    print 'end'
    return model

def get_token_vector(token,model):
    if token in model.vocab:
        return np.array(model[token])
    return np.zeros(TOKEN_REPRESENTATION_SIZE)


def get_vectorized_token_sequence(sequence,model,max_sequence_length,reverse=False):
    vectorized_token_sequence = np.zeros((max_sequence_length,TOKEN_REPRESENTATION_SIZE),dtype=np.float)
    
    for idx,token in enumerate(sequence):
        vectorized_token_sequence[idx] = get_token_vector(token,model)

    if reverse:
        vectorized_token_sequence = vectorized_token_sequence[::-1]

    return vectorized_token_sequence

def get_iterable_sentences(processed_corpus_path):
    for line in codecs.open(processed_corpus_path,'r','utf-8'):
        yield line.strip()

def get_tokens_voc(tokenized_sentences):
    token_counter = Counter()

    for line in tokenized_sentences:
        for token in line:
            token_counter.update([token])
    
    token_voc = [token for token, _ in token_counter.most_common()[:VOCAB_MAX_SIZE]]
    print token_voc
    token_voc.append(EMPTY_SYMBOL)
    return set(token_voc)

def tokenize(sentence):
    return list(jieba.cut(sentence))

def get_tokenized_sentences(iterable_sentences):
    for line in iterable_sentences:
        tokenized_sentence = tokenize(line)
        tokenized_sentence.append(EOS_SYMBOL)
        yield tokenized_sentence

def get_transformed_tokenized_sentences(tokenized_sentences, tokens_voc):
    for line in tokenized_sentences:
        transformed_line = []

        for token in line:
            if token not in tokens_voc:
                token = EMPTY_SYMBOL

            transformed_line.append(token)
        yield transformed_line

def process_corpus(corpus_path):
    iterable_sentences = get_iterable_sentences(corpus_path)

    #for line in iterable_sentences:
    #    print line

    tokenized_sentences = get_tokenized_sentences(iterable_sentences)
    
    #for ts in tokenized_sentences:
    #    print ts

    tokenized_sentences_for_voc, tokenized_sentences_for_transform = tee(tokenized_sentences)

    tokens_voc = get_tokens_voc(tokenized_sentences_for_voc)
    
    #print tokens_voc
    #for tv in tokens_voc:
    #    print tv
    
    transformed_tokenized_sentences = get_transformed_tokenized_sentences(tokenized_sentences_for_transform,tokens_voc)

    index_to_token = dict(enumerate(tokens_voc))

    return transformed_tokenized_sentences,index_to_token

def get_processed_sentence_and_index_to_token(corpus_path,processed_corpus_path='',token_index_path=''):
    #if os.path.isfile(processed_corpus_path) and os.path.isfile(token_index_path):
    #    processed_sentences_lines = getIterableSentences(processed_corpus_path)
    print 'process training sentences...'
    processed_sentences, index_to_token = process_corpus(corpus_path)
    print 'end'
    return processed_sentences,index_to_token

'''
    get nn model

'''
def get_nn_model(token_dict_size):

    '''model = Sequential()
    seq2seq = AttentionSeq2seq(
    #seq2seq = SimpleSeq2seq(
        input_dim = TOKEN_REPRESENTATION_SIZE,
        input_length = INPUT_SEQUENCE_LENGTH,
        hidden_dim = HIDDEN_LAYER_DIMENSION,
        output_dim = token_dict_size,
        output_length = ANSWER_MAX_TOKEN_LENGTH,
        depth = 4
    )
    model.add(seq2seq)
    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    '''
    dropout = 0.1 
    model = Sequential()
    encoder_top_layer = LSTM(HIDDEN_LAYER_DIMENSION,input_dim=TOKEN_REPRESENTATION_SIZE,input_length=INPUT_SEQUENCE_LENGTH,return_sequences=True)

    decoder_top_layer = AttentionDecoder(hidden_dim=HIDDEN_LAYER_DIMENSION,output_dim=HIDDEN_LAYER_DIMENSION,output_length=ANSWER_MAX_TOKEN_LENGTH,state_input=False,return_sequences=True)
    #model.add(Embedding(input_dim=TOKEN_REPRESENTATION_SIZE,output_dim=HIDDEN_LAYER_DIMENSION,input_length=INPUT_SEQUENCE_LENGTH))
    model.add(encoder_top_layer)
    model.add(Dropout(dropout))
    model.add(LSTM(HIDDEN_LAYER_DIMENSION,return_sequences=False))
    model.add(RepeatVector(ANSWER_MAX_TOKEN_LENGTH))
    model.add(decoder_top_layer)
    model.add(Dropout(dropout))
    model.add(LSTM(HIDDEN_LAYER_DIMENSION,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(token_dict_size)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    #if os.path.isfile(NN_MODEL_PATH):
    #    model.load_weights(NN_MODEL_PATH)


    return model


'''
    train nn model

'''

def _batch(tokenized_sentences,batch_size=1000):
    batch = []
    for line in tokenized_sentences:
        batch.append(line)
        if len(batch) == 2*batch_size:
            #yield batch
            #batch = []
            break

    return batch
    #yield []

def get_training_batch(w2v_model,tokenized_sentences,token_to_index):

    token_voc_size = len(token_to_index)

    X = np.zeros((SAMPLES_BATCH_SIZE,INPUT_SEQUENCE_LENGTH,TOKEN_REPRESENTATION_SIZE))
    Y = np.zeros((SAMPLES_BATCH_SIZE,ANSWER_MAX_TOKEN_LENGTH,token_voc_size))
    
    #for sent_batch in _batch(tokenized_sentences,SAMPLES_BATCH_SIZE)
    sent_batch = _batch(tokenized_sentences,SAMPLES_BATCH_SIZE)
    sen_idx = 0
    
    for s_idx in xrange(0,len(sent_batch),2):
            
        #print ('input:',sent_batch[s_idx][:INPUT_SEQUENCE_LENGTH])
        #print ('output:',sent_batch[s_idx+1][:ANSWER_MAX_TOKEN_LENGTH])

        #print sent_batch[s_idx],sent_batch[s_idx+1]
        for t_idx,token in enumerate(sent_batch[s_idx][:INPUT_SEQUENCE_LENGTH]):
            X[sen_idx,t_idx] = get_token_vector(token,w2v_model)

        for t_idx,token in enumerate(sent_batch[s_idx+1][:ANSWER_MAX_TOKEN_LENGTH]):
            Y[sen_idx,t_idx,token_to_index[token]] = 1
            
        sen_idx += 1
        #yield X,Y
    return X,Y

def train_model(nn_model,w2v_model,tokenized_sentences,index_to_token):
    print 'training nn model...'
    #print ('Original input:',tokenized_sentences[0])
    #print ('Expected output:',tokenized_sentences[1])
    token_to_index = dict(zip(index_to_token.values(),index_to_token.keys()))
   
    x_train,y_train = get_training_batch(w2v_model,tokenized_sentences,token_to_index)
    
    for full_data_pass_num in xrange(1,FULL_LEARN_ITER_NUM+1):
        
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


def readData(filename):
    pkl_file = open(filename,"rb")
    res = pickle.load(pkl_file)
    pkl_file.close()

    data = []
    for key in res:
        data.append(key)
        data.append(res[key])
    
    file = codecs.open("data.txt","w","utf-8")
    for v in data:
        file.write(v+"\n")
    file.close

    return data

def predict_sentence(sentence,nn_model,w2v_model,index_to_token):
    input_sentence = tokenize(sentence+EOS_SYMBOL)[:INPUT_SEQUENCE_LENGTH]
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
  
   processed_sentences,index_to_token = get_processed_sentence_and_index_to_token(TRAIN_SET_PATH)
   w2v_model = get_language_model(CORPUS_FILE_PATH)
   nn_model = get_nn_model(token_dict_size=len(index_to_token))
   train_model(nn_model,w2v_model,processed_sentences,index_to_token)
   
   print 'verfiy nn model...'
   iterable_test_sentences = get_iterable_sentences(TEST_SET_PATH)
   for test_sent in iterable_test_sentences:
       predicted_answer = predict_sentence(test_sent,nn_model,w2v_model,index_to_token)
       print test_sent,predicted_answer
   print 'end'


learn_and_verify()

#data = readData('data.bk')
#print data[-5:]
#language_model = get_language_model(CORPUS_FILE_PATH)
#print len(language_model[u"女人"])
#print language_model.vocab
