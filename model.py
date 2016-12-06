
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import TimeDistributedDense,Activation,Dropout
from keras.models import Sequential
from seq2seq.models import AttentionSeq2seq,SimpleSeq2seq
from seq2seq.layers.decoders import AttentionDecoder
from keras.layers import RepeatVector,TimeDistributed,Dense
from config import *

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
