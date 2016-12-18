# -*- coding: utf-8 -*-
import codecs
import uniout
from collections import Counter
from gensim.models import Word2Vec
import numpy as np
from config import *

def tokenize(sentence):
    return list(sentence)

def get_tokens_voc(tokenize_sentence):
    token_counter = Counter()

    for line in tokenize_sentence:
        for token in line:
            token_counter.update([token])

    token_voc = [token for token, _ in token_counter.most_common()[:VOCAB_MAX_SIZE]]
    #token_voc.append(EOS_SYMBOL)
    return token_voc;

def get_token_vector(token,model):
    if token in model.vocab:
        return np.array(model[token])
    return np.zeros(TOKEN_REPRESENTATION_SIZE)

def read_corpus(process_corpus):
    sentences = []
    for line in codecs.open(process_corpus,'r','utf-8'):
        sentences.append(line)

    return sentences

def process_corpus(sentences):
    tokenize_sentences = []
    for line in sentences:
        tokenize_sentences.append(tokenize(line))
    
    tokens_voc = get_tokens_voc(tokenize_sentences)
    index_to_token = dict(enumerate(tokens_voc))
    return tokenize_sentences,index_to_token




def get_language_model(full_file_path):
    print 'getting language model...'
    model = Word2Vec.load_word2vec_format(full_file_path,binary=True,unicode_errors='ignore')
    print 'end'
    return model
