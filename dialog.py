import codecs
import uniout

TRAINING_DATA_FILE = "./data/train/train.txt"

with codecs.open(TRAINING_DATA_FILE,encoding='utf-8') as f:
    data = f.read()

#print data
total_len = len(data)
#print total_len
words = list(set(data))
print len(words)
#print words
