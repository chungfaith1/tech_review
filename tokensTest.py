import tensorflow as tf
import sys
import time
import numpy
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.text import hashing_trick
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

class KerasModel:
    def __init__(self):
        self.params = params

    def build_model(self):
        print("implement later")

    def fit_model(self):
        print()

class KerasData:
    def __init__(self):
        # Extracted data from txt's
        self.motiv_quotes = []
        self.demotiv_quotes = []
        # Hashing_trick vars
        self.vocab = None
        self.vocab_size = None
        self.encoded_quotes = []
        self.quotes = None
        # Collisions analysis
        self.collisions = {}
        self.collisions_count = 0

    # Store txt data as str lists
    def init_data(self):
        f = open("./motivational.txt", "r", errors = 'ignore')
        for line in f.readlines():
            self.motiv_quotes.append(line)

        f = open("./demotivational.txt", "r", errors = 'ignore')
        for line in f.readlines():
            self.demotiv_quotes.append(line)

    # Tokenize text with hashing_trick
    def hashing_method(self):
        # get vocab size
        motiv = self.flatten(self.motiv_quotes)
        demotiv = self.flatten(self.demotiv_quotes)
        self.vocab = set(text_to_word_sequence(motiv + " " + demotiv))
        self.vocab_size = len(self.vocab)
        
        # perform hash encoding
        self.quotes = self.motiv_quotes + self.demotiv_quotes
        before = time.time()
        for quote in self.quotes:
            self.encoded_quotes.append(hashing_trick(quote, round(self.vocab_size*1.5), hash_function='md5'))
        after = time.time()
        diff = (after-before) * 1000
        print("hashing trick time: " + str(diff) + " ms")

        # PADDED HASH DATA FOR TRAINING
        self.padded_encoded_quotes = pad_sequences(self.encoded_quotes, maxlen=280)
        #print(self.encoded_quotes)
        #print("----------------------------------------")
        print(self.padded_encoded_quotes)

    # Tokenize text with Tokenizer class
    def token_method(self):
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token= '<OOV>')
        data = self.motiv_quotes + self.demotiv_quotes

        self.max_len = 0
        for d in data:
            if len(d) > self.max_len:
                self.max_len = len(d)
        before = time.time()

        # Fit tokenizer on documents
        tokenizer.fit_on_texts(data)
        print("word index: " + str(tokenizer.word_index))
        encoded_data = tokenizer.texts_to_sequences(data)
        after = time.time()
        diff = (after-before) * 1000
        print("token trick time: " + str(diff) + " ms")
        # PADDED TOKENIZED DATA FOR TRAINING
        self.padded_token_quotes = pad_sequences(encoded_data, maxlen=self.max_len)
        print(self.padded_token_quotes)

    # Build and train model with padded data
    def train(self):
        # build model
        self.embedding_dim = 16
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
        )
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model2 = tf.keras.Sequential([
        tf.keras.layers.Embedding(self.max_hashKey+10, 16, input_length=self.max_len),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(5, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ]
        )
        model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # get Tokenized train and test data
        # 70 motivational, 70 demotivational quotes

        self.token_train = (self.padded_token_quotes[0:70, 0:280]).tolist() + (self.padded_token_quotes[130:200, 0:280]).tolist()
        self.train_label = [1]*70 + [0]*70 # 1 = motivational, 0 = demotivational
        self.token_test  = (self.padded_token_quotes[70:130, 0:280]).tolist()
        self.test_label  = [1]*30 + [0]*30

        # get Hashed train data (has same labels as tokenized data)
        self.hash_train = (self.padded_encoded_quotes[0:70, 0:280]).tolist() + (self.padded_encoded_quotes[130:200, 0:280]).tolist()
        self.hash_test  = (self.padded_encoded_quotes[70:130, 0:280]).tolist()
        self.token_test  = (self.padded_token_quotes[70:130, 0:280]).tolist()
        self.test_label  = [1]*30 + [0]*30

        # train model with token data
        num_epochs = 20
        print("TRAINING WITH TOKENIZER DATA")
        model.fit(
        self.token_train,
        self.train_label,
        epochs=num_epochs,
        validation_data = (self.token_test, self.test_label)
        )

        # train model with hash data
        print("TRAINING WITH HASHED DATA")
        model2.fit(
        self.hash_train,
        self.train_label,
        epochs=num_epochs,
        validation_data = (self.hash_test, self.test_label)
        )        

        
    # helper: concats list of strings to 1 string
    def flatten(self, arr):
        str = ""
        for sublist in arr:
            for item in sublist:
                str = str + item
        return str

    # helper: count number of collisions
    def get_collisions(self):
        for i in range(len(self.quotes)):
            words = text_to_word_sequence(self.quotes[i])
            for j in range(len(words)):
                word = words[j]
                num = self.encoded_quotes[i][j]

                if num in self.collisions:
                    if (not word in self.collisions[num]): # new word, same hash = collision!
                        self.collisions_count += 1
                        l = self.collisions[num] + [word]
                        self.collisions[num] = l
                else: # new hash id
                    self.collisions[num] = [word]
        
        collision_words = 0
        self.max_hashKey = 0
        for num in self.collisions:
            print(str(num) + ": " + str(self.collisions[num]))
            if len(self.collisions[num]) > 1:
                collision_words += 1
            if num > self.max_hashKey:
                self.max_hashKey = num

        print("vocab size: " + str(self.vocab_size))
        print("dictionary size: " + str(len(self.collisions)))
        print("number of hash ids with collisions: " + str(collision_words))

def main():
    numpy.set_printoptions(threshold=sys.maxsize)
    # Get data
    data = KerasData()
    data.init_data() 
    print("****************** Hashing Method ******************")
    data.hashing_method()
    data.get_collisions()
    print("****************** Tokenizer Method ******************")
    data.token_method()
    print("****************** Model Results ******************")
    data.train()
    

if __name__ == '__main__':
    main()