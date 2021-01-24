import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer() #initializing lemmatizer to get the root word

intents = json.loads(open("intents.json").read()) #reading intents.json

words = []  #contains words
classes = []    #contains tags
documents = []  #contains dictionary of words and tags
ignore_letters = ['?', '!', '.', ','] #ignore symbols

for intent in intents['intents']:   #search for intents as in json
    for pattern in intent['patterns']:  #search for pattern as in json
        word_list = nltk.word_tokenize(pattern) #tokenize divides string into substring / sentence to word
        words.extend(word_list) #to extend the array 
        documents.append((word_list, intent['tag'])) #append the word and tag to document

        if intent['tag'] not in classes:    #add tag to classes if not present
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]    #lemmatize the word except the ignore_letters
words = sorted(set(words))  #remove repitions and convert set to list
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb')) #create a pickle file
pickle.dump(classes, open('classes.pkl', 'wb')) #create a pickle file

training = [] #storing training data
output_empty = [0]*len(classes) #zero array of length = number of tags/classes

for document in documents:
    bag = [] #using bag of words method
    word_patterns = document[0] #first element is tokenized word
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns] #lemmatize the words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) #if word occurs 1 else 0
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] =  1 #for current tag output 1 for other output 0
    training.append([bag, output_row]) #append dictionary of bag of words and class

random.shuffle(training) #randomize the training data
training = np.array(training) #type cast to np.array

train_x = list(training[:, 0]) #bag of words
train_y = list(training[:, 1]) #class

model = Sequential() #appropiate model having 1 input layer and 1 output layer
model.add(Dense(128, input_shape = (len(train_x[0]), ), activation = 'relu')) #adding a dense layer of 128 neurons using Rectified Linear Unit
model.add(Dropout(0.5)) #adding dropout layer to prevent overfitting
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax')) #using softmax as probability is required

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True) #stochastic gradient descent
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy']) #compiling model using categorical as morethan 2 classes are there

hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1) #fitting the model 200 times

model.save('speakModel.h5', hist)
print("[+]Done")