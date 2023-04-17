print("Loading...\nThis may take a few seconds, I'll let you know when I'm ready...\n")

import json
import string
import random
import nltk

#Correct Punkt certificate errors:
#import ssl
#try:
#    _create_unverified_https_context = ssl._create_unverified_context
#except AttributeError:
#    pass
#else:
#    ssl._create_default_https_context = _create_unverified_https_context
#nltk.download()

import numpy as np
#Supress TF errors:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed
import tensorflow as tf


from nltk.stem import WordNetLemmatizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
tf.keras.utils.disable_interactive_logging()
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

#Import json file.
data_file = open('intents.json').read()
data = json.loads(data_file)

#Create data for X and Y:

words = []
classes = []
data_x = []
data_y = []

for intent in data["intents"]:
  for pattern in intent["patterns"]:
    tokens = nltk.word_tokenize(pattern)
    words.extend(tokens)
    data_x.append(pattern)
    data_y.append(intent["tag"])

  if intent["tag"] not in classes:
    classes.append(intent["tag"])

#Get stem of words
lemmatizer = WordNetLemmatizer()

#Lem the words and conver to lowercase
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
#sort the words and make sure no duplication
words = sorted(set(words))
classes = sorted(set(classes))

#Text to numbers (Bag of Words):
training = []
out_empty = [0] * len(classes)
#BoW:
for idx, doc in enumerate(data_x):
  bow = []
  text = lemmatizer.lemmatize(doc.lower())
  for word in words:
    bow.append(1) if word in text else bow.append(0)

  output_row = list(out_empty)
  output_row[classes.index(data_y[idx])] = 1

  training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:,0]))
train_y = np.array(list(training[:,1]))

#Neural Net:

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',
             optimizer=adam,
             metrics=["accuracy"])
#print(model.summary())

model.fit(x=train_x, y=train_y, epochs=150, verbose=1)

#Preprocess user input:
def clean_text(text):
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens

def bag_of_words(text, vocab):
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens:
    for idx, word in enumerate(vocab):
      if word == w:
        bow[idx] = 1
  return np.array(bow)

def pred_class(text, vocab, labels):
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.5
  y_pred = [[indx, res] for indx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list

def get_response(intents_list, intents_json):
  if len(intents_list) == 0:
    result = "Sorry! I don't understand."
  else:
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
      if i["tag"] == tag:
        result = random.choice(i["responses"])
        break
  return result

print("Hello User, please ask me a question.\nPress 0 if you don't want to chat anymore.")
while True:
  message = input("")
  if message == "0":
    break
  intents = pred_class(message, words, classes)
  result = get_response(intents, data)
  print(result)

