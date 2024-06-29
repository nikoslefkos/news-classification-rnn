#imports
import json
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, SimpleRNN,MaxPooling1D, GlobalMaxPooling1D, GlobalMaxPool1D, LSTM
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Dropout
from keras.layers import Dense, LSTM, Embedding
from tensorflow import keras
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical


df=pd.read_json('C:/Users/Dell/Desktop/News_Category_Dataset_v3.json',lines=True)
df=df.drop(columns=['authors','link','date']) #removing unnecessary columns from the dataset
df.head() 

input_df=df.copy()
input_df['full_text'] = input_df['headline'] + ' ' + input_df['short_description'] #merging headline with the short description into one column
input_df.drop(['headline','short_description'], inplace=True, axis=1) 
input_df['word_count'] = input_df['full_text'].apply(lambda x: len(str(x))) 
input_df.head()

#function for removing special characters and digits
def keep_data(text):
    cleaned_text = ''.join(char if char.isalpha() or char.isspace() else ' ' for char in text)
    return cleaned_text

#function for text preprocessing
def preprocess_text(text):
    text = keep_data(text) 
    text = text.lower() 
    words = text.split()
    preprocessed_text = ' '.join(words)
    return preprocessed_text

input_df['full_text'] = input_df['full_text'].apply(lambda x: preprocess_text(str(x)))

#function for merging similar categories
def group_categories(text):
    first_array = ["QUEER VOICES", "BUSINESS", "PARENTS", "BLACK VOICES", "THE WORLDPOST", "STYLE", "GREEN", "TASTE", "WORLDPOST", "SCIENCE", "TECH", "MONEY", "ARTS", "COLLEGE", "LATINO VOICES", "CULTURE & ARTS", "FIFTY", "GOOD NEWS"]
    second_array = ["GROUPS VOICES", "BUSINESS & FINANCES", "PARENTING", "GROUPS VOICES", "WORLD NEWS", "STYLE & BEAUTY", "ENVIRONMENT", "FOOD & DRINK", "WORLD NEWS", "SCIENCE & TECH", "SCIENCE & TECH", "BUSINESS & FINANCES", "ARTS & CULTURE", "EDUCATION", "GROUPS VOICES", "ARTS & CULTURE", "MISCELLANEOUS",  "MISCELLANEOUS"]
    
    if text in first_array:
        index = first_array.index(text)
        return second_array[index]
    else:
        return text
    

input_df['category'] = input_df['category'].apply(lambda x: group_categories(str(x)))
print(len(input_df['category'].unique()))

# one hot encoding using keras tokenizer and pad sequencing
X = input_df['full_text']
encoder = LabelEncoder()
y = encoder.fit_transform(input_df['category'])
print("shape of input data: ", X.shape)
print("shape of target variable: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print(type(X_test))
print(type(X_train))
tokenizer = Tokenizer(num_words=10000, oov_token='<00V>') 
tokenizer.fit_on_texts(X_train)

#conversion of strings into sequences in the form of integer arrays
train_seq = tokenizer.texts_to_sequences(X_train)
train_padseq = pad_sequences(train_seq, maxlen=130) 

test_seq = tokenizer.texts_to_sequences(X_test)
test_padseq = pad_sequences(test_seq, maxlen=130)

word_index = tokenizer.word_index
max_words = 80000  # total number of words to consider in embedding layer
total_words = len(word_index)
num_classes = len(input_df['category'].unique())
maxlen = 130 # max length of sequence 
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print("Length of word index:", total_words)

#creating LSTM model
model=Sequential([
    Embedding(total_words, 100, input_length=maxlen),
    LSTM(64,dropout=0.2,recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
print(model.summary())

model.compile(loss="mse",
                   optimizer="rmsprop",
                   metrics=["mae"])

#training
history = model.fit(train_padseq,
                         y_train,
                         batch_size=64,
                         epochs=10,
                         validation_split=0.2,
                         shuffle=True)
test_loss3, test_acc3 = model.evaluate(test_padseq, y_test, verbose=0)
print("test loss and accuracy:", test_loss3, test_acc3)

import matplotlib.pyplot as plt


# Plot MAE
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('MAE vs. Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()