import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from nltk.corpus import stopwords
import re, nltk, string, collections

'''
Load in the data
'''
# Load in training data:
df = pd.read_csv('/Users/johnwillis/Desktop/PycharmProjects/data310/m5/final/data/train.csv')
df = df.dropna()
df = df.reset_index()

df['title'] = [i + ' ' for i in df['title']] # add space to sep headline from authors
df['title']= df['title'] + df['author'] # add author to headline
print(df['title'][1])

X = df['title']
y = df.label

# Load in  testing data:
test_data = pd.read_csv('/Users/johnwillis/Desktop/PycharmProjects/data310/m5/final/data/test.csv')
test_labels = pd.read_csv('/Users/johnwillis/Desktop/PycharmProjects/data310/m5/final/data/submit.csv')
test_data['label'] = test_labels['label']
test_data = test_data.dropna()
test_data = test_data.reset_index()

test_data['title'] = [i + ' ' for i in test_data['title']]
#test_data['title']=test_data['title']+test_data['author']

X_test = test_data['title']
y_test = test_data.label

'''
Clean the training data
'''

def CleanText(text_bank):
    ''' Clean text prior to tokenization: remove punc/spec char, lowercase,
    split, stem/remove stopwords, rejoin strings and append to corpus.
    The corpus is cleaned text data - useful for WordClouds/graphics.
    '''
    import re
    from nltk.stem import PorterStemmer
    ps = PorterStemmer()

    corpus = []
    for i in range(len(text_bank)):
        print(i/len(text_bank)*100,'% processed')
        review = (re.sub('[^a-zA-Z]',' ', text_bank[i]))
        review = review.lower().split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

X_corpus = CleanText(X)
X_test_corpus = CleanText(X_test)

# Viewing cleaned text:
for i in range(2):
    print('Original sentence: ', X[i])
    print('Processed sentence: ', X_corpus[i])

print(len(X_corpus), len(X_test_corpus))

# Encode text into a one-hot tensor:
VOCAB_SIZE = 5000
onehot_representation=[one_hot(words, VOCAB_SIZE) for words in X_corpus]
onehot_representation_test=[one_hot(words, VOCAB_SIZE) for words in X_test_corpus]

# Pad sequences to be the same length:
PADDING_LENGTH=25
padded_title=pad_sequences(onehot_representation, padding='pre' , maxlen=PADDING_LENGTH)
padded_title_test=pad_sequences(onehot_representation_test, padding='pre' , maxlen=PADDING_LENGTH)

# Prepare data for training:
X_train = np.array(padded_title)
y_train = np.array(y)
len(X_train)
len(y_train)

# Create training split:
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=.2)

# Creating the model:
EMBEDDING_OUTPUT_DIM = 40
model = tf.keras.Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_OUTPUT_DIM, input_length=PADDING_LENGTH),
    Dropout(0.5),
    Bidirectional(LSTM(75)),
    Dropout(0.4),
    Dense(32,activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Model training:
history = model.fit(X_train, y_train,
                    validation_data=(X_val,y_val),
                    epochs=5,
                    batch_size=128)

# Plot training results
from matplotlib import pyplot as plt
plt.figure(figsize=(8,8))
plt.plot(history.history['loss'],'--')
plt.plot(history.history['val_loss'],'--')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Performance', fontsize=16)
plt.ylabel('Loss and Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.legend(['Train loss', 'Val loss', 'Train acc', 'Val acc'], loc='best')
plt.show()

'''
Evaluate model
'''

X_test = np.array(padded_title_test)
y_test = np.array(y_test)

from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)
loss, acc = model.evaluate(X_test, y_test)

import seaborn as sns
plt.figure(figsize = (10,8), facecolor=None)
sns.heatmap(cm, annot=True, cmap=plt.cm.coolwarm, linewidths=0, fmt="d")
plt.title(label=("Model's Performance on Test Data"), fontsize=16)
plt.xlabel('Predicted Label', fontsize=16)
plt.ylabel('Actual Label', fontsize=16)
plt.tight_layout(pad=1)
plt.show()

print('The model correctly labeled',np.round(acc*100, 2),'% of the testing data')

'''
WordClouds of train and test data
'''
# Train:
from wordcloud import WordCloud, STOPWORDS
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      stopwords=stopwords,
                      min_font_size=10).generate(' '.join(X_corpus))
plt.figure(figsize=(8, 8), facecolor=None)
plt.title('Train WordCloud')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# Test:
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=800,
                      background_color='black',
                      stopwords=stopwords,
                      min_font_size=10).generate(' '.join(X_test_corpus))
plt.figure(figsize=(8, 8), facecolor=None)
plt.title('Test WordCloud')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

'''
Further Data Exploration
'''

'''
See what the model is getting right/wrong
'''
idxRight = [i for i in range(len(y_pred)) if y_pred[i] == y_test[i]]
idxWrong = [i for i in range(len(y_pred)) if y_pred[i] != y_test[i]]

# Print some headlines that were incorrectly classified:
for i in range(5):
    print(X_test_corpus[idxWrong[i]])
    if y_pred[idxWrong[i]] == 1:
        print('Predicted Label: Unreliable')
    else:
        print('Predicted Label: Reliable')

'''
Analyzing word groupings
'''
# Create separate indexes of fake/real training and testing headlines
test_idxFake = [i for i in range(len(test_data['label'])) if (test_data['label'][i]) == 1] # idx of fake training headlines
test_idxReal = [i for i in range(len(test_data['label'])) if (test_data['label'][i]) == 0]

train_idxFake = [i for i in range(len(df['label'])) if (df['label'][i]) == 1]  # idx of fake training headlines
train_idxReal = [i for i in range(len(df['label'])) if (df['label'][i]) == 0]  # idx of fake training headlines

# Print out some fake and real headlines:
for i in range(5): # FAKE
    print('Fake Train Headline ',i,":",df['title'][train_idxFake[i]])
    print('Fake Test Headline ',i,":",test_data['title'][test_idxFake[i]])

for i in range(5): # REAL
    print('Real Train Headline ', i, ":", df['title'][train_idxReal[i]])
    print('Real Test Headline ', i, ":", test_data['title'][test_idxReal[i]])

from nltk.util import ngrams # library for determining ngrams

# Train text bigrams:
train_corpusFake = [X_corpus[i] for i in train_idxFake]
train_corpusReal = [X_corpus[i] for i in train_idxReal]

train_FakeBigrams = ngrams((' '.join(train_corpusFake).split(' ')), 2)
train_RealBigrams = ngrams((' '.join(train_corpusReal).split(' ')), 2)

tfbCount = collections.Counter(train_FakeBigrams)
tfb_top = tfbCount.most_common(10)

trbCount = collections.Counter(train_RealBigrams)
trb_top = trbCount.most_common(10)
trb_top_df = pd.DataFrame(trb_top, columns=('bigram', 'count'))


import plotly.express as px
fig = px.bar(trb_top_df, x='bigram', y='count',title='Counts of top bigrams', template='plotly_white', labels={'ngram': 'Bigram', 'count': 'Count'})
fig.show()