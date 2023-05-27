# Named Entity Recognition

## AIM:

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset:
We aim to develop an LSTM-based neural network model using Bidirectional Recurrent Neural Networks for recognizing the named entities in the text. Bidirectional Recurrent Neural Networks connect two hidden layers of opposite directions to the same output. With this form of generative deep learning, the output layer can get information from past and future states simultaneously.


## DESIGN STEPS:

### STEP 1:
Import the necessary packages.

### STEP 2:
Load the dataset, and fill the null values using forward fill

### STEP 3:
Create a list of words, and tags. Also find the number of unique words and tags in the dataset.

### STEP 4:
Create a dictionary for the words and their Index values. Do the same for the tags as well.Train and test the dataset.

### STEP 5:
Perform padding the sequences to acheive the same length of input data.

### STEP 6:
Build a model using Input, Embedding, Bidirectional LSTM, Spatial Dropout, Time Distributed Dense Layers.

### STEP 7:
Compile the model and fit the train sets and validation sets.

### STEP 8
Plot the necessary graphs for analysis. A custom prediction is done to test the model manually.


## PROGRAM:
```python
### Developed By    : Venkatesh E
### Register Number : 212221230119
```
```python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model

data = pd.read_csv("/content/drive/MyDrive/Clg/19AI413 Deep Learning/Datasets/ner_dataset.csv", encoding="latin1")
data.head ()

data = data.fillna (method="ffill") 
data ['Word'].nunique ()
data ['Tag'].nunique ()

words = list (data ['Word'].unique ())
words.append ("ENDPAD")
tags = list (data ['Tag'].unique ()) 

tags

num_words = len (words)
num_tags = len (tags)

num_words

class SentenceGetter (object):
    def __init__ (self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [ (w, p, t) for w, p, t in zip(s ["Word"].values.tolist (),
                                                            s ["POS"].values.tolist (),
                                                            s ["Tag"].values.tolist ())]
        self.grouped = self.data.groupby ("Sentence #").apply (agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next (self):
        try:
            s = self.grouped ["Sentence: {}".format (self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter (data)
sentences = getter.sentences

len (sentences)
sentences [0]

word2idx = {w: i + 1 for i, w in enumerate (words)}
tag2idx = {t: i for i, t in enumerate (tags)}

word2idx

plt.hist([len(s) for s in sentences], bins=50)
plt.show()

X1 = [ [word2idx [w [0]] for w in s] for s in sentences]

type (X1 [0])
X1 [0]
max_len = 50

X = sequence.pad_sequences (maxlen = max_len,
                            sequences = X1, padding = "post",
                            value = num_words-1)

X [0]

y1 = [ [tag2idx [w [2]] for w in s] for s in sentences]

y = sequence.pad_sequences (maxlen = max_len,
                           sequences = y1,
                           padding = "post",
                           value = tag2idx ["O"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

X_train [0]
y_train [0]

input_word = layers.Input (shape = (max_len,))
embedding_layer = layers.Embedding (input_dim = num_words,
                                   output_dim = 50,
                                   input_length = max_len) (input_word)
dropout_layer = layers.SpatialDropout1D (0.1) (embedding_layer) 
bidirectional_lstm = layers.Bidirectional (layers.LSTM (units = 100, 
                                                        return_sequences = True, 
                                                        recurrent_dropout = 0.1)) (dropout_layer)
output = layers.TimeDistributed (layers.Dense (num_tags, activation = "softmax")) (bidirectional_lstm) 
model = Model (input_word, output)

model.summary()

model.compile (optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit (
    x = X_train,
    y = y_train,
    validation_data = (X_test, y_test),
    batch_size = 32, 
    epochs = 3,
)

metrics = pd.DataFrame (model.history.history)
metrics.head ()

metrics [ ['accuracy','val_accuracy']].plot ()

metrics [ ['loss','val_loss']].plot ()

i = 20
p = model.predict (np.array ( [X_test [i]]))
p = np.argmax (p, axis=-1)
y_true = y_test [i]
print ("{:15}{:5}\t {}\n".format ("Word", "True", "Pred"))
print ("-" *30)
for w, true, pred in zip (X_test[i], y_true, p[0]):
    print ("{:15}{}\t{}".format (words[w-1], tags[true], tags[pred]))
```


## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:

![1](https://user-images.githubusercontent.com/94154252/235593087-1c00aa16-66bb-4c44-b1e8-28a73b89cbb7.png)

![2](https://user-images.githubusercontent.com/94154252/235593104-cdf37bfa-8400-4b27-83ba-533b75cf6c30.png)

![3](https://user-images.githubusercontent.com/94154252/235593118-88ada549-87d3-48ab-ad5e-14a68f2d3c53.png)

![4](https://user-images.githubusercontent.com/94154252/235593125-3799312e-5dec-4e77-84ba-22ec0ca2faa7.png)

### Sample Text Prediction:

![5](https://user-images.githubusercontent.com/94154252/235593149-efc9d1c9-d95f-486e-a0ee-9e0441e60ca3.png)

## RESULT:
Thus we have developed a Deep Learning model successfully. 
