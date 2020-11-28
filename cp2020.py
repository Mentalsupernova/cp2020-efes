%python

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

%python
prefix = "/mnt/data/s3/cp2020/"
dataset = pd.read_csv(prefix + "big_train_no_codes.csv",encoding="cp1251",delimiter=";")

%python
dataset.isna().sum()
#dataset = dataset.drop(columns= ['Республика Алтай','Республика Тыва','Камчатский край'])
dataset = dataset.fillna(1)
def getStringInData(cellString):
    if cellString == '#VALUE!':
        cellString = 0
        return cellString
    else:
        pass
    
cols = list(dataset.columns)
dataset[cols] = dataset[cols].apply(pd.to_numeric,errors='coerce')
#dataset = dataset.apply()

#dataset = pd.to_numeric(dataset)
dataset = dataset.astype(int)


%python
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


%python
train_labels = train_dataset.pop('Энергопотребление (анализируемый показатель), млн. кВт в час')
test_labels = test_dataset.pop('Энергопотребление (анализируемый показатель), млн. кВт в час')

%python
def build_model():
  model = keras.Sequential([
    keras.layers.Flatten(input_shape=[len(train_dataset.keys())]),
    layers.Dense(150, activation='relu'),
    layers.Dense(150, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

  loses = ['mse','categorical_crossentropy']
  metrics = [['mae', 'mse'],]
  model.compile(loss=loses[1],
                optimizer=optimizer,
                metrics=metrics[0])
  return model

  %python
model = build_model()

%python
model.summary()

%python
example_batch = train_dataset[:10]
example_result = model.predict(example_batch)
example_result


#simple ver
%python
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])
#with early stop
  %python
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

%python
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


%python
z.show(test_predictions)