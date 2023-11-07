import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import datasets, layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import sys

train_data = pd.read_json('ember2018/train_features_0.jsonl', lines=True).sample(frac=0.01, random_state=1)

x_data = train_data.iloc[:, 6:].values # extract all columns except the first six

y_data = train_data.iloc[:, 3].values # extract first column for Type

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = models.Sequential()
model.add(layers.Normalization(input_shape=[500,], axis=None))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

adam = optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=adam,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=300,
                    validation_data=(x_test, y_test),
                    batch_size=32)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xlim([0, 300])
plt.ylim([0, 2])
plt.legend(loc='lower right')

train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test,  y_test)

print("Training Error:", str(train_acc))
print("Testing Error: ", str(test_acc))