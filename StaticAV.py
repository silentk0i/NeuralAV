import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import datasets, layers, models, optimizers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

peh_data = pd.read_csv('PE_Header.csv')

peh_data.info()

x_data = peh_data.iloc[:, 2:].values # extract all columns except the first two

y_data = peh_data.iloc[:, 1].values # extract first column for Type

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

# Normalization
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = models.Sequential()
model.add(layers.Normalization(input_shape = [52,], axis = None))
model.add(layers.Dense(2, activation='sigmoid', activity_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(layers.Dense(1, activation = 'sigmoid', activity_regularizer=tf.keras.regularizers.L2(0.01)))
model.summary()

adam = optimizers.Adam(learning_rate=0.3)

model.compile(optimizer='adam',
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=300,
                    validation_data=(x_test, y_test))

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