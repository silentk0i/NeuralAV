import ember
import numpy as np
from tqdm import tqdm
from tensorflow.keras import datasets, layers, models, optimizers
from sklearn.preprocessing import StandardScaler

#create the train/test split (change PATH to where teh ember2018 file is stored tho)
X_train, y_train, X_test, y_test = ember.read_vectorized_features("PATH")

train_rows = (y_train != -1)
X = X_train[train_rows]
Y = y_train[train_rows]

#Normalize data between 0-1 for better output
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.fit_transform(X_test)


#define model architecture 
model = models.Sequential()
model.add(layers.Dense(2400, activation='relu', input_dim=2351))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1200, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1200, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

#compile model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#train model for 20 epochs
model.fit(x=X_scaled, y=Y, batch_size=128, epochs=20, verbose=1, shuffle=True, validation_split = 0.01)

score = model.evaluate(x=X_test_scaled, y=y_test, batch_size=32, verbose=1)