import numpy as np
import matplotlib.pyplot as plt 
import keras
import sklearn.model_selection

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# generate a sequence of real values between 0 and 1.
def generate_sequence(length):
	return np.array([i/float(10) for i in range(length)])
 
X_train = generate_sequence(100)
#print (X_train)
y_train = ['%.1f' %(i+0.1) for i in X_train]
y_train = [float(i) for i in y_train]
#print (y_train)
#plt.scatter(X_train,y_train)
#plt.show()
X_test = generate_sequence(20)
X_test = [i+float(10) for i in X_test]
#print (X_test)
#print (X_train.shape[1])

model = keras.models.Sequential()
model.add(keras.layers.normalization.BatchNormalization(input_shape = (1,)))
model.add(keras.layers.core.Dense(32, activation='relu'))
#model.add(keras.layers.core.Dropout(rate=0.5))
#model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
#model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.normalization.BatchNormalization())
model.add(keras.layers.core.Dense(32, activation='relu'))
#model.add(keras.layers.core.Dropout(rate=0.5))
model.add(keras.layers.core.Dense(1,activation='softmax'))
model.compile(loss="mse", optimizer="rmsprop",metrics=["accuracy"])
print(model.summary())

model.fit(X_train, y_train, batch_size = 100, epochs=100, verbose=1)
y_test = model.predict(X_test)
print (y_test)