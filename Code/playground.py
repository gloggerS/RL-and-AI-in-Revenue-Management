import numpy as np
from keras.layers import Dense
from keras.models import Sequential

from keras.utils.vis_utils import plot_model

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(10,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='categorical_crossentropy')

a = np.array(model.get_weights())         # save weights in a np.array of np.arrays
model.set_weights(a + 1)                  # add 1 to all weights in the neural network
b = np.array(model.get_weights())         # save weights a second time in a np.array of np.arrays
print(b - a)
