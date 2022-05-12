import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print(tf.__version__) #version should be at least 1.15.x
clases = ['Taxi', "Minibus", "Teleferico", "Puma"]
input = pd.read_csv("dataset1.csv")
output = pd.read_csv("output.csv")
#aux = np.array([1, 0, 3, 0, 3, 1, 2, 0, 1, 0, 1, 2])
train_values = input.to_numpy()[0:-3]
train_labels = output.to_numpy()[0:-3]

test_values = input.to_numpy()[-3:]
test_labels = output.to_numpy()[-3:]

print(train_values[0])
print(train_labels[0])
print(test_values[0])
print(test_labels[0])
model = Sequential()
model.add(Dense(22, input_shape=[22]))
model.add(Dense(26, activation='relu'))
model.add(Dense(26, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

log = model.fit(train_values, 
                train_labels, 
                epochs = 30,
                validation_data = (test_values, test_labels))

# make predictions for test data
predictions = model.predict(test_values)

# plot accuracy per epoch
plt.plot(log.history['accuracy'], label='Training acc')
plt.plot(log.history['val_accuracy'], label='Testing acc')
plt.xlabel("epochs")
plt.ylabel("acc")
plt.legend()
plt.grid()

aux = 1
print(predictions)
for i in predictions:
    predicted_label = np.argmax(i)
    print("la prediccion para el dato de entrenamiento " + str(aux) + " es: " + clases[predicted_label])
    aux = aux +1