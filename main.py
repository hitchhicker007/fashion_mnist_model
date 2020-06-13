import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_imgs, train_lbls), (test_imgs, test_lbls) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_imgs,train_lbls)

test_loss, test_acc = model.evaluate(test_imgs,test_lbls)
print("Tested Acc: ", test_acc)

model.save('trained_model')