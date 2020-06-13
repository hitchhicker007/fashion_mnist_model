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

model = keras.models.load_model('trained_model')
predictions = model.predict(test_imgs)
plt.figure(figsize=(5,5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_imgs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_lbls[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()