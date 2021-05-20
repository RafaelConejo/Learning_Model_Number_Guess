import keras
import tensorflow as tf
import numpy as np

# MNIST dataset es una base de datos con 60.000 imagenes de número del 0-9
MNIST_dataset = tf.keras.datasets.mnist
"""
tf.keras.datasets.mnist.load_data(
    path='mnist.npz'
)
"""

# Realizo las separaciones del train y del test
(x_train, y_train), (x_test, y_test) = MNIST_dataset.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# 28x28 es la escala de gris en las imagenes
# assert x_train.shape == (60000, 28, 28)
for train_set in range(len(x_train)):
    for fila in range(28):
        for valor in range(28):
            if x_train[train_set][fila][valor] != 0:
                x_train[train_set][fila][valor] = 1


# Establezco que el modelo sea secuencial, agrupando en pila
modelo = tf.keras.models.Sequential()

# Añado configuraciones al modelo
# Flatten para que se conserve el peso de cada dato por el cambio de formato de datos
modelo.add(tf.keras.layers.Flatten())
# Establezco el tamaño de las matrices de salida a 128
modelo.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
modelo.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
modelo.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Para compilar necesito optimizar en kelas
# Adam es un algoritmo de optimización para el descenso de gradiente estocástico para train en deep learning
# el loss Calcula la pérdida de entropía cruzada entre las etiquetas y las predicciones.
# metrics Calcula la frecuencia con la que las predicciones son iguales a las etiquetas.
modelo.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Llamamos a fit (), que entrenará el modelo dividiendo los datos en "lotes" de tamaño batch_size, 
# e iterando repetidamente sobre todo el conjunto de datos para un número determinado de épocas (epochs).
modelo.fit(x_train, y_train, epochs = 5)

modelo.save('mi_modelo.model')

print("Modelo guardado")


