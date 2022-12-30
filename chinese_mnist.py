#!/usr/bin/env python
# coding: utf-8

# # Pràctica 3: chinese_mnist
# 
# En aquest fitxer es presentara tot el codi utilitzat per fer la comparació dels models i fer una classificació numèrica de la base de dades.

# In[181]:


from __future__ import absolute_import, division, print_function, unicode_literals
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display as display


from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2
import os
import matplotlib.image as mpimg
from tensorflow.python import metrics

import logging

from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler


# In[183]:


# Carreguem dataset
dataset = pd.read_csv('chinese_mnist.csv')
print("Dimensionalitat de la BBDD:", dataset.shape)
print("\nTabla de la BBDD:")
display.display(dataset)
sortides = dataset.sort_values('code')['character'].unique()
print(sortides)
print(str(sortides[0]))
print(str(sortides[0]))
print(sortides[0])
print(sortides[0])
print(sortides[0])
print(sortides[0])



 # In[188]:


X = []
y = []
for row in dataset.itertuples():
    suite_id = row[1]
    sample_id = row[2]
    code = row[3]
    file_name = f"input_{suite_id}_{sample_id}_{code}.jpg"
    
    #Guarda la imagen
    img = cv2.imread(f"data/data/{file_name}")

    #Convierte a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Aplicamos threshold
    x = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)[1]
    X.append(x)

    #Hacemos one_hot encode
    one_hot = [0] * 15
    one_hot[code - 1] = 1
    y.append(one_hot)

X = np.array(X)

#salida One Hot Encoded
y = np.array(y)

#Salida sin One Hot Encoded
#Salida sin One Hot Encoded
y_n = dataset['value']
x_n = X
#y = dataset['character']

#dataset One Hot Encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#clase Callback para limitante
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()


#Crear el modelo, este caso tendra 3 capas de 128 neuronas

modelCNN = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=[64, 64, 1], activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),#2,2 es el tamaño de la matriz
    
    tf.keras.layers.Conv2D(64, (3,3), input_shape=[64, 64, 1], activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),#2,2 es el tamaño de la matriz
    
    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dense(128, activation=tf.nn.relu), #1a capa oculta activacion relu
    tf.keras.layers.Dense(128, activation=tf.nn.relu), #2a capa oculta activacion relu
    tf.keras.layers.Dense(128, activation=tf.nn.relu), #2a capa oculta activacion relu
    tf.keras.layers.Dense(15, activation=tf.nn.softmax), #capa de salida 15 salidas posibles
])
modelCNN.compile(
    optimizer = 'adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelCNN.summary()


# In[282]:


historialCNN = modelCNN.fit(X_train, y_train, epochs = 50, callbacks=myCallback())
print("RNC entrenada")

#Clase para definir el servidor http. Solo recibe solicitudes POST.
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        print("Peticion recibida")

        #Obtener datos de la peticion y limpiar los datos
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)

        #Realizar transformacion para dejar igual que los ejemplos que usa MNIST
        arr = np.fromstring(data, np.float32, sep=",")
        arr = arr.reshape(64,64)
        arr = np.array(arr)
        arr = arr.reshape(1, 64, 64, 1)

        #Realizar y obtener la prediccion
        prediction_values = modelCNN.predict(arr)
        prediction = np.argmax(prediction_values)
        print("Prediccion final: " + str(prediction))
        prediction = sortides[prediction]
        print("Prediccion final: " + prediction)


        # Regresar respuesta a la peticion HTTP
        self.send_response(200)
        # Evitar problemas con CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(prediction.encode())

# Iniciar el servidor en el puerto 8000 y escuchar por siempre
# Si se queda colgado, en el admon de tareas buscar la tarea de python y finalizar tarea
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()
# In[ ]:




