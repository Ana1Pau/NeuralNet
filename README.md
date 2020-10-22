# IntroducingNNets

# NeuralNet
Implementation of a NN

！pip3 install utils
import utils

## Cargamos librerias
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *


## Inicializar parametros
def init_params(X, Y, seed=1234):
    np.random.seed(seed)
    n, m = X.shape
    W = np. random.randn(1, m)x(1/np.sqrt(m))
    b = 0
    return W, b

## Combinacion lineal ## First Layer pre-activation
def z(X, W, b):
      return np.dot(X, W.transpose())+ b

## Funcion de activacion
def a(z):
     return 1/(1+np.exp(-z))

def loss(Y, A):
    return -1*np.mean(Y*np.log(A) + (1-Y)*np.log(1-A))

def update_a(X, Y, W, b, A, learning_rate=.001):
     W = W - learning_rate*(1/n)*np.dot((A-Y).transpose(), X)
     b = b - learning_rate*(1/n)*np.sum(A-Y)

def predict(X, W, b, thresh=.5):
    Z = z(X, W, b)
    A = a(Z)
    return A, A >= thresh
## Entrenamiento
def train(X, Y, seed=1234, learning_rate=.001, epochs=20):
    ## Inicializar parametros
    loss_h = []
    W, b = init_params(X, Y, seed)
    for _ in range(epochs):
        # Z
        Z = z(X, W, b)
        # A
        A = a(Z)
        # Perdida
        l = loss(Y, A)
        if not _ % 100:
            print("Perdida: %s | Iter: %s" % (l, _))
        loss_h.append(l)
        # Update parameters
        W, b = update_a(X, Y, W, b, A, learning_rate)
    return W, b, loss_h
    
   ## Cargamos los datos
    data = load_iris()
    X = data.data 
    Y = data.target
   ## Clasificacion binaria
    Y = np.where(Y < 2, 1, 0).reshape(-1, 1)
    X = X
   ## Obtenemos parametros de dimensión.
    n, m = X.shape
   ## Normalizamos los datos
    X = (X - np.sum(X, axis=1).reshape(n, 1)/n)/np.std(X, axis=1).reshape(n, 1)
