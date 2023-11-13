from torchvision.datasets import FER2013
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import get_loader
from network import Network
from plot_losses import PlotLosses

def validation_step(val_loader, net, cost_function):
    '''
        Realiza un epoch completo en el conjunto de validación
        args:
        - val_loader (torch.DataLoader): dataloader para los datos de validación
        - net: instancia de red neuronal de clase Network
        - cost_function (torch.nn): Función de costo a utilizar

        returns:
        - val_loss (float): el costo total (promedio por minibatch) de todos los datos de validación
    '''
    val_loss = 0.0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch['transformed']
        batch_labels = batch['label']
        device = net.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            #Dabbura, I. (2018, abril 1). Coding neural network — forward propagation and backpropagtion. Towards Data Science. https://towardsdatascience.com/coding-neural-network-forward-propagation-and-backpropagtion-ccf8cf369f76
            def sigmoid(Z):
                A = 1 / (1 + np.exp(-Z))
                return A, Z

            def relu(Z):
                A = np.maximum(0, Z)
                return A, Z
            
            def forwardpass(A_prev, W, b):
                Z = np.dot(W, A_prev) + b
                cache = (A_prev, W, b)
                return Z, cache
            
            def activacionlineal(A_prev, W, b, funcion):
                assert funcion == "sigmoide" or \
                    funcion == "relu"

                if funcion == "sigmoide":
                    Z, linear_cache = activacionlineal(A_prev, W, b)
                    A, activation_cache = sigmoid(Z)

                elif funcion == "relu":
                    Z, linear_cache = activacionlineal(A_prev, W, b)
                    A, activation_cache = relu(Z)

                assert A.shape == (W.shape[0], A_prev.shape[1])

                cache = (linear_cache, activation_cache)
                return A, cache
            
            def modeloforward(X, parameters, capasocultasfuncion="relu"):
                A = X                           
                caches = []                     
                L = len(parameters) // 2        

                for l in range(1, L):
                    A_prev = A
                    A, cache = activacionlineal(
                        A_prev, parameters["W" + str(l)], parameters["b" + str(l)],
                        funcion=capasocultasfuncion)
                    caches.append(cache)

                AL, cache = activacionlineal(
                    A, parameters["W" + str(L)], parameters["b" + str(L)],
                    funcion="sigmoide")
                caches.append(cache)

                assert AL.shape == (1, X.shape[1])
                return AL, caches
            
            forwardpass()
            activacionlineal()
            modeloforward()
            
            def costo(AL, y):
                m = y.shape[1]              
                cost = - (1 / m) * np.sum(
                    np.multiply(y, np.log(AL)) + np.multiply(1 - y, np.log(1 - AL)))
                return cost
        
            costo()
            
            
    # TODO: Regresa el costo promedio por minibatch
            costo()
        return ...

def train():
    # Hyperparametros
    learning_rate = 1e-4
    n_epochs=100
    batch_size = 256

    # Train, validation, test loaders
    train_dataset, train_loader = \
        get_loader("train",
                    batch_size=batch_size,
                    shuffle=True)
    val_dataset, val_loader = \
        get_loader("val",
                    batch_size=batch_size,
                    shuffle=False)
    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    # Instanciamos tu red
    modelo = Network(input_dim = 48,
                     n_classes = 7)

    # TODO: Define la funcion de costo
    criterion = ...

    # Define el optimizador
    optimizer = ...

    best_epoch_loss = np.inf
    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch['transformed']
            batch_labels = batch['label']
            # TODO Zero grad, forward pass, backward pass, optimizer step
            ...

            # TODO acumula el costo
            ...

        # TODO Calcula el costo promedio
        train_loss = ...
        val_loss = validation_step(val_loader, modelo, criterion)
        tqdm.write(f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        # TODO guarda el modelo si el costo de validación es menor al mejor costo de validación
        ...
        plotter.on_epoch_end(epoch, train_loss, val_loss)
    plotter.on_train_end()

if __name__=="__main__":
    train()