import os
import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from client import Client
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

class Server:
    '''
    clients = [client1, client2, client3,...,clientn]
    
    dataset_sizes = [client1_dataset_size, client2_dataset_size, client3_dataset_size,..., clientn_dataset_size]
    
    '''
    def __init__(self, clients, dataset_sizes, model, clients_local_epochs, batch_size, global_epochs):
        self.clients = clients
        self.server_model = model
        self.clients_local_epochs = clients_local_epochs
        self.global_epochs = global_epochs
        self.batch_size = batch_size
        self.dataset_sizes = dataset_sizes
        self.total_datasets_size = self.get_total_datasets_size()
        self.updated_clients_models(self.server_model.get_weights())
    
    #pega a soma do tamanho dos datasets dos clientes
    def get_total_datasets_size(self):
        sum = 0
        for size in self.dataset_sizes:
            sum = sum + size
        return sum
    
    def updated_clients_models(self, weights=0):
        for client in self.clients:
            new_model = keras.models.clone_model(self.server_model)
            if weights != 0:
                new_model.set_weights(weights)
            new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')
            client.update_model(new_model)
    
    def federated_avarage(self):
        #models = [model1, model2]
        models = [client.get_model() for client in self.clients]
        models_weights = [model.get_weights() for model in models]
        
        #print("Pesos antes da federacao")
        #print(models_weights)
        
        #frac = 75652/151305
        
        #primeiro multiplica os pesos de todas as RNA's pela sua ponderação (N_k/N)
        for i in range(0, len(models_weights)):
        #for model_weight in models_weights:
            frac = self.dataset_sizes[i] /self.total_datasets_size
            for weights in models_weights[i]:
              weights *= frac
        
        #soma os pesos de todos os modelos
        new_weights = []
        #print('antes da soma')
        #print(models_weights)
        new_weights.append(
                [sum(weights)
                    for weights in zip(*models_weights)])
        
        #print('depois da soma')
        #print(*new_weights)
        return new_weights
        #model3.set_weights(*new_weights)
    def train_federated(self):
        for i in range(0, self.global_epochs):
            print("Época Global: " + str(i+1))
            for client in self.clients:
                print("Client: " + str(client.get_client_id()))
                client.train(self.clients_local_epochs, self.batch_size)
            #fedavg
            new_weights = self.federated_avarage()
            
            #print("Pesos federados")
            #print(new_weights)
            
            #new_global_model = keras.models.clone_model(self.server_model)
            #new_global_model.set_weights(*new_weights)
            #self.server_model = new_global_model
            self.updated_clients_models(*new_weights)
        
        return self.server_model


    def preprocess_dataset(self, dataset):
        X = dataset.iloc[:, 0:len(dataset.columns) - 1].values
        y = dataset.iloc[:, len(dataset.columns) - 1].values
        
        scaler = StandardScaler()
        le = LabelEncoder()
        
        X = scaler.fit_transform(X)
        y = le.fit_transform(y)
        return X, y
    def get_metrics(self, validation_set):
        X_test, y_test = self.preprocess_dataset(validation_set)
        y_pred = self.server_model.predict(X_test)
        y_pred_binary = np.around(y_pred)
        return classification_report(y_test, y_pred_binary, digits=4)


initializer = tf.keras.initializers.RandomUniform(seed=2)
initializer._random_generator._force_generator = True
bias = tf.keras.initializers.Zeros()
'''
model = keras.models.Sequential([
    keras.layers.Dense(2, 
                       input_dim=12, 
                       activation = 'relu',
                       kernel_initializer = initializer,
                       bias_initializer = bias),
    keras.layers.Dense(1, 
                       activation = 'sigmoid',
                       kernel_initializer = initializer,
                       bias_initializer = bias),
])'''

model = keras.models.Sequential([
    keras.layers.Dense(30, 
                       input_dim=12, 
                       activation = 'relu',
                       kernel_initializer = initializer,
                       bias_initializer = bias),
    keras.layers.Dense(15,
                       activation = 'relu',
                       kernel_initializer = initializer,
                       bias_initializer = bias),
    keras.layers.Dense(1, 
                       activation = 'sigmoid',
                       kernel_initializer = initializer,
                       bias_initializer = bias),
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')

base = pd.read_csv('C:/Users/carlo/Desktop/Mestrado/Experimento FL/Etapa 1 - Escolha de modelo e data mining/IoTID20/IoTID20_preprocessada.csv')

train_dataset, test_dataset = train_test_split(base, test_size=0.2, stratify=base['Label'])

client1  = Client(1, train_dataset)
client2  = Client(2, train_dataset)
client3  = Client(3, train_dataset)
clients_list = [client1, client2, client3]
clients_dataset_sizes = [client1.get_dataset_size(), client2.get_dataset_size(), client3.get_dataset_size()]

            #clients, dataset_sizes, model, clients_local_epochs, batch_size, global_epochs
server = Server(clients_list, clients_dataset_sizes, model, 10, None, 2)

global_model = server.train_federated()

print(server.get_metrics(test_dataset))
#MANDAR TESTAR P/ CADA CLIENTE FOR CLIENTE IN CLIENTES TESTE A VALIDAÇÃO