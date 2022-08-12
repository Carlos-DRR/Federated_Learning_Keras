import os
import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from client import Client
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
from server import Server
from utils import Utils
import copy


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
])

'''

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

#0 é anomalia 1 é normal

base = pd.read_csv('C:/Users/carlo/Desktop/Mestrado/Experimento FL/Etapa 1 - Escolha de modelo e data mining/IoTID20/IoTID20_preprocessada.csv')

                 #83 17
train_dataset, test_dataset = train_test_split(base, test_size=0.1, stratify=base['Label'])

utl = Utils(train_dataset, 'Label', [[70,30], [30, 70], [50, 50], [100, 0], [0, 100]])

groups_datasets_list = utl.get_samples_by_proportions()
#groups_datasets_list = utl.get_stratified_sample()

client1  = Client(1, groups_datasets_list[0])
client2  = Client(2, groups_datasets_list[1])
client3  = Client(3, groups_datasets_list[2])
client4  = Client(4, groups_datasets_list[3])
client5  = Client(5, groups_datasets_list[4])
clients_list = [client1, client2, client3, client4, client5]
clients_dataset_sizes = [client1.get_dataset_size(), client2.get_dataset_size(), client3.get_dataset_size(), client4.get_dataset_size(), client5.get_dataset_size()]

            #clients, dataset_sizes, model, clients_local_epochs, batch_size, global_epochs
server = Server(clients_list, clients_dataset_sizes, model, 100, None, 100)

global_model = server.train_federated()

for client in clients_list:
    validation_set = copy.deepcopy(test_dataset)
    print("Client: " + str(client.get_client_id()))
    print(client.get_metrics(validation_set))
    
