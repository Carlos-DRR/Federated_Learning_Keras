import os
import tensorflow as tf
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" #If the line below doesn't work, uncomment this line (make sure to comment the line below); it should help.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


from client import Client
from tensorflow import keras
import pandas as pd
from server import Server
from utils import Utils
from sklearn.model_selection import StratifiedKFold

initializer = tf.keras.initializers.RandomUniform(seed=2)
initializer._random_generator._force_generator = True
bias = tf.keras.initializers.Zeros()
max_clients = 5

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

skf = StratifiedKFold(n_splits = 10)

base = base.iloc[:, :]
y = base.iloc[:, len(base.columns) - 1]

skf_iteration = 1
with open('results.txt', 'w') as f:
    for train_index, test_index in skf.split(base, y):
        train_dataset = base.iloc[train_index]
        test_dataset = base.iloc[test_index]
        # O tamanho de [[70,30], [30, 70], [50, 50], [100, 0], [0, 100]] deve ser o mesmo da qtd de clientes
        utl = Utils(train_dataset, 'Label', [[70,30], [30, 70], [50, 50], [100, 0], [0, 100]])
        groups_datasets_list = utl.get_stratified_sample()
        print(len(groups_datasets_list))
        #Cria clientes
        clients_list = []
        for i in range(0, max_clients):
            client = Client(i + 1, groups_datasets_list[i])
            clients_list.append(client)
            
        #Pega o tamanho do dataset dos clientes
        clients_dataset_sizes = []
        for client in clients_list:
            clients_dataset_sizes.append(client.get_dataset_size())
        
        server = Server(clients_list, clients_dataset_sizes, model, 100, None, 100)
    
        server.train_federated()
                
        f.write("Iteração do Cross Validation Estratificado: " + str(skf_iteration))
        f.write('\n')
            
        print("Iteração do Cross Validation Estratificado: " + str(skf_iteration))
        for client in clients_list:
            #f.write("Client: " + str(client.get_client_id()))
            #f.write('\n')
            #f.write(client.get_metrics(validation_set))
            #f.write('\n')
            print("Client: " + str(client.get_client_id()))
            print(client.get_metrics(test_dataset))
    
        f.write("Modelo do servidor")
        f.write('\n')
        f.write(server.get_metrics(test_dataset))
        f.write('\n')
        print("Modelo do servidor")
        print(server.get_metrics(test_dataset))
        
        
        skf_iteration = skf_iteration + 1
    
