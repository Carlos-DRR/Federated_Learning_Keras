from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

class Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
        self.dataset_size = len(dataset)
    
    def update_model(self, new_model):
        self.model = new_model
    
    def get_dataset_size(self):
        return self.dataset_size
    def get_client_id(self):
        return self.id
    def preprocess_dataset(self, dataset):
        X = self.dataset.iloc[:, 0:len(dataset.columns) - 1].values
        y = self.dataset.iloc[:, len(dataset.columns) - 1].values
        
        scaler = StandardScaler()
        le = LabelEncoder()
        
        X = scaler.fit_transform(X)
        y = le.fit_transform(y)
        return X, y
        
    def train(self, epochs = 10, batch_size=None):
        if hasattr(self, 'model'):
            X, y = self.preprocess_dataset(self.dataset)
            #print("Pesos iniciais")
            #print(self.model.get_weights())
            self.model.fit(X, y, epochs = epochs, batch_size=batch_size)
        else:
            print('O cliente ' + str(self.id) + ' não possui modelo de AM')
        
    def get_model(self):
        return self.model
    
    def get_metrics(self, validation_set):
        X_test, y_test = self.preprocess_dataset(validation_set)
        y_pred = self.model.predict(X_test)
        y_pred_binary = np.around(y_pred)
        return classification_report(y_test, y_pred_binary, digits=4)
 
'''   
initializer1 = tf.keras.initializers.RandomUniform(seed=2)
initializer1._random_generator._force_generator = True
bias1 = tf.keras.initializers.Zeros()

model1 = keras.models.Sequential([
    keras.layers.Dense(30, 
                       input_dim=12, 
                       activation = 'relu',
                       kernel_initializer = initializer1,
                       bias_initializer = bias1),
    keras.layers.Dense(15,
                       activation = 'relu',
                       kernel_initializer = initializer1,
                       bias_initializer = bias1),
    keras.layers.Dense(1, 
                       activation = 'sigmoid',
                       kernel_initializer = initializer1,
                       bias_initializer = bias1),
    
])

model1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy')

base = pd.read_csv('C:/Users/carlo/Desktop/Mestrado/Experimento FL/Etapa 1 - Escolha de modelo e data mining/IoTID20/IoTID20_preprocessada.csv')

train_dataset, test_dataset = train_test_split(base, test_size=0.2)

client1  = Client(1, train_dataset)

client1.update_model(model1)
client1.train_model(20)
print(client1.get_metrics(test_dataset))

print(client1.get_model_weights())'''
