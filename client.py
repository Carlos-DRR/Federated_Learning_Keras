from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np


class Client:
    def __init__(self, id, dataset):
        self.id = id
        self.dataset = dataset
        self.dataset_size = len(dataset)
        
        #self.scaler = StandardScaler()
        self.le = LabelEncoder()
    
    def update_model(self, new_model):
        self.model = new_model
    
    def get_dataset_size(self):
        return self.dataset_size
    def get_client_id(self):
        return self.id
    def preprocess_dataset(self, dataset):
        X = self.dataset.iloc[:, 0:len(dataset.columns) - 1].values
        y = self.dataset.iloc[:, len(dataset.columns) - 1].values
        
        y = self.le.fit_transform(y)
        return X, y
        
    def train(self, epochs = 10, batch_size=None):
        if hasattr(self, 'model'):
            X, y = self.preprocess_dataset(self.dataset)
            #print("Pesos iniciais")
            #print(self.model.get_weights())
            self.model.fit(X, y, epochs = epochs, batch_size=batch_size, verbose=0)
        else:
            print('O cliente ' + str(self.id) + ' não possui modelo de AM')

    
    def get_metrics(self, validation_set):
        X_test, y_test = self.preprocess_dataset(validation_set)
        print("X do Cliente: " + str(self.id))
        print(X_test[0])
        print("Y do Cliente: " + str(self.id))
        print(y_test[0])
        #print("Modelo final dentro do cliente: ")
        #print(self.model.get_weights())#printa o peso aqui
        y_pred = self.model.predict(X_test)
        y_pred_binary = np.around(y_pred)
        return classification_report(y_test, y_pred_binary, digits=4)
            
    def get_model(self):
        return self.model
 
