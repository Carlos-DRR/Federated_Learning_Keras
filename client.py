from sklearn.metrics import classification_report
import numpy as np


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
    
    def preprocess_dataset(self, validation_set=None):
        if validation_set is None:
            X = self.dataset.iloc[:, 0:len(self.dataset.columns) - 1].values
            y = self.dataset.iloc[:, len(self.dataset.columns) - 1].values
        else:
            X = validation_set.iloc[:, 0:len(validation_set.columns) - 1].values
            y = validation_set.iloc[:, len(validation_set.columns) - 1].values

        return X, y
        
    def train(self, epochs = 10, batch_size=None):
        if hasattr(self, 'model'):
            X, y = self.preprocess_dataset()
            print('O cliente ' + str(self.id) + ' possui ' + str(self.dataset_size) + ' exemplos de treinamento')
            self.model.fit(X, y, epochs = epochs, batch_size=batch_size, verbose=0)
        else:
            print('O cliente ' + str(self.id) + ' não possui modelo de AM')

    
    def get_metrics(self, validation_set):
        X_test, y_test = self.preprocess_dataset(validation_set)
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred_binary = np.around(y_pred)

        return classification_report(y_test, y_pred_binary, digits=4)
            
    def get_model(self):
        return self.model
    

 
