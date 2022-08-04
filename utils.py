import numpy as np

class Utils:
    def __init__(self, dataset, target_attribute):
        self.dataset = dataset
        self.get_max_samples_by_label(target_attribute)
    

    # recebe dist = [['anomalia', classe1%], ['normal', classe2%],... ,['ataque',classen%]]
    # ex: classe1% = 0.70
    # listas distribution_list e labels  tem o mesmo tamanho
    # ex: labels = [anomalia, normal] distribution_list
    def print_dataset_size(self):
        print('O dataset tem ' + str(len(self.dataset)) + ' registros')
    
    def get_max_samples_by_label(self, target_attribute):
        total_records = len(self.dataset)
        classes_percentage = list((self.dataset[target_attribute].value_counts()/self.dataset[target_attribute].count())*100)
        classes_samples = []
        for percentage in classes_percentage:
            records = ((percentage/100) * total_records)
            classes_samples.append(np.floor(records))
            
        self.max_classes_samples_by_label = classes_samples
        return classes_samples
    
    '''
        [[0.7, 0.3],
         [0.3, 0.7],
         [0.5, 0.5]]
    
    '''
    def get_max_proportional_samples_by_label(self, distribution_list):
        classes_multiplier = []
        for dist in zip(*distribution_list):
            print(dist)
            print(sum(dist))
            classes_multiplier.append(sum(dist))
        
        samples_sizes = []
        for i in range(0, len(self.max_classes_samples_by_label)):
            samples_sizes.append(self.max_classes_samples_by_label[i]/classes_multiplier[i])
        print(samples_sizes)
        self.max_proportional_samples_by_label = samples_sizes
        '''
            [A_samples, N_samples]
        
        '''
        return samples_sizes
        
    '''
            A   N
        [[0.7, 0.3],
         [0.3, 0.7],
         [0.5, 0.5]]
    
    '''
    def get_stratified_sample(self, distribution_list, target_attribute):

        labels = np.unique(list(self.dataset[target_attribute])) # [Anomaly, Normal]
        max_proportions_list = self.get_max_proportional_samples_by_label(distribution_list) #[A_max_qtd, N_max_qtd]
        for prop in max_proportions_list:
            print('a')
            print(prop)
        datasets_groups = []
        for proportions in distribution_list: # [[0.7, 0.3],[0.3, 0.7],[0.5, 0.5]]
            samples_by_labels = []
            for i in range(0, len(proportions)):
                samples = (max_proportions_list[i] * proportions[i])
                
                #print(samples)
                samples = int(np.floor(samples)) # qtd de amostras para as proporções máximas consideradas
                print(str(max_proportions_list[i]) + " * " + str(proportions[i]) + " = " + str(samples))
                samples_df = self.dataset[self.dataset[target_attribute] == labels[i]].sample(n = samples)
                self.dataset.drop(samples_df.index, axis=0,inplace = True)
                self.dataset.reset_index(inplace=True, drop=True)
                samples_by_labels.append(samples_df)
            group_dataset = pd.concat(samples_by_labels, ignore_index= True)
            datasets_groups.append(group_dataset)
        
        return datasets_groups
    '''
    def get_stratified_sample(self, sample_size, distribution_list, target_attribute):
        samples_by_labels = []
        for dist in distribution_list:
            samples = (sample_size * dist[1])
            samples = int(np.floor(samples))
            #print(samples)
            samples_df = self.dataset[self.dataset[target_attribute] == dist[0]].sample(n = samples)
            self.dataset.drop(samples_df.index, axis=0,inplace = True)
            self.dataset.reset_index(inplace=True, drop=True)
            samples_by_labels.append(samples_df)
        group_dataset = pd.concat(samples_by_labels, ignore_index= True)
        return group_dataset'''
    '''        
    def get_stratified_sample(self, target_attribute, sample_size):
        sample = \
        self.dataset.groupby(target_attribute, group_keys=False).\
            apply(lambda x: x.sample(int(np.rint(sample_size*len(x)/len(self.dataset))))).\
            sample(frac=1).reset_index(drop=True)
        return sample'''
            
    def get_dataset(self):
        return self.dataset
    
import pandas as pd

base = pd.read_csv('C:/Users/carlo/Desktop/Mestrado/Experimento FL/Etapa 1 - Escolha de modelo e data mining/IoTID20/IoTID20_preprocessada.csv')

#print(np.unique(base['Label']))
 
a = (base['Label'].value_counts()/base['Label'].count())*100
print(a)


utl = Utils(base, 'Label')
'''
x = utl.get_stratified_sample(47014, [['Anomaly', 0.3], ['Normal', 0.7]], 'Label') # 70 30
y = utl.get_stratified_sample(47014, [['Anomaly', 0.7], ['Normal', 0.3]], 'Label')
z = utl.get_stratified_sample(55970, [['Anomaly', 0.5], ['Normal', 0.5]], 'Label')
w = utl.get_stratified_sample(30261, [['Anomaly', 0.3], ['Normal', 0.2]], 'Label')
k = utl.get_stratified_sample(30261, [['Anomaly', 0.5], ['Normal', 0.2]], 'Label')
#x = utl.split_dataset_n_distributions([['Anomaly', 0.7], ['Normal', 0.3]], 5, 'Label')'''

#l = utl.maximum_samples_by_label('Label')
#for x in l:
#    print(x)
    
dist_list = [
     [0.3, 0.7],
     [0.7, 0.3],
     [0.2, 0.8]]

dataframes = utl.get_stratified_sample(dist_list, 'Label')
#print(dataframes)

for dataframe in dataframes:
    #a = (utl.get_dataset()['Label'].value_counts()/utl.get_dataset()['Label'].count())*100
    print((dataframe['Label'].value_counts()/dataframe['Label'].count())*100)

#a = (utl.get_dataset()['Label'].value_counts()/utl.get_dataset()['Label'].count())*100
#print(list(a))

#print(np.unique(base['Label'], return_counts=True))
#utl.print_dataset_size()