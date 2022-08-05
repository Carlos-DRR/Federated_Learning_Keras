import numpy as np
import copy
class Utils:
    def __init__(self, dataset, target_attribute, groups):
        self.dataset = dataset
        self.target_attribute = target_attribute
        self.total_samples = len(self.dataset)/groups
        self.get_dataset_proportions(target_attribute)
        self.get_dataset_samples_by_proportions()
  
     
    def get_stratified_sample(self, target_attribute, n_groups):
        groups_list = []
        for i in range(0, n_groups):
            sample = \
            self.dataset.groupby(target_attribute, group_keys=False).\
                apply(lambda x: x.sample(int(np.rint(self.total_samples*len(x)/len(self.dataset))))).\
                sample(frac=1).reset_index(drop=True)
            groups_list.append(sample)
        return groups_list
    
    # [82, 17]
    def get_dataset_proportions(self, target_attribute):
        samples = list((self.dataset['Label'].value_counts()/self.dataset['Label'].count())*100)
        self.dataset_proportions = samples
        print(samples)

    # [124593, 26712]
    def get_dataset_samples_by_proportions(self):
        samples_by_proportions = []
        for sample_prop in self.dataset_proportions:
            sample_value = np.floor((self.total_samples * sample_prop)/100)
            samples_by_proportions.append(sample_value)
        self.samples_by_proportions = samples_by_proportions
        print(self.samples_by_proportions)
    # [[60,30,10], [90,10], [50,50]]
    
    def get_samples_proportions(self, proportions_list, target_attribute):  
        proportions_list_temp = copy.deepcopy(proportions_list)       
        list_datasets = []
        #samples_by_prop = copy.deepcopy(self.samples_by_proportions)
        #self.dataset_proportions são as props reais
        #proportions são as props pedidas
        #values_list = []
        for proportions in proportions_list_temp:  #[90,10]
            samples_by_prop_new = [0] * len(proportions)
            real_min_prop = min(self.dataset_proportions)
            #aqui
            real_min_prop_index = self.dataset_proportions.index(real_min_prop)
            #
            if real_min_prop < proportions[real_min_prop_index]:
                real_val = min(self.samples_by_proportions)
                samples_by_prop_new[real_min_prop_index] = np.floor(real_val)
                prop = proportions[real_min_prop_index]
                next_index = real_min_prop_index
                del proportions[real_min_prop_index]

            else:
                real_val = max(self.samples_by_proportions)
                real_max_prop_index = self.samples_by_proportions.index(real_val)
                samples_by_prop_new[real_max_prop_index] = np.floor(real_val)
                prop = proportions[real_max_prop_index]
                next_index = real_max_prop_index
                del proportions[real_max_prop_index]
            
            
            
            while len(proportions) > 0:
                proportion = proportions[0]
                next_index = next_index - 1

                samples = (real_val * proportion)/prop
                samples_by_prop_new [next_index] = np.floor(samples)
                
                proportions.remove(proportion)
                
            list_datasets.append(samples_by_prop_new)
        #print(list_datasets)
        return list_datasets
    
    '''
    def get_samples_proportions(self, proportions_list, target_attribute):  
        proportions_list_temp = copy.deepcopy(proportions_list)       
        list_datasets = []
        samples_by_prop = copy.deepcopy(self.samples_by_proportions)
        for proportions in proportions_list_temp:
            samples_by_prop_new = [0] * len(proportions)
            min_proportion_value = min(proportions)
            sample_prop = (sum(samples_by_prop)* min_proportion_value)/100

            if sample_prop > min(samples_by_prop):
                min_sample_value = min(samples_by_prop)#/len(proportions_list)
                
            else:
                min_sample_value = sample_prop
            
            min_value_index = proportions.index(min_proportion_value)
            print(min_value_index)
            samples_by_prop_new[min_value_index] = np.floor(min_sample_value)
            proportions.remove(min_proportion_value)
            
            next_index = min_value_index

            while len(proportions) > 0:
                proportion = proportions[0]
                next_index = next_index - 1

                samples = (min_sample_value * proportion)/min_proportion_value
                samples_by_prop_new [next_index] = np.floor(samples)
                
                proportions.remove(proportion)


            list_datasets.append(samples_by_prop_new)
        
        #print(list_datasets)
        return list_datasets
    '''
    def get_samples_by_proportions(self, proportios_list, groups_list):
        labels = np.unique(list(self.dataset[self.target_attribute ])) # [Anomaly, Normal]
        datasets_groups = []
        for i in range (0, len(proportios_list)):
            samples_by_labels = []
            for j in range(0, len(proportios_list[i])):
                print(labels[j] + " qtd a ser pegada: " + str(int(proportios_list[i][j])))
                print(len(groups_list[i]))
                samples_df = groups_list[i][groups_list[i][self.target_attribute] == labels[j]].sample(n = int(proportios_list[i][j]))
                groups_list[i].drop(samples_df.index, axis=0,inplace = True)
                groups_list[i].reset_index(inplace=True, drop=True)
                samples_by_labels.append(samples_df)

                print('Deu, tamanho total: ' + str(len(samples_df)))
                #print('Quantidade de registros normais restantes: ' + str(len(groups_list[i][groups_list[i]['Label'] == 'Normal'])))
            group_dataset = pd.concat(samples_by_labels, ignore_index= True)
            datasets_groups.append(group_dataset)
        return datasets_groups
    
    def get_dataset(self):
        return self.dataset
    
import pandas as pd

base = pd.read_csv('C:/Users/carlo/Desktop/Mestrado/Experimento FL/Etapa 1 - Escolha de modelo e data mining/IoTID20/IoTID20_preprocessada.csv')



utl = Utils(base, 'Label', 5)

prop_list = utl.get_samples_proportions([[70,30], [30,70], [50,50], [88,12], [10,90]], 'Label')
print(prop_list)

groups_list = utl.get_stratified_sample('Label', 5)

groups_datasets_list = utl.get_samples_by_proportions(prop_list, groups_list)




for group in groups_datasets_list:
    print((group['Label'].value_counts()/group['Label'].count())*100)
'''
x = utl.get_stratified_sample(47014, [['Anomaly', 0.3], ['Normal', 0.7]], 'Label') # 70 30
y = utl.get_stratified_sample(47014, [['Anomaly', 0.7], ['Normal', 0.3]], 'Label')
z = utl.get_stratified_sample(55970, [['Anomaly', 0.5], ['Normal', 0.5]], 'Label')
w = utl.get_stratified_sample(30261, [['Anomaly', 0.3], ['Normal', 0.2]], 'Label')
k = utl.get_stratified_sample(30261, [['Anomaly', 0.5], ['Normal', 0.2]], 'Label')
#x = utl.split_dataset_n_distributions([['Anomaly', 0.7], ['Normal', 0.3]], 5, 'Label')'''

