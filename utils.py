import numpy as np
import copy
import pandas as pd

class Utils:
    def __init__(self, dataset, target_attribute, prop_list):
        self.n_groups = len(prop_list)
        self.prop_list = prop_list
        self.dataset = dataset
        self.target_attribute = target_attribute
        self.total_samples = len(self.dataset)/self.n_groups
        self.get_dataset_proportions(target_attribute)
        self.get_dataset_samples_by_proportions()
        self.get_samples_proportions() # self.desired_samples_by_prop
        #self.get_stratified_sample(target_attribute) # self.groups_list
        
    def get_stratified_sample(self):
        groups_list = []
        for i in range(0, self.n_groups):
            sample = \
            self.dataset.groupby(self.target_attribute, group_keys=False).\
                apply(lambda x: x.sample(int(np.rint(self.total_samples*len(x)/len(self.dataset))))).\
                sample(frac=1).reset_index(drop=True)
            groups_list.append(sample)
        self.groups_list = groups_list
        return groups_list
    
    # [82, 17]
    def get_dataset_proportions(self, target_attribute):
        samples = list((self.dataset['Label'].value_counts()/self.dataset['Label'].count())*100)
        self.dataset_proportions = samples


    # [124593, 26712]
    def get_dataset_samples_by_proportions(self):
        samples_by_proportions = []
        for sample_prop in self.dataset_proportions:
            sample_value = np.floor((self.total_samples * sample_prop)/100)
            samples_by_proportions.append(sample_value)
        self.samples_by_proportions = samples_by_proportions

    
    # [[60,30,10], [90,10], [50,50]]    
    def get_samples_proportions(self):  
        proportions_list_temp = copy.deepcopy(self.prop_list)       
        list_datasets = []

        for proportions in proportions_list_temp:  
            samples_by_prop_new = [0] * len(proportions)
            real_min_prop = min(self.dataset_proportions)
            
            real_min_prop_index = self.dataset_proportions.index(real_min_prop)
            
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
            self.desired_samples_by_prop = list_datasets
        return list_datasets

    def get_samples_by_proportions(self):
        self.get_stratified_sample()
        proportions_list = self.desired_samples_by_prop
        groups_list = self.groups_list
        labels = np.unique(list(self.dataset[self.target_attribute ])) # [Anomaly, Normal]
        datasets_groups = []
        for i in range (0, len(proportions_list)):
            samples_by_labels = []
            for j in range(0, len(proportions_list[i])):
                samples_df = groups_list[i][groups_list[i][self.target_attribute] == labels[j]].sample(n = int(proportions_list[i][j]))
                groups_list[i].drop(samples_df.index, axis=0,inplace = True)
                groups_list[i].reset_index(inplace=True, drop=True)
                samples_by_labels.append(samples_df)

            group_dataset = pd.concat(samples_by_labels, ignore_index= True)
            datasets_groups.append(group_dataset)
        return datasets_groups
    
    def get_dataset(self):
        return self.dataset
'''    

base = pd.read_csv('C:/Users/carlo/Desktop/Mestrado/Experimento FL/Etapa 1 - Escolha de modelo e data mining/IoTID20/IoTID20_preprocessada.csv')


#[[Anomalia, Normal], [Anomalia, Normal], [Anomalia, Normal]]
utl = Utils(base, 'Label', [[70,30], [30,70], [50,50], [88,12], [33,67]])



groups_datasets_list = utl.get_samples_by_proportions()




for group in groups_datasets_list:
    print((group['Label'].value_counts()/group['Label'].count())*100)

'''

