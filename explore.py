import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
sns.set_style('whitegrid')

data = pd.read_csv('numerai_training_data.csv')
data = data.drop(['era','data_type','id'],axis=1)
tour = pd.read_csv('numerai_tournament_data.csv')
id = tour['id']
test = tour.drop(['era','data_type','id'], axis=1)
num_train = len(data['target'])
total = data.append(test)

features = total.drop('target', axis=1)

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)

for i in range(21):
    temp = 'feature' + str(i+1)
    #plt.boxplot(data[temp])
    #plt.title(temp)
    #plt.show()
    data[temp] = features[:num_train,i]
    test[temp] = features[num_train:,i]