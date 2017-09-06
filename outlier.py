import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('numerai_training_data.csv')
data = data.drop(['era','data_type','id'],axis=1)
tour = pd.read_csv('numerai_tournament_data.csv')
id = tour['id']
tour = tour.drop(['era','id'], axis = 1)
validation = tour[tour['data_type'] == 'validation'].drop('data_type', axis=1)
dataVal = data.append(validation)
num_val = tour['data_type'].value_counts()['validation']
test = tour.drop('data_type', axis=1)
num_train = len(data['target'])
total = data.append(test)

featTemp = total.drop('target', axis=1)
col = featTemp.columns
scaler = StandardScaler()
scaler.fit(featTemp)
featTemp = scaler.transform(featTemp)
features = pd.DataFrame(featTemp, columns = col)
print(features.head())
features['total'] = [0] * len(features['feature1'])
for i in range(21):
    temp = 'feature' + str(i+1)
    features['total'] += abs(features[temp])

outliers = features[(features.total > 30) & (features.index < num_train)]
feat = features.values
for i in range(21):
    temp = 'feature' + str(i+1)
    #plt.boxplot(data[temp])
    #plt.title(temp)
    #plt.show()
    dataVal[temp] = feat[:num_train+num_val,i]
    test[temp] = feat[num_train:,i]

dataVal = dataVal.drop(dataVal.index[outliers.index])
features.to_csv('features.csv')