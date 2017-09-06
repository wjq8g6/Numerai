
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from math import sqrt
sns.set_style('whitegrid')



# Defines error function
def rmse_cv(model, X_train, Y_train):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train, scoring="neg_mean_squared_error", cv=5))
    return (rmse)


# Function to train a linear regression model on the data
def train_model(df_train, df_test):
    X_train = df_train.drop('target', axis = 1)
    Y_train = df_train['target']
    X_pred = df_test.drop('target', axis = 1)

    # Uses lasso for regression model
    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.003, 0.0005]).fit(X_train, Y_train)
    prediction = model_lasso.predict(X_pred)
    loss = log_loss(Y_train, model_lasso.predict(X_train))
    print("Lasso loss: ", loss)
    return prediction


def random_forest(df_train, df_test):
    X_train = df_train.drop('target', axis=1)
    Y_train = df_train['target']
    X_pred = df_test.drop('target', axis=1)

    model = RandomForestClassifier(n_estimators=1000,max_features='auto', n_jobs=-1, min_samples_leaf=5000)
    model.fit(X_train,Y_train)
    predictions = model.predict_proba(X_pred)
    loss = log_loss(Y_train, model.predict_proba(X_train)[:,1])
    print("Random forest loss: ", loss)
    ret = predictions[:,1]
    return ret

def XGB(df_train, df_test):
    X_train = df_train.drop('target', axis=1)
    Y_train = df_train['target']
    X_pred = df_test.drop('target', axis=1)

    model = XGBClassifier()
    model.fit(X_train, Y_train)
    predictions = model.predict_proba(X_pred)
    ret = predictions[:, 1]
    return ret

def KNN(df_train, df_test):
    X_train = df_train.drop('target', axis=1)
    Y_train = df_train['target']
    X_pred = df_test.drop('target', axis=1)
    trainx, testx, trainy, testy = train_test_split(X_train, Y_train,test_size=0.3)

    model = KNeighborsClassifier(n_neighbors=5000)
    model.fit(X_train, Y_train)
    #loss = log_loss(testy, model.predict_proba(testx)[:, 1])
    predictions = model.predict_proba(X_pred)
    ret = predictions[:, 1]
    return ret

def ada(df_train, df_test):
    X_train = df_train.drop('target', axis=1)
    Y_train = df_train['target']
    X_pred = df_test.drop('target', axis=1)
    trainx, testx, trainy, testy = train_test_split(X_train, Y_train, test_size=0.3)

    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)
    model.fit(X_train, Y_train)
    #loss = log_loss(testy, model.predict_proba(testx)[:, 1])
    #print("Ada loss: ", loss)
    predictions = model.predict_proba(X_pred)
    loss = log_loss(Y_train, model.predict_proba(X_train)[:, 1])
    print("Ada loss: ", loss)
    ret = predictions[:, 1]
    return ret

def gradboost(df_train, df_test):
    X_train = df_train.drop('target', axis=1)
    Y_train = df_train['target']
    X_pred = df_test.drop('target', axis=1)
    trainx, testx, trainy, testy = train_test_split(X_train, Y_train, test_size=0.3)

    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    # loss = log_loss(testy, model.predict_proba(testx)[:, 1])
    # print("Ada loss: ", loss)
    loss = log_loss(Y_train, model.predict_proba(X_train)[:, 1])
    print("Grad boost loss: ", loss)
    predictions = model.predict_proba(X_pred)
    ret = predictions[:, 1]
    return ret

def process_data():

    data = pd.read_csv('numerai_training_data.csv')
    data = data.drop(['era', 'data_type', 'id'], axis=1)
    tour = pd.read_csv('numerai_tournament_data.csv')
    id = tour['id']
    tour = tour.drop(['era', 'id'], axis=1)
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
    features = pd.DataFrame(featTemp, columns=col)
    features['total'] = [0] * len(features['feature1'])
    for i in range(21):
        temp = 'feature' + str(i + 1)
        features['total'] += abs(features[temp])

    outliers = features[features.total > 72]
    feat = features.values
    for i in range(21):
        temp = 'feature' + str(i + 1)
        # plt.boxplot(data[temp])
        # plt.title(temp)
        # plt.show()
        dataVal[temp] = feat[:num_train + num_val, i]
        test[temp] = feat[num_train:, i]

    dataVal = dataVal.drop(dataVal.index[outliers.index])
    return dataVal,test, id

def pro2():
    data = pd.read_csv('numerai_training_data.csv')
    data = data.drop(['era', 'data_type', 'id'], axis=1)
    tour = pd.read_csv('numerai_tournament_data.csv')
    id = tour['id']
    tour = tour.drop(['era', 'id'], axis=1)
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
    features = pd.DataFrame(featTemp, columns=col)
    features['total'] = [0] * len(features['feature1'])
    for i in range(21):
        temp = 'feature' + str(i + 1)
        features['total'] += abs(features[temp])

    outliers = features[(features.total > 50) & (features.index < num_train)]
    print("Number of outliers: ", len(outliers.index))
    feat = features.values
    for i in range(21):
        temp = 'feature' + str(i + 1)
        # plt.boxplot(data[temp])
        # plt.title(temp)
        # plt.show()
        dataVal[temp] = feat[:num_train + num_val, i]
        test[temp] = feat[num_train:, i]

    dataVal = dataVal.drop(dataVal.index[outliers.index])
    return dataVal, test, id

dataVal,test, id = pro2()

#pred1 = train_model(data, test)
pred2 = random_forest(dataVal, test)
#pred3 = XGB(data, test)
#pred4 = KNN(data, test)
#pred5 = ada(dataVal, test)
pred6 = gradboost(dataVal, test)
predA = (pred2 + pred6) / 2

'''
for i in range(21):
    temp = 'feature' + str(i+1)
    plt.hist(data[temp], bins='auto')
    plt.title(temp)
    plt.show()
'''

results = pd.DataFrame(columns=['id', 'probability'])
results['id'] = id.values
results['probability'] = pred2
print(results.head())
results.to_csv('results.csv', index=False)

results2 = pd.DataFrame(columns=['id', 'probability'])
results2['id'] = id.values
results2['probability'] = predA
print(results2.head())
results2.to_csv('results2.csv', index=False)

results3 = pd.DataFrame(columns=['id', 'probability'])
results3['id'] = id.values
results3['probability'] = pred6
print(results3.head())
results3.to_csv('results3.csv', index=False)