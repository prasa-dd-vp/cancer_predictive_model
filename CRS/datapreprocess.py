#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 10:03:54 2019

@author: venkat-pt2636
"""
#importing packages
import pandas as pd
import numpy as np

#importing the dataset
df = pd.read_csv("seer_final_csv.csv")

#Renaming columns
df.rename(columns={'Marital status at diagnosis': 'marital status',
                   'Race/ethnicity':'Race',
                   'Reason no cancer-directed surgery':'reason',
                   'Tumor marker 1 (1990-2003)':'tumor marker 1',
                   'Tumor marker 2 (1990-2003)':'tumor marker 2',
                   'Breast - Adjusted AJCC 6th Stage (1988+)':'stage',
                   'Breast - Adjusted AJCC 6th T (1988+)':'T',
                   'Breast - Adjusted AJCC 6th N (1988+)':'N',
                   'Breast - Adjusted AJCC 6th M (1988+)':'M',}, inplace=True)

#deleting unwanted columns
df.drop(['First malignant primary indicator','Survival months flag'], axis = 1, inplace = True)

#Converting object values to integers

#marital status
df['marital status'] = df['marital status'].map({'Married (including common law)':1,
                                                 'Widowed':2,
                                                 'Single (never married)':3,
                                                 'Divorced':4,
                                                 'Separated':5,
                                                 'Unknown':6,
                                                 'Unmarried or Domestic Partner':7})

#Race
df['Race'] = df['Race'].map({'White':1,
                            'Black':2,
                            'Filipino':3,
                            'Chinese':4,
                            'Japanese':5,
                            'Hawaiian':6,
                            'American Indian/Alaska Native':7,
                            'Other Asian (1991+)':8,
                            'Korean (1988+)':9,
                            'Vietnamese (1988+)':10,
                            'Asian Indian (2010+)':11,
                            'Asian Indian or Pakistani, NOS (1988+)':12,
                            'Unknown':13})
df['Race'].replace(np.NaN,13,inplace = True)
df['Race'] = df['Race'].astype("int64")

#Laterality
df['Laterality'] = df['Laterality'].map({'Left - origin of primary':1,
                                         'Right - origin of primary':2,
                                         'Paired site, but no information concerning laterality':3,
                                         'Bilateral, single primary':4,
                                         'Only one side - side unspecified':5})


#reason
df['reason'] = df['reason'].map({'Surgery performed':1,
                            'Not recommended':2,
                            'Recommended but not performed, unknown reason':3,
                            'Unknown; death certificate; or autopsy only (2003+)':4,
                            'Recommended but not performed, patient refused':5,
                            'Not recommended, contraindicated due to other cond; autopsy only (1973-2002)':6,
                            'Recommended, unknown if performed':7,
                            'Not performed, patient died prior to recommended surgery':8})


#tumor marker 1
df['tumor marker 1'].replace("Blank(s)",9, inplace= True)
df['tumor marker 1'] = df['tumor marker 1'].astype("int64")

#tumor marker 2
df['tumor marker 2'].replace("Blank(s)",9, inplace= True)
df['tumor marker 2'] = df['tumor marker 2'].astype("int64")

#stage
df = df[df.stage != 'Blank(s)']
df.dropna(subset=['N','stage','T','M'], inplace=True)
df['stage'] = df['stage'].map({'IV':1,
                            'IIA':2,
                            'I':3,
                            'IIIA':4,
                            'IIB':5,
                            'IIIC':6,
                            'UNK Stage':7,
                            'IIIB':8,
                            'IIINOS':9,
                            '0':10})

#T
df['T'] =  df['T'].map({'T2':1,
                        'Any T, Mets':2,
                        'T1c':3,
                        'TX Adjusted':4,
                        'T3':5,
                        'T1b':6,
                        'T4b':7,
                        'T4d':8,
                        'T1a':9,
                        'T4a':10,
                        'T1mic':11,
                        'T4c':12,
                        'T0':13,
                        'Tis':14})  
#N
df['N'] = df['N'].map({'N0':1,
                       'N1':2,
                       'NX Adjusted':3,
                       'N3':4,
                       'N2':5})

#M
df['M'] = df['M'].map({'M0':1,
                       'M1':2,
                       'MX':3})

#data pre-processing
print(df.head())
print(df.info())

#Converting the dataframe to csv file
df.to_csv("seer_preprocessed.csv",index = False)




















# K-Means Clustering
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('seer_preprocessed.csv')
X = dataset.iloc[:, [1,2,3,6,7,8,9,10,11,12,13,14,15]].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 400, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 300, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

dataset = pd.concat([dataset, pd.DataFrame(y_kmeans)], axis=1)
dataset.rename(columns={0: 'Final_Stage'}, inplace=True)

dataset.to_csv("seer_preprocessed_final.csv",index = False)

















from django.http import JsonResponse, HttpResponse
from google.oauth2 import service_account
import dialogflow_v2 as df
credentials = service_account.Credentials.from_service_account_file('prasad-eaaf07e3eb38.json')
session_client = df.SessionsClient(credentials=credentials)
session = session_client.session_path("prasad-b784b","1234567")


while(True):    
    print("=" * 40)
    inputText = input("Enter the text:")
    text_input = df.types.TextInput(text = inputText,language_code = "en") 
    query_input = df.types.QueryInput(text = text_input)
    response = session_client.detect_intent(session=session, query_input=query_input)
    try:
        
        response_text = response.query_result
    except:
        response_text = "Am not sure about that!"
    print('Response: {} \n'.format(response_text)) 
    
    
    
    
    
    
    
    
    
    
# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('seer_preprocessed_final.csv')
X = dataset.iloc[:, [1,2,3,6,7,8,11]].values
y = dataset.iloc[:, -1].values





# Encoding categorical data
from keras.utils import to_categorical
y = to_categorical(y)
y = y.astype("int64")
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 7))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(np.array([[2,1,2,5,3,1,2,3,12]]))
np.argmax(y_pred)
y_pred = (y_pred > 0.5)

y_test = y_test.astype("bool")



# Step 12: Convert the Model to json
model_json = classifier.to_json()
with open("./model.json","w") as json_file:
  json_file.write(model_json)

# Step 13: Save the weights in a seperate file
classifier.save_weights("./model.h5")


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)