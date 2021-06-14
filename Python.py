# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:04:03 2021

@author: lucas
"""
import seaborn as sns
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# read data
customer_data=pd.read_excel("C:/Users/lucas/Desktop/BankChurners.xlsx")
#print(customer_data)
print(customer_data.head(4))
print(customer_data.tail(2))

# data understanding
#print(customer_data.dtypes)
print(customer_data.shape)

a = customer_data.describe()
print(round(a,2))


sns.pairplot(customer_data)

scatter_matrix(customer_data)
plt.show()
plt.hist(customer_data["Attrition_Flag"])
sns.set(rc={'figure.figsize':(15,10)})
customer_data.hist()
plt.show()

# data exploration
sns.violinplot(x="Attrition_Flag",y="Total_Trans_Ct", hue="Attrition_Flag", data=customer_data);
plt.show()
sns.violinplot(x="Attrition_Flag",y="Total_Ct_Chng_Q4_Q1", hue="Attrition_Flag", data=customer_data);
plt.show()

#data reduction
data = customer_data.drop(["CLIENTNUM"],axis=1)
print(data)
print(data.shape)
print(data.columns)

data.dropna(inplace=True)
print(data)
print(data.shape)
print(data.columns)

#data transformation
data['Average_Amount_per_Transaction'] = (data["Total_Trans_Amt"])/(data["Total_Trans_Ct"])
print(data)
print(data.shape)
print(data.columns)






# data type changing
print("Attrition_Flag' : ",data['Attrition_Flag'].unique())
print("Gender' : ",data['Gender'].unique())
print("Education_Level' : ",data['Education_Level'].unique())
print("Marital_Status' : ",data['Marital_Status'].unique())
print("Income_Category' : ",data['Income_Category'].unique())
print("Card_Category' : ",data['Card_Category'].unique())



from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
data['Attrition_Flag'] = le.fit_transform(data['Attrition_Flag'])
data['Gender'] = le.fit_transform(data['Gender'])
data['Education_Level'] = le.fit_transform(data['Education_Level'])
data['Marital_Status'] = le.fit_transform(data['Marital_Status'])
data['Income_Category'] = le.fit_transform(data['Income_Category'])
data['Card_Category'] = le.fit_transform(data['Card_Category'])
#display the initial records
print(data.head())
print(data.dtypes)

print("Attrition_Flag' : ",data['Attrition_Flag'].unique())
print("Gender' : ",data['Gender'].unique())
print("Education_Level' : ",data['Education_Level'].unique())
print("Marital_Status' : ",data['Marital_Status'].unique())
print("Income_Category' : ",data['Income_Category'].unique())
print("Card_Category' : ",data['Card_Category'].unique())

# data correlation
plt.figure(figsize=(20,20))
#cor=sns.heatmap(data.corr(),annot=True)
#plt.show()


dataFinal = data.drop(["Education_Level"],axis=1)
print(dataFinal.shape)


# log transformation
import numpy as np
dataFinal["Log Credit_Limit"] = np.log(dataFinal["Credit_Limit"])

plt.hist(dataFinal["Log Credit_Limit"])

dataFinal = dataFinal.drop(["Credit_Limit"],axis=1)
print(dataFinal.dtypes)


#data mining
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

target = dataFinal['Attrition_Flag']
data = dataFinal.drop(["Attrition_Flag","Marital_Status"],axis=1)
print(data.head(n=2))
print(data.shape)
print(data.dtypes)

from sklearn.model_selection import train_test_split
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
print(data_train.shape)
print(data_test.shape)
print(target_train.shape)
print(target_test.shape)

#LogisticRegression
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
pred = log.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())
#print the accuracy score of the model
print("logistic accuracy : ",accuracy_score(target_test, pred, normalize = True))

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
decision_tree = RandomForestClassifier(max_depth=2, random_state=0)
decision_tree = decision_tree.fit(data_train, target_train).predict(data_test)
print("randomtree accuracy : ",accuracy_score(target_test, pred, normalize = True))




#Naive-Bayes
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))

#LinearSVC
svc_model = LinearSVC(random_state=0)
#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))



#KNeighbors
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))



from yellowbrick.classifier import ClassificationReport 
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(gnb, classes=['1','0'])
visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
visualizer.score(data_test, target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(svc_model, classes=['1','0'])
visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
visualizer.score(data_test, target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data

from yellowbrick.classifier import ClassificationReport
# Instantiate the classification model and visualizer
visualizer = ClassificationReport(neigh, classes=['1','0'])
visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
visualizer.score(data_test, target_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data









