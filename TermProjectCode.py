import re
import sys
import numpy as np
import pandas as pd
from operator import add

from sklearn.datasets import make_blobs, make_swiss_roll, make_moons, make_circles
from matplotlib import pyplot
import matplotlib.cm as cm

from pyspark import SparkContext
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

def SupportVectorMachine(traindata,
                       max_iteration,
                       learningRate,
                       svm_lambda,
                       train_size):

    #Initialization
    prev_cost = 0
    L_cost = []
    parameter_size = len(traindata.take(1)[0][1])
    np.random.seed(805) #My area code
    parameter_vector = np.zeros(parameter_size) #Initialize with zeros
    
    for i in range(max_iteration):

        bc_weights = parameter_vector

        gradientCost = traindata.treeAggregate((np.zeros(parameter_size), 0, 0), lambda x, y: (x[0] + (svm_lambda * bc_weights)\
                          if y[0] * (np.dot(y[1], bc_weights)) >= 1 else x[0] + (svm_lambda * bc_weights) - np.dot(y[0], y[1]), \
                          (x[1] + (np.maximum(0, 1 - (y[0] * np.dot(y[1], bc_weights))))), (x[2] + 1)), add) #TreeAggregate

        cost = (svm_lambda * np.linalg.norm(bc_weights)) + ((1.0 / gradientCost[2]) * gradientCost[1]) #Cost
        
        #Calculate gradients
        gradient_derivative = gradientCost[0] #Gradient

        parameter_vector = parameter_vector - learningRate * gradient_derivative #Gradient Descent
        
        print(i+1, " Cost =", cost)
        
        prev_cost = cost
        L_cost.append(cost)
        
    print('')
    return parameter_vector, L_cost
    


#Formulate Test Predictions
n_feature = 2 #Number of features (Binary classification)
n_components = 2 #Number of clusters (For figure 1)
noise = 0.05 #Noise of curved data (For figures 2, 3, and 4)
factor = 0.75 #Factor of distance between circles (For figure 4)

max_iter = 50 #Number of iterations
n=2000 #Sample size

#Make Blobs
X1, y1 = make_blobs(n_samples=n, #Number of samples
                  centers=n_components,
                  n_features=n_feature,
                  random_state = 805)


##Code to plot pretty classification datasets from BigDataAnalytics GitHub Repository 'Spark-Example-20a-Sgdm-with-Tree-Aggregate.ipynb'
##Repository maintained by Dimitar Trajanov, originally uploaded by Kia Teymourian
##Originally authored by Yi Rong (yirong@bu.edu) and Xiaoyang Wang (gnayoiax@bu.edu)

#Scatterplot to Visualize Data
df1 = pd.DataFrame(dict(x=X1[:,0], y=X1[:,1], label=y1))
cluster_name = set(y1)
colors = dict(zip(cluster_name, cm.rainbow(np.linspace(0, 1, len(cluster_name)))))
fig, ax = pyplot.subplots()
grouped = df1.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key].reshape(1,-1))
pyplot.title('Figure 1: Blobs'.format(n_components))
pyplot.show()



#To RDD
rdd_X1, rdd_y1 = sc.parallelize(X1), sc.parallelize(y1) #Create RDDs

#Split Data to Train/Test
traindata1, testdata1 = rdd_y1.zip(rdd_X1).randomSplit([0.9, 0.1], seed=805)
traindata1 = traindata1.map(lambda x: (-1 if x[0] < 1 else 1, np.append(x[1],1))) #Add an additional feature for the bias term
traindata1.cache()

#Training Size
train_size = traindata1.count()

#Run SVM
pv, l_cost = SupportVectorMachine(traindata1, max_iter, 0.0001, 0.00005, train_size)

traindata1.unpersist()

#Formulate Test Predictions
testresults1 = testdata1.map(lambda x: (x[0], (np.dot(x[1],pv[:-1]) + pv[-1]))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Dot product of parameters and X_test (plus bias) determines prediction
testresults1.cache()

#Performance Metrics
TP = testresults1.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults1.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults1.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults1.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults1.unpersist()



#Make Spiral
X2, y2 = make_swiss_roll(n_samples=n, #Number of samples
                  noise=noise * 7.5, #Factor of 7.5 spreads data points a bit
                  random_state=805)

X2 = np.delete(X2, 1, 1) #Remove one dimension so we are kept to 2D space
y2 = np.where(y2 < 10, 0, 1) #Assign labels based on location in modified swiss roll

##Code to plot pretty classification datasets from BigDataAnalytics GitHub Repository 'Spark-Example-20a-Sgdm-with-Tree-Aggregate.ipynb'
##Repository maintained by Dimitar Trajanov, originally uploaded by Kia Teymourian
##Originally authored by Yi Rong (yirong@bu.edu) and Xiaoyang Wang (gnayoiax@bu.edu)

#Scatterplot to visualize data
df2 = pd.DataFrame(dict(x=X2[:,0], y=X2[:,1], label=y2))
cluster_name = set(y2)
colors = dict(zip(cluster_name, cm.rainbow(np.linspace(0, 1, len(cluster_name)))))
fig, ax = pyplot.subplots()
grouped = df2.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key].reshape(1,-1))
pyplot.title('Figure 2: Spiral'.format(n_components))
pyplot.show()



#To RDD
rdd_X2, rdd_y2 = sc.parallelize(X2), sc.parallelize(y2) #Create RDDs

#Split Data to Train/Test
traindata2, testdata2 = rdd_y2.zip(rdd_X2).randomSplit([0.9, 0.1], seed=805)
traindata2 = traindata2.map(lambda x: (-1 if x[0] < 1 else 1, np.append(x[1],1))) #Add an additional feature for the bias term
traindata2.cache()

#Training Size
train_size = traindata2.count()

#Run SVM
pv, l_cost = SupportVectorMachine(traindata2, max_iter, 0.0001, 0.00005, train_size)

traindata2.unpersist()

#Formulate Test Predictions
testresults2 = testdata2.map(lambda x: (x[0], (np.dot(x[1],pv[:-1]) + pv[-1]))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Dot product of parameters and X_test (plus bias) determines prediction
testresults2.cache()

#Performance Metrics
TP = testresults2.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults2.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults2.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults2.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults2.unpersist()



rdd_X2, rdd_y2 = sc.parallelize(X2), sc.parallelize(y2) #Create RDDs

#Split Data to Train/Test
traindata2, testdata2 = rdd_y2.zip(rdd_X2).randomSplit([0.9, 0.1], seed=805)
traindata2, testdata2 = traindata2.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][0]*x[1][1], x[1][0]**2, x[1][1]**2))),\
                          testdata2.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][0]*x[1][1], x[1][0]**2, x[1][1]**2)))
                          # x1, x2, x1x2, x1**2, x2**2
traindata2 = traindata2.map(lambda x: (-1 if x[0] < 1 else 1, np.append(x[1],1))) #Add an additional feature for the bias term
traindata2.cache()

#Training Size
train_size = traindata2.count()

#Run SVM
pv, l_cost = SupportVectorMachine(traindata2, max_iter, 0.000001, 0.00005, train_size)

traindata2.unpersist()

#Formulate Test Predictions
testresults2 = testdata2.map(lambda x: (x[0], (np.dot(x[1],pv[:-1]) + pv[-1]))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Dot product of parameters and X_test (plus bias) determines prediction
testresults2.cache()

#Performance Metrics
TP = testresults2.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults2.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults2.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults2.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults2.unpersist()



#Make Moons
X3, y3 = make_moons(n_samples=n, #Number of samples
                  noise=noise,
                  random_state=805)

##Code to plot pretty classification datasets from BigDataAnalytics GitHub Repository 'Spark-Example-20a-Sgdm-with-Tree-Aggregate.ipynb'
##Repository maintained by Dimitar Trajanov, originally uploaded by Kia Teymourian
##Originally authored by Yi Rong (yirong@bu.edu) and Xiaoyang Wang (gnayoiax@bu.edu)

#Scatterplot to Visualize Data
df3 = pd.DataFrame(dict(x=X3[:,0], y=X3[:,1], label=y3))
cluster_name = set(y3)
colors = dict(zip(cluster_name, cm.rainbow(np.linspace(0, 1, len(cluster_name)))))
fig, ax = pyplot.subplots()
grouped = df3.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key].reshape(1,-1))
pyplot.title('Figure 3: Moons'.format(n_components))
pyplot.show()


#To RDD
rdd_X3, rdd_y3 = sc.parallelize(X3), sc.parallelize(y3) #Create RDDs

#Split Data to Train/Test
traindata3, testdata3 = rdd_y3.zip(rdd_X3).randomSplit([0.9, 0.1], seed=805)
traindata3, testdata3 = traindata3.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][0] * x[1][1], x[1][1] * x[1][0]**2, x[1][0] * x[1][1]**2, x[1][0]**3, x[1][1]**3))),\
                          testdata3.map(lambda x: (x[0], (x[1][0], x[1][1],  x[1][0] * x[1][1], x[1][1] * x[1][0]**2, x[1][0] * x[1][1]**2, x[1][0]**3, x[1][1]**3)))
                          #x1, x2, x1x2, (x1**2)(x2), (x1)(x2**2), x1**3, x2**3
traindata3 = traindata3.map(lambda x: (-1 if x[0] < 1 else 1, np.append(x[1],1))) #Add an additional feature for the bias term
traindata3.cache()

#Training Size
train_size = traindata3.count()

#Run SVM
pv, l_cost = SupportVectorMachine(traindata3, max_iter, 0.01, 0.00005, train_size)

traindata3.unpersist()

#Formulate Test Predictions
testresults3 = testdata3.map(lambda x: (x[0], (np.dot(x[1],pv[:-1]) + pv[-1]))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Dot product of parameters and X_test (plus bias) determines prediction
testresults3.cache()

#Performance Metrics
TP = testresults3.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults3.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults3.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults3.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults3.unpersist()



#Make Circles
X4, y4 = make_circles(n_samples=n, #Number of samples
                  noise=noise,
                  factor=factor,
                  random_state=805)

##Code to plot pretty classification datasets from BigDataAnalytics GitHub Repository 'Spark-Example-20a-Sgdm-with-Tree-Aggregate.ipynb'
##Repository maintained by Dimitar Trajanov, originally uploaded by Kia Teymourian
##Originally authored by Yi Rong (yirong@bu.edu) and Xiaoyang Wang (gnayoiax@bu.edu)

#Scatterplot to Visualize Data
df4 = pd.DataFrame(dict(x=X4[:,0], y=X4[:,1], label=y4))
cluster_name = set(y4)
colors = dict(zip(cluster_name, cm.rainbow(np.linspace(0, 1, len(cluster_name)))))
fig, ax = pyplot.subplots()
grouped = df4.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key].reshape(1,-1))
pyplot.title('Figure 4: Circles'.format(n_components))
pyplot.show()



#To RDD
rdd_X4, rdd_y4 = sc.parallelize(X4), sc.parallelize(y4) #Create RDDs

#Split Data to Train/Test
traindata4, testdata4 = rdd_y4.zip(rdd_X4).randomSplit([0.9, 0.1], seed=805)
traindata4, testdata4 = traindata4.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][0]**2 + x[1][1]**2))),\
                          testdata4.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][0]**2 + x[1][1]**2)))
                          #x1, x2, x1**2 + x2**2
traindata4 = traindata4.map(lambda x: (-1 if x[0] < 1 else 1, np.append(x[1],1))) #Add an additional feature for the bias term
traindata4.cache()

#Training Size
train_size = traindata4.count()

#Run Simple LinearSVM
pv, l_cost = SupportVectorMachine(traindata4, max_iter, 0.0025, 0.00025, train_size)

traindata4.unpersist()

#Formulate Test Predictions
testresults4 = testdata4.map(lambda x: (x[0], (np.dot(x[1],pv[:-1]) + pv[-1]))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Dot product of parameters and X_test (plus bias) determines prediction
testresults4.cache()

#Performance Metrics
TP = testresults4.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults4.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults4.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults4.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults4.unpersist()



#Functions for Polynomial Kernel

def getPolyRDD(data_rdd, degree):
  data_rdd_features = data_rdd.map(lambda x: (x[1][0], x[1][1])) #Grab only data point coordinates
  data_rdd_labels = data_rdd.map(lambda x: x[0]) #Grab labels

  features = np.array(data_rdd_features.collect())
  feature_matrix = np.array((1 + np.dot(features,features.T))**degree) #Polynomial kernel

  new_features = sc.parallelize(feature_matrix)
  data_rdd_new = data_rdd_labels.zip(new_features) #Put back together RDD
  return(data_rdd_new)

def getPoly(X, degree):
  features = X

  feature_matrix = (1 + np.dot(X,X.T))**degree #Polynomial kernel

  return(np.array(feature_matrix))
  
# matrix_X = getPoly(X, 2)



from sklearn.metrics.pairwise import pairwise_kernels

def getGaussian(X, sigma):
  features = X
  feature_len = len(features)
  feature_matrix = np.zeros((feature_len, feature_len)) #Make a 'm x m' similarity matrix to store our values for every [i,j] pair
  for i in range(feature_len):
    for j in range(feature_len): #Iterate through
      feature_matrix[i][j] = float(np.exp(-((np.linalg.norm(features[i] - features[j]))**2)/(2 * (sigma)**2))) #Gaussian kernel
  return(np.array(feature_matrix))

def rbf_kernel(x,y):
  return float(np.exp(-((np.linalg.norm(x - y))**2)/(2 * (0.1**2)))) #Gaussian kernel

#rbf_kernel = pairwise_kernels(X, metric = 'rbf_kernel')



#To RDD
rdd_X2, rdd_y2 = sc.parallelize(getGaussian(X2, 0.1)), sc.parallelize(y2) #Create RDDs

#Split Data to Train/Test
traindata2, testdata2 = rdd_y2.zip(rdd_X2).randomSplit([0.9, 0.1], seed=805)
traindata2 =  traindata2.map(lambda x: (-1 if x[0] < 1 else 1, x[1]))
traindata2.cache()

#Training Size
train_size = traindata2.count()

#Run Gaussian SVM
pv, l_cost = SupportVectorMachine(traindata2, max_iter, 0.02, 0.00001, train_size)

traindata2.unpersist()

#Formulate Test Predictions
testresults2 = testdata2.map(lambda x: (x[0], (sum(x[1] * pv)))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Sum of parameters * X_test determines prediction
testresults2.cache()

#Performance Metrics
TP = testresults2.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults2.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults2.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults2.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults2.unpersist()



#To RDD
rdd_X3, rdd_y3 = sc.parallelize(getGaussian(X3, 0.1)), sc.parallelize(y3) #Create RDDs

#Split Data to Train/Test
traindata3, testdata3 = rdd_y3.zip(rdd_X3).randomSplit([0.9, 0.1], seed=805)
traindata3 =  traindata3.map(lambda x: (-1 if x[0] < 1 else 1, x[1]))
traindata3.cache()

#Training Size
train_size = traindata3.count()

#Run Gaussian SVM
pv, l_cost = SupportVectorMachine(traindata3, max_iter, 0.00025, 0.000001, train_size)

traindata3.unpersist()

#Formulate Test Predictions
testresults3 = testdata3.map(lambda x: (x[0], (sum(x[1] * pv)))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Sum of parameters * X_test determines prediction
testresults3.cache()

#Performance Metrics
TP = testresults3.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults3.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults3.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults3.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults3.unpersist()



#To RDD
rdd_X4, rdd_y4 = sc.parallelize(getGaussian(X4, 0.1)), sc.parallelize(y4) #Create RDDs

#Split Data to Train/Test
traindata4, testdata4 = rdd_y3.zip(rdd_X4).randomSplit([0.9, 0.1], seed=805)
traindata4 = traindata4.map(lambda x: (-1 if x[0] < 1 else 1, x[1]))
traindata4.cache()

#Training Size
train_size = traindata4.count()

#Run Gaussian SVM
pv, l_cost = SupportVectorMachine(traindata4, max_iter, 0.00015, 0.000001, train_size)

traindata4.unpersist()

#Formulate Test Predictions
testresults4 = testdata3.map(lambda x: (x[0], (sum(x[1] * pv)))).map(lambda x: (x[0], (x[1] > 0).astype(int))) #Sum of parameters * X_test determines prediction
testresults4.cache()

#Performance Metrics
TP = testresults4.filter(lambda x: x[0] == 1 and x[1] == 1).count() #TP
FP = testresults4.filter(lambda x: x[0] == 0 and x[1] == 1).count() #FP
FN = testresults4.filter(lambda x: x[0] == 1 and x[1] == 0).count() #FN
TN = testresults4.filter(lambda x: x[0] == 0 and x[1] == 0).count() #TN

F1 = TP/(TP + (0.50 * (FP + FN))) #F1 Score

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)

testresults4.unpersist()



from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score



#Check sklearn - Only Linear functionality available for MLlib

#Make Spiral
X2, y2 = make_swiss_roll(n_samples=n, #Number of samples
                  noise=noise * 7.5,
                  random_state=805)

X2 = np.delete(X2, 1, 1) #Remove one dimension so we are kept to 2D space
y2 = np.where(y2 < 10, 0, 1) #Assign labels based on location in modified swiss roll

rdd_X2, rdd_y2 = sc.parallelize(X2), sc.parallelize(y2) #Create RDDs

#Split Data to Train/Test
traindata2, testdata2 = rdd_y2.zip(rdd_X2).randomSplit([0.9, 0.1], seed=805)

#Collect to Obtain Same Train/Test Split
X_train = traindata2.map(lambda x: x[1]).collect()
y_train = traindata2.map(lambda x: x[0]).collect()
X_test = testdata2.map(lambda x: x[1]).collect()
y_test = testdata2.map(lambda x: x[0]).collect()

model2 = make_pipeline(StandardScaler(), SVC(max_iter= max_iter, kernel= 'rbf', gamma= 0.1)) #Gaussian kernel
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

confusion_matrix2 = confusion_matrix(y_test, y_pred)
TP = confusion_matrix2[0][0]
FP = confusion_matrix2[0][1]
FN = confusion_matrix2[1][0]
TN = confusion_matrix2[1][1]
F1 = f1_score(y_test, y_pred) 

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)



#Try Again with 100 Maximum Iterations

#Make Spiral
X2, y2 = make_swiss_roll(n_samples=n, #Number of samples
                  noise=noise * 7.5,
                  random_state=805)

X2 = np.delete(X2, 1, 1)
y2 = np.where(y2 < 10, 0, 1)

rdd_X2, rdd_y2 = sc.parallelize(X2), sc.parallelize(y2) #Create RDDs

#Split Data to Train/Test
traindata2, testdata2 = rdd_y2.zip(rdd_X2).randomSplit([0.9, 0.1], seed=805)

#Collect to Obtain Same Train/Test Split
X_train = traindata2.map(lambda x: x[1]).collect()
y_train = traindata2.map(lambda x: x[0]).collect()
X_test = testdata2.map(lambda x: x[1]).collect()
y_test = testdata2.map(lambda x: x[0]).collect()

model2 = make_pipeline(StandardScaler(), SVC(max_iter= max_iter * 2, kernel= 'rbf', gamma= 0.1)) #Gaussian kernel
model2.fit(X_train, y_train)

y_pred = model2.predict(X_test)

confusion_matrix2 = confusion_matrix(y_test, y_pred)
TP = confusion_matrix2[0][0]
FP = confusion_matrix2[0][1]
FN = confusion_matrix2[1][0]
TN = confusion_matrix2[1][1]
F1 = f1_score(y_test, y_pred)

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)



#Make Moons
X3, y3 = make_moons(n_samples=n, #Number of samples
                  noise=noise,
                  random_state=805)

rdd_x3, rdd_y3 = sc.parallelize(X3), sc.parallelize(y3) #Create RDDs

#Split Data to Train/Test
traindata3, testdata3 = rdd_y3.zip(rdd_X3).randomSplit([0.9, 0.1], seed=805)

#Collect to Obtain Same Train/Test Split
X_train = traindata3.map(lambda x: x[1]).collect()
y_train = traindata3.map(lambda x: x[0]).collect()
X_test = testdata3.map(lambda x: x[1]).collect()
y_test = testdata3.map(lambda x: x[0]).collect()

model3 = make_pipeline(StandardScaler(), SVC(max_iter= max_iter, kernel= 'rbf', gamma= 0.1)) #Gaussian kernel
model3.fit(X_train, y_train)

y_pred = model3.predict(X_test)

confusion_matrix3 = confusion_matrix(y_test, y_pred)
TP = confusion_matrix3[0][0]
FP = confusion_matrix3[0][1]
FN = confusion_matrix3[1][0]
TN = confusion_matrix3[1][1]
F1 = f1_score(y_test, y_pred)

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)



#Make Circles
X4, y4 = make_circles(n_samples=n, #Number of samples
                  noise=noise,
                  factor=factor,
                  random_state=805)

rdd_x4, rdd_y4 = sc.parallelize(X4), sc.parallelize(y4) #Create RDDs

#Split Data to Train/Test
traindata4, testdata4 = rdd_y4.zip(rdd_X4).randomSplit([0.9, 0.1], seed=805)

#Collect to Obtain Same Train/Test Split
X_train = traindata4.map(lambda x: x[1]).collect()
y_train = traindata4.map(lambda x: x[0]).collect()
X_test = testdata4.map(lambda x: x[1]).collect()
y_test = testdata4.map(lambda x: x[0]).collect()

model4 = make_pipeline(StandardScaler(), SVC(max_iter= max_iter, kernel= 'rbf', gamma= 0.1)) #Gaussian kernel
model4.fit(X_train, y_train)

y_pred = model4.predict(X_test)

confusion_matrix4 = confusion_matrix(y_test, y_pred)
TP = confusion_matrix4[0][0]
FP = confusion_matrix4[0][1]
FN = confusion_matrix4[1][0]
TN = confusion_matrix4[1][1]
F1 = f1_score(y_test, y_pred)

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)



#Try Again with 100 Maximum Iterations

#Make Circles
X4, y4 = make_circles(n_samples=n, #Number of samples
                  noise=noise,
                  factor=factor,
                  random_state=805)

rdd_x4, rdd_y4 = sc.parallelize(X4), sc.parallelize(y4) #Create RDDs

#Split Data to Train/Test
traindata4, testdata4 = rdd_y4.zip(rdd_X4).randomSplit([0.9, 0.1], seed=805)

#Collect to Obtain Same Train/Test Split
X_train = traindata4.map(lambda x: x[1]).collect()
y_train = traindata4.map(lambda x: x[0]).collect()
X_test = testdata4.map(lambda x: x[1]).collect()
y_test = testdata4.map(lambda x: x[0]).collect()

model4 = make_pipeline(StandardScaler(), SVC(max_iter= max_iter * 2, kernel= 'rbf', gamma= 0.1)) #Gaussian kernel
model4.fit(X_train, y_train)

y_pred = model4.predict(X_test)

confusion_matrix4 = confusion_matrix(y_test, y_pred)
TP = confusion_matrix4[0][0]
FP = confusion_matrix4[0][1]
FN = confusion_matrix4[1][0]
TN = confusion_matrix4[1][1]
F1 = f1_score(y_test, y_pred)

print("TP: ", TP, "FP: ", FP, "FN: ", FN, "TN: ", TN)
print("F1 Score: ", F1)