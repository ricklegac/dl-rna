from sklearn.datasets import load_iris
import pandas as pd 
import numpy as np  

#load dataset
iris_dataset = load_iris()

#print(iris_dataset)

#dataset label visualization 
print(iris_dataset.target_names)

#read the dataset with pandas 
df = pd.DataFrame(np.c_[iris_dataset['data'], iris_dataset['target']], columns=iris_dataset['feature_names']+['target'])
print(df)

#DATASET VISUALIZATION 

#two dimension graphic represent of dataset
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(10,7))

plt.scatter(df["pedal length(cm)"][df["target"]==0],df["peal widht(cm)"][df["target"]==0], c="b",label="setosa")