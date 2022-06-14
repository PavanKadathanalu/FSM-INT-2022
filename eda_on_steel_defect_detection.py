# -*- coding: utf-8 -*-
"""EDA on Steel Defect Detection

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lntLBPJNCB2tvcDTy5sGCOmgb9QLptmX

# **In this Notebook Exploratory Data analysis has been Performed on train.csv**
"""

#Importing basic libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

"""Reading the train.csv file"""

train_df = pd.read_csv("train.csv")
train_df.head(10)

train_df.tail()

train_df.shape

"""Therefore there are 7095 rows and 3 Attributes in the given dataset"""

train_df.isnull().sum()

"""Since there is no NULL/NAN/NA values in the given data set hence no data cleaning is required"""

train_df.describe()

train_df.info()

"""So, we can conclude that ImageId and EncodedPixel are of string/object type whereas ClassId is of integer type

**Let's analyse number of labels for each defect type**
"""

defect1 = train_df[train_df['ClassId']==1].EncodedPixels.count()
defect2 = train_df[train_df['ClassId']==2].EncodedPixels.count()
defect3 = train_df[train_df['ClassId']==3].EncodedPixels.count()
defect4 = train_df[train_df['ClassId']==4].EncodedPixels.count()

print('There are {} defect1 images'.format(defect1))
print('There are {} defect2 images'.format(defect2))
print('There are {} defect3 images'.format(defect3))
print('There are {} defect4 images'.format(defect4))

#Plotting bar graph based on the count of each labels
labels = '1','2','3','4'
sizes = [defect1,defect2,defect3,defect4]

fig = plt.figure(figsize=(10,5))

#creating the bar plot
plt.bar(labels,sizes,color=['blue','orange','green','red'],width = 0.5)
plt.xlabel('Class')
plt.ylabel('count')
plt.title('Types of defects')
plt.show()

"""Here we can observe that defect type 3 is more dominant compared to any other defects and defect type 2 is least occuring defects. Hence there is a class imbalance.

**Now let us check whether the single image has more than one defect simultaneously**
"""

labels_per_image = train_df.groupby('ImageId')['EncodedPixels'].count()
fig,ax = plt.subplots(figsize=(6,6))
ax.hist(labels_per_image)
ax.set_title('Number of Labels per Image')

print('There are {} images with 1 label'.format(labels_per_image[labels_per_image==1].count()))
print('There are {} images with 2 label'.format(labels_per_image[labels_per_image==2].count()))
print('There are {} images with 3 label'.format(labels_per_image[labels_per_image==3].count()))
print('There are {} images with 4 label'.format(labels_per_image[labels_per_image==3].count()))

"""**Conclusion:**

1.   Most of images with defects contain the defects of only one type
2.   In rare cases an image contains the defects of two different types simulataneously.

**And all the images is of size 256X1600**
"""