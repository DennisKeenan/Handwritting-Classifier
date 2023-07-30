from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as mp
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix

# Read Data
RAW=datasets.load_digits()
Data=RAW.data
Target=RAW.target
Image=RAW.images
# print(RAW.DESCR)
# print(RAW.data)
# print(Data.shape)
# print(Target.shape)
# print(Image.shape)
# print(Image)

# Detecting number through Pixel, and turning it to Image
# while(True):
#     number=int(input("Please input the image number: "))
#     print(Image[number])
#     print(Image[number].ravel())
#     print(Data[number])
#     print(Target[number])
#     mp.imshow(Image[number],cmap=mp.cm.gray_r)
#     mp.axis("Off")
#     mp.title(Target[number])
#     mp.show()

# Images Data
# figures,axis=mp.subplots(3,10,figsize=(15,6))
# for ax,im,number in zip(axis.ravel(),Image,Target):
#     ax.axis("Off")
#     ax.imshow(im,cmap=mp.cm.gray_r)
#     ax.set_title(number)
# mp.show()

# Training
x_train,x_test,y_train,y_test=train_test_split(Data,Target,test_size=0.33,random_state=99,stratify=Target)
# print(x_train.shape)
# print(x_test.shape)
KNN=KNeighborsClassifier(n_neighbors=3)
KNN.fit(x_train,y_train)
y_predict=KNN.predict(x_test)
# print(y_test)
# print(y_predict)

# Test Report
report=classification_report(y_test,y_predict)
print(report)

# Matrix
matrix=confusion_matrix(y_true=y_test,y_pred=y_predict)
# print(matrix)
aesthetic_matrix=sb.heatmap(matrix,annot=True,cmap="nipy_spectral_r")
aesthetic_matrix.set_title("Confusion Matrix")
# mp.show()

# Accuracy Score
accuracy=KNN.score(x_test,y_test)
print(accuracy)