from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as mp

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
#     print(Target[number])
#     mp.imshow(Image[number],cmap=mp.cm.gray_r)
#     mp.axis("Off")
#     mp.title(Target[number])
#     mp.show()

figures,axis=mp.subplots(3,10,figsize=(15,6))
for ax,im,number in zip(axis.ravel(),Image,Target):
    ax.axis("Off")
    ax.imshow(im,cmap=mp.cm.gray_r)
    ax.set_title(number)
mp.show()