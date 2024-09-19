from PIL import Image
import os
import csv

import numpy as np
import pdb




with open('training_onehot_label.csv', 'r') as csv_file:


    csv_reader = list(csv.reader(csv_file))
    length = len(csv_reader)



    dataList = []
    labelList = []

    for i in range(length):
        
        row = csv_reader[i]
        name = row[0]
        label = row[1]


        img = Image.open('./train/%s.jpg'%name)
        npimg = np.array(img)

        dataList.append(npimg)
        labelList.append(int(label))

        print(name)


    dataList = np.array(dataList, dtype=np.uint8)
    labelList = np.array(labelList, dtype=np.uint8)


    np.save('./dataList.npy', dataList)
    np.save('./labelList.npy', labelList)


print('done')
