from PIL import Image
import os
import csv

import pandas as pd
import numpy as np
import pdb



reader = pd.read_csv('./train_label.csv')
cls_list = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

cls_num = np.zeros(7)

with open('training_onehot_label.csv', 'w') as csvfile:
    
    writer = csv.writer(csvfile)

    for i in range(len(reader)):
        name = reader['image'][i]

        oh_label = np.zeros((7))
        for j in range(7):
            oh_label[j] = reader[cls_list[j]][i].astype(np.uint8)


        label = np.argmax(oh_label)

        img = Image.open('./train/%s.jpg'%name)
        npimg = np.array(img)
        print(name, npimg.shape, label)

        cls_num[label] += 1


        writer.writerow([name, label])


print(cls_num)
