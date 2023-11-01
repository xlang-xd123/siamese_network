#!/usr/bin/python
#coding:utf-8

import os

path_imgs = 'CASIA-Iris-Lamp/'
i = 0
for label in os.listdir(path_imgs):
    if i > 300:
        print(label)
        label_path = path_imgs+ label
        for img in os.listdir(label_path + '/L/'):
            img_path = label_path + '/L/' + img + ' '
            with open("CASIA-Iris-Lamp/test.txt", "a") as f:
                f.write(str(img_path) +str(int(label)) + '\n')
        for img in os.listdir(label_path + '/R/'):
            img_path = label_path + '/R/' + img + ' '
            with open("CASIA-Iris-Lamp/test.txt", "a") as f:
                f.write(str(img_path) +str(int(label)) + '\n')
    else:
        print(label)
        label_path = path_imgs+ label
        for img in os.listdir(label_path + '/L/'):
            img_path = label_path + '/L/' + img + ' '
            with open("CASIA-Iris-Lamp/train.txt", "a") as f:
                f.write(str(img_path) +str(int(label)) + '\n')
        for img in os.listdir(label_path + '/R/'):
            img_path = label_path + '/R/' + img + ' '
            with open("CASIA-Iris-Lamp/train.txt", "a") as f:
                f.write(str(img_path) +str(int(label)) + '\n')
    i += 1
