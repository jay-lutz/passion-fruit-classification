import csv
import sys

import numpy as np
import tensorflow as tf

import cv2 as c
from sklearn.model_selection import train_test_split


class PassionFruitData:
    def __init__(self, test_size=0.7, model='alex'):
        read = csv.reader(open('Train.csv', newline=''))
        init = False
        classes = list()
        for row in read:
            if not init:
                init = True
            else:
                if row[1] == 'fruit_healthy':
                    classes.append((tf.one_hot(0, 3), row[0]))
                elif row[1] == 'fruit_woodiness':
                    classes.append((tf.one_hot(1, 3), row[0]))
                else:
                    classes.append((tf.one_hot(2, 3), row[0]))

        train, test = train_test_split(classes, test_size=test_size, shuffle=True)

        count = 0

        train_img = list()
        test_img = list()
        train_y = list()
        test_y = list()
        for cl in train:
            if model == 'alex':
                train_img.append(tf.image.resize(
                    tf.image.per_image_standardization(c.imread('./Train_Images/'+cl[1]+'.jpg')), (227, 227)))
            elif model == 'resnet' or model == 'mobile':
                train_img.append(tf.image.resize(c.imread('./Train_Images/'+cl[1]+'.jpg'), (224, 224)))
            train_y.append(cl[0])
            count += 1
            sys.stdout.write('\rImages Loaded: %i' % count)
            sys.stdout.flush()
        for cl in test:
            if model == 'alex':
                test_img.append(tf.image.resize(
                    tf.image.per_image_standardization(c.imread('./Train_Images/' + cl[1] + '.jpg')), (227, 227)))
            elif model == 'resnet' or model == 'mobile':
                test_img.append(tf.image.resize(c.imread('./Train_Images/' + cl[1] + '.jpg'), (224, 224)))
            test_y.append(cl[0])
            count += 1
            sys.stdout.write('\rImages Loaded: %i' % count)
            sys.stdout.flush()

        self.train = dict()
        self.test = dict()
        self.train['x'] = np.array(train_img)
        self.train['y'] = np.array(train_y)
        self.test['x'] = np.array(test_img)
        self.test['y'] = np.array(test_y)

    def train_data(self):
        return self.train

    def test_data(self):
        return self.test
