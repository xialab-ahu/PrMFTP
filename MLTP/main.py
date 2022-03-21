#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:27
# @Author  : ywh
# @File    : main.py
# @Software: PyCharm

import os
import time
from pathlib import Path
import numpy as np
from train import train_main

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
modelDir = 'model'
Path(modelDir).mkdir(exist_ok=True)
t = time.localtime(time.time())

peptide_type = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
                'AVP',
                'BBP', 'BIP',
                'CPP', 'DPPIP',
                'QSP', 'SBP', 'THP']


def staticTrainandTest(y_train, y_test):
    # static number
    data_size_tr = np.zeros(len(peptide_type))
    data_size_te = np.zeros(len(peptide_type))

    for i in range(len(y_train)):
        for j in range(len(y_train[i])):
            if y_train[i][j] > 0:
                data_size_tr[j] += 1

    for i in range(len(y_test)):
        for j in range(len(y_test[i])):
            if y_test[i][j] > 0:
                data_size_te[j] += 1

    print("TrainingSet:\n")
    for i in range(len(peptide_type)):
        print('{}:{}\n'.format(peptide_type[i], int(data_size_tr[i])))

    print("TestingSet:\n")
    for i in range(len(peptide_type)):
        print('{}:{}\n'.format(peptide_type[i], int(data_size_te[i])))

    return data_size_tr


def PadEncode(data, label, max_len):  # 序列编码
    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e = [], []
    sign = 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
            label_e.append(label[i])
    return np.array(data_e), np.array(label_e)


def getSequenceData(first_dir, file_name):
    # getting sequence data and label
    data, label = [], []
    path = "{}/{}.txt".format(first_dir, file_name)

    with open(path) as f:
        for each in f:
            each = each.strip()
            if each[0] == '>':
                label.append(np.array(list(each[1:]), dtype=int))  # Converting string labels to numeric vectors
            else:
                data.append(each)

    return data, label


def TrainAndTest(tr_data, tr_label, te_data, te_label, data_size):
    # Call training method
    train = [tr_data, tr_label]
    test = [te_data, te_label]
    threshold = 0.5
    model_num = 10  # model number
    test.append(threshold)

    train_main(train, test, model_num, modelDir, data_size)

    tt = time.localtime(time.time())
    with open(os.path.join(modelDir, 'time.txt'), 'a+') as f:
        f.write('finish time: {}m {}d {}h {}m {}s'.format(tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))

def main():
    # getting data
    first_dir = 'dataset'

    max_length = 50  # the longest length of the peptide sequence

    # getting train data and test data
    train_sequence_data, train_sequence_label = getSequenceData(first_dir, 'train')
    test_sequence_data, test_sequence_label = getSequenceData(first_dir, 'test')

    # Converting the list collection to an array
    y_train = np.array(train_sequence_label)
    y_test = np.array(test_sequence_label)

    # The peptide sequence is encoded and the sequences that do not conform to the peptide sequence are removed
    x_train, y_train = PadEncode(train_sequence_data, y_train, max_length)
    x_test, y_test = PadEncode(test_sequence_data, y_test, max_length)

    # Counting the number of each peptide in the training set and the test set, and return the total number of the training set
    data_size = staticTrainandTest(y_train, y_test)

    # training and predicting the data
    TrainAndTest(x_train, y_train, x_test, y_test, data_size)

if __name__ == '__main__':
    main()
