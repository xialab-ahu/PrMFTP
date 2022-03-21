# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: predictor.py
# @Software: PyCharm


import os

from model import MultiHeadAttention

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from pathlib import Path
from keras.models import load_model
import argparse


def ArgsGet():
    parse = argparse.ArgumentParser(description='MLTP')
    parse.add_argument('--file', type=str, default='test.fasta', help='fasta file')
    parse.add_argument('--out_path', type=str, default='MLTP/result', help='output path')
    args = parse.parse_args()
    return args


def get_data(file):
    # getting file and encoding
    seqs = []
    names = []
    with open(file) as f:
        for each in f:
            if each == '\n':
                continue
            elif each[0] == '>':
                names.append(each)
            else:
                seqs.append(each.rstrip())

    # encoding
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    max_len = 50
    data_e = []
    for i in range(len(seqs)):
        length = len(seqs[i])
        elemt, st = [], seqs[i]
        for j in st:
            index = amino_acids.index(j)
            elemt.append(index)
        if length < max_len:
            elemt += [0] * (max_len - length)
        data_e.append(elemt)

    return data_e, names


def predict(test, h5_model):
    dir = 'MLTP/model'
    print('predicting...')
    for ii in range(0, len(h5_model)):
        # 1.loading model
        h5_model_path = os.path.join(dir, h5_model[ii])
        load_my_model = load_model(h5_model_path,custom_objects={'MultiHeadAttention': MultiHeadAttention})

        # 2.predict
        score = load_my_model.predict(test)

        # 3.getting score
        if ii == 0:
            temp_score = score
        else:
            temp_score += score

    # getting prediction label
    score_label = temp_score / len(h5_model)
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < 0.5:
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    return score_label


def test_my(test, output_path, names):
    # models
    h5_model = []
    model_num = 10
    for i in range(1, model_num + 1):
        h5_model.append('model{}.h5'.format(str(i)))

    # prediction
    result = predict(test, h5_model)

    # label
    peptides = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
                'AVP',
                'BBP', 'BIP',
                'CPP', 'DPPIP',
                'QSP', 'SBP', 'THP']
    functions = []
    for e in result:
        temp = ''
        for i in range(len(e)):
            if e[i] == 1:
                temp = temp + peptides[i] + ','
            else:
                continue
        if temp == '':
            temp = 'none'
        if temp[-1] == ',':
            temp = temp.rstrip(',')
        functions.append(temp)

    output_file = os.path.join(output_path, 'result.txt')
    with open(output_file, 'w') as f:
        for i in range(len(names)):
            f.write(names[i])
            f.write('functions:' + functions[i] + '\n')


if __name__ == '__main__':
    args = ArgsGet()
    file = args.file  # fasta file
    output_path = args.out_path  # output path

    # building output path directory
    Path(output_path).mkdir(exist_ok=True)

    # reading file and encoding
    data, names = get_data(file)
    data = np.array(data)

    # prediction
    test_my(data, output_path, names)
