#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:27
# @Author  : ywh
# @File    : test.py
# @Software: PyCharm

import os
import time
from evaluation import evaluate
from keras.models import load_model
from model import MultiHeadAttention
import numpy as np
from sklearn.metrics import roc_curve

filenames = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP',
             'AVP',
             'BBP', 'BIP',
             'CPP', 'DPPIP',
             'QSP', 'SBP', 'THP']

def score_threshold(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    maxindex = (tpr - fpr).tolist().index(max(tpr - fpr))
    threshold = thresholds[maxindex]
    return threshold


def predict(X_test, y_test, thred, para, h5_model, first_dir):

    print("Prediction is in progress")

    for ii in range(0, len(h5_model)):

        h5_model_path = os.path.join(first_dir, h5_model[ii])
        load_my_model = load_model(h5_model_path, custom_objects={'MultiHeadAttention': MultiHeadAttention})

        # 2.predict
        score = load_my_model.predict(X_test)

        if ii == 0:
            score_pro = score
        else:
            score_pro += score
    score_pro = np.array(score_pro)
    score_pro = score_pro / len(h5_model)

    score_label = score_pro
    # getting prediction label
    for i in range(len(score_label)):
        for j in range(len(score_label[i])):
            if score_label[i][j] < thred: # throld
                score_label[i][j] = 0
            else:
                score_label[i][j] = 1

    # evaluation
    aiming, coverage, accuracy, absolute_true, absolute_false = evaluate(score_label, y_test)

    print("Prediction is done")
    print('aiming:', aiming)
    print('coverage:', coverage)
    print('accuracy:', accuracy)
    print('absolute_true:', absolute_true)
    print('absolute_false:', absolute_false)
    print('\n')

    # saving results
    data = []
    data.append('aiming:{}'.format(str(aiming)))
    data.append('coverage:{}'.format(str(coverage)))
    data.append('accuracy:{}'.format(str(accuracy)))
    data.append('absolute_true:{}'.format(str(absolute_true)))
    data.append('absolute_false:{}'.format(str(absolute_false)))
    data.append('\n')
    with open("result/result.txt", 'ab') as x:
        np.savetxt(x, np.asarray(data), fmt="%s\t")



def test_main(test, para, model_num, modelDir):
    h5_model = []
    for i in range(1, model_num + 1):
        h5_model.append('model{}.h5'.format('_' + str(i)))

    # step2:predict
    predict(test[0], test[1], test[2], para, h5_model,modelDir)

    ttt = time.localtime(time.time())
    with open("result/result.txt", 'ab') as f:
        v = []
        v.append(
            'finish time: {}m {}d {}h {}m {}s'.format(ttt.tm_mon, ttt.tm_mday, ttt.tm_hour, ttt.tm_min, ttt.tm_sec))
        np.savetxt(f, np.asarray(v), fmt="%s\t")
