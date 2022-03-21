#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:28
# @Author  : ywh
# @File    : model.py
# @Software: PyCharm

from keras import initializers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, Embedding, Convolution1D, MaxPooling1D, Concatenate, Dropout
from keras.layers import Flatten, Dense, CuDNNLSTM
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.wrappers import Bidirectional


class MultiHeadAttention(Layer):
    def __init__(self, output_dim, num_head, kernel_initializer='glorot_uniform', **kwargs):
        self.output_dim = output_dim
        self.num_head = num_head
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        return {"output_dim": self.output_dim, "num_head": self.num_head}

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                 shape=(self.num_head, 3, input_shape[2], self.output_dim),
                                 initializer=self.kernel_initializer,
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo',
                                  shape=(self.num_head * self.output_dim, self.output_dim),
                                  initializer=self.kernel_initializer,
                                  trainable=True)
        self.built = True

    def call(self, x):
        q = K.dot(x, self.W[0, 0])
        k = K.dot(x, self.W[0, 1])
        v = K.dot(x, self.W[0, 2])
        e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
        e = e / (self.output_dim ** 0.5)
        e = K.softmax(e)
        outputs = K.batch_dot(e, v)
        for i in range(1, self.W.shape[0]):
            q = K.dot(x, self.W[i, 0])
            k = K.dot(x, self.W[i, 1])
            v = K.dot(x, self.W[i, 2])
            # print('q_shape:'+str(q.shape))
            e = K.batch_dot(q, K.permute_dimensions(k, [0, 2, 1]))  # 把k转置，并与q点乘
            e = e / (self.output_dim ** 0.5)
            e = K.softmax(e)
            # print('e_shape:'+str(e.shape))
            o = K.batch_dot(e, v)
            outputs = K.concatenate([outputs, o])
        z = K.dot(outputs, self.Wo)
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


def model_base(length, out_length, para):
    ed = para['embedding_dimension']
    ps = para['pool_size']
    fd = para['fully_dimension']
    dp = para['drop_out']
    lr = para['learning_rate']
    l2value = 0.001

    main_input = Input(shape=(length,), dtype='int64', name='main_input')
    x = Embedding(output_dim=ed, input_dim=21, input_length=length, name='Embadding')(main_input)

    a = Convolution1D(64, 2, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    apool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(a)

    b = Convolution1D(64, 3, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    bpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(b)

    c = Convolution1D(64, 8, activation='relu', border_mode='same', W_regularizer=l2(l2value))(x)
    cpool = MaxPooling1D(pool_length=ps, stride=1, border_mode='same')(c)

    merge = Concatenate(axis=-1)([apool, bpool, cpool])
    merge = Dropout(dp)(merge)

    x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(merge)

    x = MultiHeadAttention(80, 5)(x)
    x = Flatten()(x)
    x = Dense(fd, activation='relu', W_regularizer=l2(l2value))(x)

    output = Dense(out_length, activation='sigmoid', name='output', W_regularizer=l2(l2value))(x)

    model = Model(inputs=main_input, output=output)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model
