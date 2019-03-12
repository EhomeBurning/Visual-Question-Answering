import sys
from random import shuffle
import argparse
import scipy
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Reshape
from keras.layers import merge,Input,Dense,Flatten
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils

from sklearn import preprocessing
from sklearn.externals import joblib
questions_train = open('../data/preprocessed/questions_train2014.txt', 'r').read().splitlines()
# with open('../data/preprocessed/questions_train2014.txt', 'r') as f:
#     i=0
#     while (i<100):# 由于原始爬取的json文件太大，采用取用一部分的数据
#         i += 1
#         print(u'正在载入第%s行......' % i)
#         lines = f.readline()  # 使用逐行读取的方法
#         print(lines)
answers_train = open('../data/preprocessed/answers_train2014_modal.txt', 'r').read().splitlines()
images_train = open('../data/preprocessed/images_train2014.txt', 'r').read().splitlines()

import operator
from collections import defaultdict
# 设定最多选取多少个回答
max_answers = 1000
answer_fq= defaultdict(int)
# 并为所有的问答，构造一个字典dict
for answer in answers_train:
    answer_fq[answer] += 1
# 按照出现次数，排序
sorted_fq = sorted(answer_fq.items(), key=operator.itemgetter(1), reverse=True)[0:max_answers]
top_answers, top_fq = zip(*sorted_fq)
new_answers_train=[]
new_questions_train=[]
new_images_train=[]
# 只提取top 1000问答相关
for answer,question,image in zip(answers_train, questions_train, images_train):
    if answer in top_answers:
        new_answers_train.append(answer)
        new_questions_train.append(question)
        new_images_train.append(image)

# 把新的数据赋值
questions_train = new_questions_train
answers_train = new_answers_train

images_train = new_images_train
labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
nb_classes = len(list(labelencoder.classes_))
joblib.dump( labelencoder,'../data/labelencoder3.pkl')

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]
    return image_matrix
def get_answers_matrix(answers, encoder):
    # string转化成数字化表达
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    # 并构造成标准的matrix
    return Y
vgg_model_path = '../Downloads/coco/vgg_feats.mat'
import spacy
nlp=spacy.load('en')
def get_questions_tensor_timeseries(questions, nlp, timesteps):
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            if j<timesteps:
                questions_tensor[i,j,:] = tokens[j].vector

    return questions_tensor
# 参数们
max_len = 30
word_vec_dim= 384
img_dim = 4096
dropout = 0.5
activation_mlp = 'tanh'
num_epochs = 1
model_save_interval = 5
num_hidden_units_mlp = 1024
num_hidden_units_lstm = 512
num_hidden_layers_mlp = 3
num_hidden_layers_lstm = 1
batch_size = 128

# 先造一个图片模型，也就是专门用来处理图片部分的
'''这个是使用keras 1.*版本的用法'''
# image_model = Sequential()
# image_model.add(Reshape((img_dim,), input_shape = (img_dim,)))
'''在keras 2.0以后的版本中，没有layers.Merge 只有layers.merge,所以只能替换如下'''

i0=Input(shape=100,name='image_mode1')
flatten=Flatten()(i0)
image_mode1=Dense(img_dim)(flatten)
image_mode1=Activation('relu')(image_mode1)
image_mode1=Dense(512)(image_mode1)
# 在来一个语言模型，专门用来处理语言的
# 因为，只有语言部分，需要LSTM。
'''在keras 2.0以后的版本中，没有layers.Merge 只有layers.merge,所以只能替换如下'''
# language_model = Sequential()
# if num_hidden_layers_lstm == 1:
#     language_model.add(LSTM(output_dim = num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim)))
# else:
#     language_model.add(LSTM(output_dim = num_hidden_units_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim)))
#     for i in range(num_hidden_layers_lstm-2):
#         language_model.add(LSTM(output_dim = num_hidden_units_lstm, return_sequences=True))
#     language_model.add(LSTM(output_dim = num_hidden_units_lstm, return_sequences=False))
'''在keras 2.0以后的版本中，没有layers.Merge 只有layers.merge,所以只能替换如下'''
if num_hidden_layers_lstm == 1:
    language_model=LSTM(output_dim = num_hidden_units_lstm, return_sequences=False, input_shape=(max_len, word_vec_dim))
else:
    language_model=LSTM(output_dim = num_hidden_units_lstm, return_sequences=True, input_shape=(max_len, word_vec_dim))
    for i in range(num_hidden_layers_lstm-2):
        language_model=LSTM(output_dim = num_hidden_units_lstm, return_sequences=True)
    language_model=LSTM(output_dim = num_hidden_units_lstm, return_sequences=False)

# 接下来，把楼上两个模型merge起来，
# 做最后的一步“分类”
# model = Sequential()
model=merge.concatenate()
for i in range(num_hidden_layers_mlp):
    model.add(Dense(num_hidden_units_mlp, init='uniform'))
    model.add(Activation(activation_mlp))
    model.add(Dropout(dropout))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# 同理，我们把模型结构存下来
json_string = model.to_json()
model_file_name = '../data/lstm_1_num_hidden_units_lstm_' + str(num_hidden_units_lstm) + \
                  '_num_hidden_units_mlp_' + str(num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
                  str(num_hidden_layers_mlp) + '_num_hidden_layers_lstm_' + str(num_hidden_layers_lstm)
open(model_file_name + '.json', 'w').write(json_string)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print ('Compilation done')

from six.moves import filter, filterfalse, map, range, zip, zip_longest
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
print ('loaded vgg features')
image_ids = open('../data/coco_vgg_IDMap.txt').read().splitlines()
img_map = {}
for ids in image_ids:
    id_split = ids.split()
    img_map[id_split[0]] = int(id_split[1])

nlp = spacy.load('en')
print ('loaded word2vec features...')
## training
print ('Training started...')
for k in range(num_epochs):

    progbar = generic_utils.Progbar(len(questions_train))

    for qu_batch,an_batch,im_batch in zip(grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
                                          grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
                                          grouper(images_train, batch_size, fillvalue=images_train[-1])):

        X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, max_len)
        X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
        Y_batch = get_answers_matrix(an_batch, labelencoder)
        loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
        progbar.add(batch_size, values=[("train loss", loss)])


    if k%model_save_interval == 0:
        model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))

model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k))