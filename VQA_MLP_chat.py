import sys
from random import shuffle
import argparse
import pickle
import numpy as np
import scipy.io
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
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

# 接下来，我们把所有的1000个答案给编号。
# 这样我们选取某个答案的时候，就不用使用“全称”，可以使用他们的label。
# 我们简单用到sklearn的preprocessing功能来赋值label
# 当然，我们之后一直需要使用这个一一对应的label，所以我们存下来。

labelencoder = preprocessing.LabelEncoder()
labelencoder.fit(answers_train)
nb_classes = len(list(labelencoder.classes_))
joblib.dump( labelencoder,'../data/labelencoder3.pkl')
#
# info = joblib.load(open('../data/labelencoder3.pkl','rb'))
# print(info)
# 我们写一个method，可以把所有的answers转化成数字化的label
def get_answers_matrix(answers, encoder):
    # string转化成数字化表达
    y = encoder.transform(answers)
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    # 并构造成标准的matrix
    return Y
vgg_model_path = '../Downloads/coco/vgg_feats.mat'
# 导入下载好的vgg_fearures (这是个matlab的文档，没关系，scipy可以读取)
# 读入VGG features
features_struct = scipy.io.loadmat(vgg_model_path)
VGGfeatures = features_struct['feats']
# 跟图片一一对应
image_ids = open('../data/coco_vgg_IDMap.txt').read().splitlines()
id_map = {}
for ids in image_ids:
    id_split = ids.split()
    id_map[id_split[0]] = int(id_split[1])
# 这时候，我们再写一个method，
#
# 是用来取得任何一个input图片的“数字化表达形式”的，也就是一个matrix
#
# 这个matrix是由vgg扫过/处理图片以后，再加一层flatten得到的4096维的数组，我们称之为CNN Features
def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in range(len(img_coco_ids)):
        image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]
    return image_matrix
# 接下来，是文字部分。
#
# 我们用SpaCy自带的英文模型。
#
# 把问句中的所有英文转化为vector，
#
# 并平均化整个句子。
#
# 这个本质上就是NLP的word2vec
# 载入Spacy的英语库 注意一定要导入pip install spacy 再输入：python -m spacy download en   下面将加载默认的模型- english-core-web
import spacy
nlp = spacy.load('en')
# 图片的维度大小
img_dim = 4096
# 句子/单词的维度大小
word_vec_dim = 384
# 这个method就是用来计算句子中所有word vector的总和，
# 目的在于把我们的文字用数字表示
def get_questions_matrix_sum(questions, nlp):
    # assert not isinstance(questions, basestring)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_matrix = np.zeros((nb_samples, word_vec_dim))
    for i in range(len(questions)):
        tokens = nlp(questions[i])
        for j in range(len(tokens)):
            questions_matrix[i,:] += tokens[j].vector
    return questions_matrix
# 我们来建立我们最简单版本的MLP模型
#
# 也就是普通的神经网络模型
#
# 注意：我这里就跑了1个epoch，只是为了Demo，真实场景中，这显然是不够的, 可以试试100次。
# 参数们
num_hidden_units = 1024
num_hidden_layers = 3
dropout = 0.5
activation = 'tanh'
# 注意：我这里就跑了1个epoch，
num_epochs = 1
model_save_interval = 10
batch_size = 128
# MLP之建造
# 输入层
model = Sequential()
model.add(Dense(num_hidden_units, input_dim=img_dim+word_vec_dim, kernel_initializer='uniform'))
model.add(Activation(activation))
model.add(Dropout(dropout))
# 中间层
for i in range(num_hidden_layers-1):
    model.add(Dense(num_hidden_units, kernel_initializer='uniform'))
    model.add(Activation(activation))
    model.add(Dropout(dropout))
# 输出层
model.add(Dense(nb_classes, kernel_initializer='uniform'))
model.add(Activation(tf.nn.softmax))
# 我们来吧构造的模型打印出来看看：
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='../data/model_mlp.png', show_shapes=True)
from IPython.display import Image
Image(filename='../data/model_mlp.png')
'''
好接下来，我们做一个保存的动作。

训练大规模网络的时候，这是一种比较保险的举措，即可以让我们回测数据，也可以在紧急情况下及时返回保存过了的模型
'''
json_string = model.to_json()
model_file_name = '../data/mlp_num_hidden_units_' + str(num_hidden_units) + '_num_hidden_layers_' + str(num_hidden_layers)
open(model_file_name  + '.json', 'w').write(json_string)
# compile模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
'''
在训练之前，我们还需要一个chunk list的方法，来把我们原始的数据分成一组一组的形式

为了我们之后的batch training做准备
'''
# 这是一个标准的chunk list方法
#  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
from more_itertools import recipes
from six.moves import filter, filterfalse, map, range, zip, zip_longest
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)
'''
开始训练！！！！！
'''
for k in range(num_epochs):
    # 给数据洗个牌先
    index_shuf = [i for i in range(len(questions_train))]
    shuffle(index_shuf)
    # 一一取出 问题，答案，和图片
    questions_train = [questions_train[i] for i in index_shuf]
    answers_train = [answers_train[i] for i in index_shuf]
    images_train = [images_train[i] for i in index_shuf]
    # 这就是个显示训练状态的bar，不写也没事，keras是默认有这个bar的
    # 但是如果你需要做一些customize的话，就可以在这个progbar上做文章
    progbar = generic_utils.Progbar(len(questions_train))
    # batch分组
    for qu_batch,an_batch,im_batch in zip(grouper(questions_train, batch_size, fillvalue=questions_train[-1]),
                                          grouper(answers_train, batch_size, fillvalue=answers_train[-1]),
                                          grouper(images_train, batch_size, fillvalue=images_train[-1])):
        X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
        X_i_batch = get_images_matrix(im_batch, id_map, VGGfeatures)
        X_batch = np.hstack((X_q_batch, X_i_batch))
        Y_batch = get_answers_matrix(an_batch,labelencoder)
        loss = model.train_on_batch(X_batch, Y_batch)
        progbar.add(batch_size, values=[("train loss", loss)])
    # 并且告诉模型，隔多久，存一次模型，比如这里，model_save_interval是10
    if k%model_save_interval == 0:
        model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))
# 顺手把最终的模型也存下来
model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))