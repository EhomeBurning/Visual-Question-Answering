from keras.models import model_from_json
import os
from IPython.display import Image
import spacy
import scipy.io as sio
from sklearn.externals import joblib
import scipy
import numpy as np
from keras.utils import np_utils, generic_utils
# 在新的环境下：
max_len = 30
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
# 载入NLP的模型
nlp = spacy.load('en')
# 以及label的encoder
labelencoder = joblib.load('../data/labelencoder.pkl')

# 接着，把模型读进去
model = model_from_json(open('../data/mlp_num_hidden_units_1024_num_hidden_layers_3.json').read())
model.load_weights('../data/mlp_num_hidden_units_1024_num_hidden_layers_3_epoch_00.hdf5')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
nlp = spacy.load('en')
labelencoder = joblib.load('../data/labelencoder.pkl')

flag = True

# 所有需要外部导入的料
caffe = r'E:\JiangHeSong\tools\caffe-windows\caffe-windows'
vggmodel = '../data/VGG_ILSVRC_19_layers.caffemodel'
prototxt = '../data/VGG-Copy1.prototxt'
img_path = '../data/test_img.png'
image_features = '../data/test_img_vgg_feats.mat'

while flag:
    # 首先，给出你要提问的图片
    img_path = str(input('Enter path to image : '))
    # 对于这个图片，我们用caffe跑一遍VGG CNN，并得到4096维的图片特征
    os.system('python extract_features.py --caffe ' + caffe + ' --model_def ' + prototxt + ' --model ' + vggmodel + ' --image ' + img_path + ' --features_save_to ' + image_features)
    print('Loading VGGfeats')
    # 把这个图片特征读入
    features_struct = sio.loadmat(image_features)
    VGGfeatures = features_struct['feats']
    print ("Loaded")
    # 然后，你开始问他问题
    question = str(input("Ask a question: "))
    if question == "quit":
        flag = False
    timesteps = max_len
    X_q = get_questions_tensor_timeseries([question], nlp, timesteps)
    X_i = np.reshape(VGGfeatures, (1, 4096))
    # 构造成input形状
    X = [X_q, X_i]
    # 给出prediction
    y_predict = model.predict_classes(X, verbose=0)
    print(labelencoder.inverse_transform(y_predict))