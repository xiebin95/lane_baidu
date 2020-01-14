#train
import os
# 使用第五张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
from utils.data_feeder import *
from lossse import *
from deeplabv3plus import deeplabv3_plus
from tensorflow.keras.callbacks import TensorBoard


data_dir = '../../private/lane_baidu/csv_data/train.csv'
train_list = pd.read_csv(data_dir)

#test_deeplabv3_plus
test_deeplabv3_plus  = deeplabv3_plus()


molde_test = test_deeplabv3_plus.net((384,1024,3),8)
print(molde_test.summary())
adam = tf.keras.optimizers.Adam()  # 优化函数，设定学习率（lr）等参数
molde_test.compile(loss=categorical_crossentropy_with_logits, optimizer=adam, metrics=['sparse_categorical_accuracy',def_mean_iou])
# molde_test.fit_generator(train_image_gen(train_list),steps_per_epoch = len(train_list)//4, epochs= 100)
batch_size = 4
steps_per_epoch = len(train_list)/batch_size
molde_test.fit_generator(train_image_gen(train_list,batch_size),steps_per_epoch = steps_per_epoch, epochs= 10,callbacks=[TensorBoard(log_dir='./tmp/log')] )
molde_test.save('test_deeplabv3_plus.h5')

