# 手写数字识别 -- CNN神经网络训练
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Convolution2D,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import Adam

# 1、载入数据
mnist = tf.keras.datasets.mnist
(train_data, train_target), (test_data, test_target) = mnist.load_data()

# 2、改变数据维度
train_data = train_data.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)
# 注：在TensorFlow中，在做卷积的时候需要把数据变成4维的格式
# 这4个维度分别是：数据数量，图片高度，图片宽度，图片通道数

# 3、归一化（有助于提升训练速度）
train_data = train_data/255.0
test_data = test_data/255.0

# 4、独热编码
train_target = tf.keras.utils.to_categorical(train_target, num_classes=10)
test_target = tf.keras.utils.to_categorical(test_target, num_classes=10)    #10种结果

# 5、搭建CNN卷积神经网络
model = Sequential()
# 5-1、第一层：卷积层+池化层
# 第一个卷积层
model.add(Convolution2D(input_shape = (28,28,1), filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
#         卷积层         输入数据                  滤波器数量      卷积核大小        步长          填充数据(same padding)  激活函数
# 第一个池化层 # pool_size
model.add(MaxPooling2D(pool_size = 2, strides = 2, padding = 'same',))
#         池化层(最大池化) 池化窗口大小   步长          填充方式

# 5-2、第二层：卷积层+池化层
# 第二个卷积层
model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))
# 64:滤波器个数      5:卷积窗口大小
# 第二个池化层
model.add(MaxPooling2D(2, 2, 'same'))

# 5-3、扁平化 （相当于把(64,7,7,64)数据->(64,7*7*64)）
model.add(Flatten())

# 5-4、第三层：第一个全连接层
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.5))

# 5-5、第四层：第二个全连接层（输出层）
model.add(Dense(10, activation='softmax'))
# 10：输出神经元个数

# 6、编译
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
#            优化器(adam)               损失函数(交叉熵损失函数)            标签

# 7、训练
model.fit(train_data, train_target, batch_size=64, epochs=50, validation_data=(test_data, test_target))

# 8、保存模型
model.save('mnist.h5')