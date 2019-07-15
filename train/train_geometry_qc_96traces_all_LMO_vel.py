# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:34:11 2018

@author: Wenbin Jiang
"""


from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from keras.callbacks import EarlyStopping


class LossHistory(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, savepath):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        fig = plt.gcf()
        plt.show()
        fig.savefig(savepath)
        
    def write_log(self,log_path,score):    
        a1 = self.accuracy['epoch']
        a2 = self.losses['epoch']
        a3 = self.val_acc['epoch']
        a4 = self.val_loss['epoch']
        with open(log_path,"w") as f:
            curves_data=[]
            f.write('\n测试结果\n')
            f.write("train—acc \t train-loss \t val-acc \t val-loss \n")
            for i in range(len(a1)):
                curves_data.append([round(a1[i],4),round(a2[i],4),round(a3[i],4),round(a4[i],4)])
                
            for i in range(len(a1)):
                temp = [str(j) for j in curves_data[i]]
                f.write('\t\t'.join(temp)+'\n')
            f.write('\nTest score:\t%.4f\n' % (score[0]))
            f.write('Test accuracy:\t%.2f%%' % (100*score[1]))


data = scipy.io.loadmat('../data/train_x.mat')
x_train = data['x_train']
data = scipy.io.loadmat('../data/train_y.mat')
y_train = data['y_train']
data = scipy.io.loadmat('../data/test_x.mat')
x_test = data['x_test']
data = scipy.io.loadmat('../data/test_y.mat')
y_test = data['y_test']

batch_size = 256
num_classes = 2
epochs = 500
img_rows, img_cols = 64, 96
y_test1=y_test

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(GlobalAveragePooling2D())
#model.add(GlobalMaxPooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

#model.compile(loss=keras.losses.binary_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy',
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = LossHistory()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history,early_stopping])

#model.fit(x_train[0:1000,:,:,0:1], y_train[0:1000,:],
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test[0:100,:,:,0:1], y_test[0:100,:]),
#          callbacks=[history])

#保存模型
model.save('net_all_LMOvel_96traces_new.h5')

# 按batch计算在某些输入数据上模型的误差
score = model.evaluate(x_test, y_test, verbose=0)

# 评估模型，输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])

#history.loss_plot('epoch')

save_img_path='Loss_plot.jpg'
log_path='Loss_log.txt'
history.loss_plot('epoch',save_img_path)
history.write_log(log_path,score)

pre_cls=model.predict_classes(x_test)

cm1 = confusion_matrix(y_test1,pre_cls)
print('Confusion Matrix : \n', cm1)

TruePositive= sum(np.diag(cm1))