import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import pickle
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre1 = sess.run(prediction, feed_dict={xs: v_xs, is_training: False})
    correct_prediction = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, is_training: False})
    return result

def compute_branch_accuracy(v_xs, v_ys):
    global branch_prediction
    y_pre1 = sess.run(branch_prediction, feed_dict={xs: v_xs, is_training: False})
    correct_prediction = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, is_training: False})
    return result


def compute_all_accuracy(v_xs, v_ys):
    global prediction
    sum_correct = 0
    data_size = len(v_xs)
    for i in range(data_size):  
       y_pre1 = sess.run(prediction, feed_dict={xs: v_xs[np.newaxis,i], is_training: False})
       if np.argmax(y_pre1,1) == np.argmax(v_ys[i]):
           sum_correct += 1
      
   
    return sum_correct / data_size

def compute_all_branch_accuracy(v_xs, v_ys):
    global branch_prediction
    sum_correct = 0
    data_size = len(v_xs)
    for i in range(data_size):  
       y_pre1 = sess.run(branch_prediction, feed_dict={xs: v_xs[np.newaxis,i], is_training: False})
       if np.argmax(y_pre1,1) == np.argmax(v_ys[i]):
           sum_correct += 1
      
   
    return sum_correct / data_size

tf.reset_default_graph()
cv_acc_list = []
cv_bacc_list = []
cv_val_list = []
cv_bval_list = []
cv_epoch = []

branch_loss_list = []
branch_acc = []
branch_val_loss = []
branch_val_acc = []
acc_list = []
loss_list = []
val_acc_list = []
val_loss_list = []

#saved weights
#TODO load weight
with open('0802.pickle', 'rb') as handle:
    par_list = pickle.load(handle)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,32,32,3])/255.  # 32*32*3
ys = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder_with_default(True, shape=())


#add this if needed
regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

#fixed
conv1_1 = tf.layers.conv2d(
      inputs=xs,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer =tf.constant_initializer(par_list[0]), #放存好的weight
      bias_initializer = tf.constant_initializer(par_list[1]),
      trainable = False
      )

conv1_2 = tf.layers.conv2d(
      inputs=conv1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer =tf.constant_initializer(par_list[2]), #放存好的weight
      bias_initializer = tf.constant_initializer(par_list[3]),
      trainable = False
      )
pool1 = tf.layers.max_pooling2d(conv1_2 , pool_size=[2, 2], strides=2)

conv2_1 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer =tf.constant_initializer(par_list[4]), #放存好的weight
      bias_initializer = tf.constant_initializer(par_list[5]),
      trainable = False
      )

conv2_2 = tf.layers.conv2d(
      inputs=conv2_1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer =tf.constant_initializer(par_list[6]), #放存好的weight
      bias_initializer = tf.constant_initializer(par_list[7]),
      trainable = False
      )

pool2 = tf.layers.max_pooling2d(conv2_2 , pool_size=[2, 2], strides=2)

#trainable
local_conv1_1 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      )

local_conv1_2 = tf.layers.conv2d(
      inputs=local_conv1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      )

local_pool1 = tf.layers.max_pooling2d(local_conv1_2, pool_size=[2, 2], strides=2)

local_conv2_1 = tf.layers.conv2d(
      inputs=local_pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      )

local_conv2_2 = tf.layers.conv2d(
      inputs=local_pool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      )

local_pool2 = tf.layers.max_pooling2d(local_conv2_2 , pool_size=[2, 2], strides=2)

flat = tf.layers.flatten(local_pool2)

dropout3 = tf.layers.dropout(
      inputs=flat, rate=0.4, training = is_training)

logits = tf.layers.dense(inputs=flat, units=10,kernel_regularizer=regularizer,name = 'logits')

prediction = tf.nn.softmax(logits)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys))

optimizer = tf.train.AdamOptimizer(0.001)

l2_loss = tf.losses.get_regularization_loss()

regularized_loss = loss + l2_loss
grads = optimizer.compute_gradients(regularized_loss)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)
train_op = optimizer.apply_gradients(grads)

#data

x_train_ = np.load('10000_train_x.npy')
y_train_ = np.load('10000_train_y.npy')

x_test = np.load('test_x.npy')
y_test = np.load('test_y.npy')

num_classes = 10
import keras

y_train_ = keras.utils.to_categorical(y_train_, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train_ = x_train_.astype('float32')
x_test = x_test.astype('float32')

# data preprocessing
x_train_[:,:,:,0] = (x_train_[:,:,:,0]-123.680)
x_train_[:,:,:,1] = (x_train_[:,:,:,1]-116.779)
x_train_[:,:,:,2] = (x_train_[:,:,:,2]-103.939)
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)
kf = KFold(n_splits=5)
kf.get_n_splits(x_train_)


for train_index, test_index in kf.split(x_train_):
    
    x_train = x_train_[train_index]
    y_train = y_train_[train_index]
    
    x_val = x_train_[test_index]
    y_val = y_train_[test_index]
    
    sess = tf.Session()
    # important step
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
        print('init1')
    else:
        init = tf.global_variables_initializer()
        print('init2')
    sess.run(init)
    print('init done')
    

    
    epochs       = 20
    iterations   = 391
    INDEX = 0
    BATCH_SIZE = 32
    TRAIN_SIZE = int(10000*4/5)
    EARLY_STOP_THRESHOLD = 1
    STOP_CHECK_EPOCH = 5
    pre_loss = 100 #a big number
    stop_epoch = 0
    
    #EPOCH = 1
    #BATCH_INDEX = 0
    #BATCH_SIZE = 50
    #TRAIN_SIZE = 32000
    
    
    
    
    for i in range(epochs):
        INDEX = 0
        while INDEX< TRAIN_SIZE:
            batch_xs = x_train[INDEX:INDEX+BATCH_SIZE]
            batch_ys = y_train[INDEX:INDEX+BATCH_SIZE]
            sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys, is_training: True})
            INDEX += BATCH_SIZE
        
        print('epoch ',i)
        #main
        print('*main')
        acc = compute_all_accuracy(
            x_train, y_train)
        print('train accuracy : ',acc)
    
    
        acc_list.append(acc)
        loss1 = sess.run(loss,feed_dict = {xs:x_train[:1000], ys: y_train[:1000],is_training: False})
        print('loss : ',loss1)
        loss_list.append(loss1)
        #main val
        val_acc = compute_all_accuracy(
                x_val,y_val
                )
        print('val accuracy',val_acc)
        loss2 = sess.run(loss,feed_dict = {xs:x_val[:1000], ys: y_val[:1000],is_training: False})
        print('val loss',loss2)
        val_acc_list.append(val_acc)
        val_loss_list.append(loss2)
        
        #early stop criteria
        #check every five epoch
        #stop when validation loss goes up
        #threshold = +1
#        stop_epoch += 1
#        if i>0:
#            if i %5 == 0:
#                if loss2 > pre_loss + EARLY_STOP_THRESHOLD:
#                    break;
#        else:
#            #initial loss
#            pre_loss = loss2
        
#        

    new_ticks = np.linspace(1, 20, 20)
    print(new_ticks)
    plt.xticks(new_ticks)

    
    plt.plot(acc_list)
    plt.plot(val_acc_list)
    plt.title('accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.show()
    plt.xticks(new_ticks)
    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'], loc='lower left')
    plt.show()
    
    break;
    

    
    cv_acc_list.append(acc_list)
    cv_val_list.append(val_acc_list)
    cv_bacc_list.append(branch_acc)
    cv_bval_list.append(branch_val_acc)
    cv_epoch.append(stop_epoch)
#main
print('**final epoch big evaluate**')
#main
print('*main')
acc = compute_all_accuracy(
    x_train_, y_train_)

print('train acc', acc)
val_acc = acc = compute_all_accuracy(
    x_test, y_test)
print('test accuracy',val_acc)
    #saver = tf.train.Saver()
    #text = '%d %.2f %s' % (1, 99.3, 'Justin')
    #name = '%s_%d_%s' % ('./joint_06',count,'/model.ckpt')
    #save_path = saver.save(sess, name)
