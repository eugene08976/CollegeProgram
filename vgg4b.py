from __future__ import print_function
import tensorflow as tf
import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre1 = sess.run(prediction, feed_dict={xs: v_xs, is_training : False})
    correct_prediction = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, is_training : False})
    return result

def compute_branch_accuracy(v_xs, v_ys):
    global branch_prediction
    y_pre1 = sess.run(branch_prediction, feed_dict={xs: v_xs,is_training: False})
    correct_prediction = tf.equal(tf.argmax(y_pre1,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, is_training:False})
    return result


def compute_all_accuracy(v_xs, v_ys):
    global prediction
    sum_correct = 0
    data_size = len(v_xs)
    for i in range(data_size):  
       y_pre1 = sess.run(prediction, feed_dict={xs: v_xs[np.newaxis,i], is_training :False})
       if np.argmax(y_pre1,1) == np.argmax(v_ys[i]):
           sum_correct += 1
      
   
    return sum_correct / data_size

def compute_all_branch_accuracy(v_xs, v_ys):
    global branch_prediction
    sum_correct = 0
    data_size = len(v_xs)
    for i in range(data_size):  
       y_pre1 = sess.run(branch_prediction, feed_dict={xs: v_xs[np.newaxis,i], is_training :False})
       if np.argmax(y_pre1,1) == np.argmax(v_ys[i]):
           sum_correct += 1
      
   
    return sum_correct / data_size

tf.reset_default_graph()

cv_acc = []
cv_bacc = []
cv_val = []
cv_bval = []
cv_epoch = []

branch_loss_list = []
branch_acc = []
branch_val_loss = []
branch_val_acc = []
acc_list = []
loss_list = []
val_acc_list = []
val_loss_list = []

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,32,32,3])/255.  # 32*32*3
ys = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder_with_default(True, shape=())

#set scale = 0 to disable this regularizer
regularizer = tf.contrib.layers.l2_regularizer(scale=0.)

#network start
conv1_1 = tf.layers.conv2d(
      inputs=xs,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv1_1'
      )
#layer_norm1_1 = tf.contrib.layers.layer_norm(conv1_1)

conv1_2 = tf.layers.conv2d(
      inputs=conv1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv1_2'
      )
#layer_norm1_2 = tf.contrib.layers.layer_norm(conv1_2)

pool1 = tf.layers.max_pooling2d(conv1_2 , pool_size=[2, 2], strides=2)



conv2_1 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv2_1')
#layer_norm2_1 = tf.contrib.layers.layer_norm(conv2_1)

conv2_2 = tf.layers.conv2d(
      inputs=conv2_1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv2_2')
#layer_norm2_2 = tf.contrib.layers.layer_norm(conv2_2)

pool2 = tf.layers.max_pooling2d(conv2_2 , pool_size=[2, 2], strides=2)

conv3_1 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv3_1')

conv3_2 = tf.layers.conv2d(
      inputs=conv3_1,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv3_2')

conv3_3 = tf.layers.conv2d(
      inputs=conv3_2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv3_3')

pool3 = tf.layers.max_pooling2d(conv3_3 , pool_size=[2, 2], strides=2)

conv4_1 = tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv4_1')

conv4_2 = tf.layers.conv2d(
      inputs=conv4_1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv4_2')

conv4_3 = tf.layers.conv2d(
      inputs=conv4_2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'conv4_3')

pool4 = tf.layers.max_pooling2d(conv4_3 , pool_size=[2, 2], strides=2)

#branch
bconv1_1 = tf.layers.conv2d(
      inputs=pool2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'bconv1_1'
      )
#layer_norm1_1 = tf.contrib.layers.layer_norm(conv1_1)

bconv1_2 = tf.layers.conv2d(
      inputs=bconv1_1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'bconv1_2'
      )
bpool1 = tf.layers.max_pooling2d(bconv1_2 , pool_size=[2, 2], strides=2)
#8*8*64
bconv2_1 = tf.layers.conv2d(
      inputs=bpool1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'bconv2_1'
      )
#layer_norm1_1 = tf.contrib.layers.layer_norm(conv1_1)

bconv2_2 = tf.layers.conv2d(
      inputs=bconv2_1,
      filters=32,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=regularizer,
      name = 'bconv2_2'
      )

#4*4*32
bpool2 = tf.layers.max_pooling2d(bconv2_2 , pool_size=[2, 2], strides=2)


branch_flat = tf.layers.flatten(bpool2)



branch_logits = tf.layers.dense(inputs=branch_flat, units=10,kernel_regularizer=regularizer,name = 'branch_logits')

branch_prediction = tf.nn.softmax(branch_logits)
  
#pool3_flat = tf.reshape(pool3, [-1, 102400])
flat = tf.layers.flatten(pool4)

dense1 = tf.layers.dense(flat, units=1500, activation=tf.nn.relu,kernel_regularizer=regularizer,name = 'dense1')
#layer_norm1 = tf.contrib.layers.layer_norm(dense1)
dropout3 = tf.layers.dropout(
      inputs=dense1, rate=0.4, training = is_training)

dense2 = tf.layers.dense(dropout3, units=500, activation=tf.nn.relu,kernel_regularizer=regularizer,name = 'dense2')

logits = tf.layers.dense(inputs=dense2, units=10,kernel_regularizer=regularizer,name = 'logits')

prediction = tf.nn.softmax(logits)

#cross_entropy = -tf.reduce_mean(ys*tf.log(prediction))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=ys))



branch_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=branch_logits, labels=ys))
#not used
def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate




# variable learning rate, not using now
#global_step = tf.Variable(0, trainable=False)
#starter_learning_rate = 0.01
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                           100000, 0.96, staircase=True)
optimizer = tf.train.AdamOptimizer(0.001)
    
l2_loss = tf.losses.get_regularization_loss()

joint_loss = 0.7*loss+0.3*branch_loss

grads = optimizer.compute_gradients(joint_loss + l2_loss)
for i, (g, v) in enumerate(grads):
    if g is not None:
        grads[i] = (tf.clip_by_norm(g, 5), v)
train_op = optimizer.apply_gradients(grads)


#train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy1)
#data process
x_train_ = np.load('40000_train_x.npy')
y_train_ = np.load('40000_train_y.npy')

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
count = 0
for train_index, test_index in kf.split(x_train_):
    count += 1
    
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
    
    
    
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)
    
    datagen.fit(x_train)
    
    batch_size   = 128
    epochs       = 100
    iterations   = 391
    EARLY_STOP_THRESHOLD = 0.01
    STOP_CHECK_EPOCH = 5
    
    
    
    #EPOCH = 1
    #BATCH_INDEX = 0
    #BATCH_SIZE = 50
    #TRAIN_SIZE = 32000
    
    
    stop_epoch = 0
    pre_loss = 100 #a big number
    for i in range(epochs):
        train_index = 0
        for j in range(iterations):
        #while(train_index < len(x_train)):
            tup = datagen.flow(x_train,y_train,batch_size=batch_size)[0]
            batch_xs = tup[0]
            batch_ys = tup[1]
            #batch_xs = x_train[train_index:train_index+batch_size]
            #batch_ys = y_train[train_index:train_index+batch_size]
            #train_index += batch_size
            sess.run(train_op, feed_dict={xs: batch_xs, ys: batch_ys})
        
        print('epoch ',i)
        #main
        print('*main')
        acc = compute_accuracy(
            x_train[0:1000], y_train[0:1000])
        print('train accuracy : ',acc)
    
    
        acc_list.append(acc)
        loss1 = sess.run(loss,feed_dict = {xs:x_train[0:1000,:,:], ys: y_train[0:1000,:],is_training :False})
        print('loss : ',loss1)
        loss_list.append(loss1)
        #main val
        val_acc = compute_accuracy(
            x_val[0:1000], y_val[0:1000])
        print('val accuracy',val_acc)
        
        loss2 = sess.run(loss,feed_dict = {xs:x_val[0:1000,:,:], ys: y_val[0:1000,:],is_training :False})
        print('val loss',loss2)
        val_acc_list.append(val_acc)
        val_loss_list.append(loss2)
        #branch
        acc = compute_branch_accuracy(
            x_train[:1000], y_train[:1000])
        branch_acc.append(acc)
        loss3 = sess.run(branch_loss,feed_dict = {xs:x_train[:1000], ys: y_train[:1000],is_training :False})
        branch_loss_list.append(loss3)
        print('branch_acc = ',acc)
        print('branch_loss = ',loss3)
    
        #branch_val
        acc = compute_branch_accuracy(
            x_val[:1000], y_val[:1000])
        loss4 = sess.run(branch_loss,feed_dict = {xs:x_val[0:1000,:,:], ys: y_val[0:1000,:],is_training :False})
        branch_val_acc.append(acc)
        branch_val_loss.append(loss4)
        print('branch_val_acc = ',acc)
        print('branch_val_loss = ',loss4)
        
        # tune early stop criterion
#        stop_epoch += 1
#        if i>0:
#            if i %5 == 0:
#                if loss2 > pre_loss + EARLY_STOP_THRESHOLD:
#                    break;
#        else:
#            #initial loss
#            pre_loss = loss2
    

    
    cv_acc.append(acc_list)
    cv_val.append(val_acc_list)
    cv_bacc.append(branch_acc)
    cv_bval.append(branch_val_acc)
    cv_epoch.append(stop_epoch)
    
   # saver = tf.train.Saver()
    #text = '%d %.2f %s' % (1, 99.3, 'Justin')
  #  name = '%s_%d_%s' % ('./joint_06_1202_2',count,'/model.ckpt')
  #  save_path = saver.save(sess, name)
    #because not doing cross validation now
    break;
    
print('**final epoch big evaluate**')
#main
print('*main')
acc = compute_all_accuracy(
    x_train_, y_train_)

print('main accuracy',val_acc)
val_acc = compute_all_accuracy(
    x_test, y_test)
print('test accuracy',val_acc)

branch_test_acc = compute_all_branch_accuracy(
    x_test, y_test)

print('branch_test acc',branch_acc)

branch_train = compute_all_branch_accuracy(
    x_train_, y_train_)

print('branch_train acc',branch_train)
#plot training epoch-accuracy plot

plt.plot(acc_list)
plt.plot(val_acc_list)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


plt.plot(loss_list)
plt.plot(val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='lower left')
plt.show()


plt.plot(branch_acc)
plt.plot(branch_val_acc)
plt.title('branch accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='lower right')
plt.show()


plt.plot(branch_loss_list)
plt.plot(branch_val_loss)
plt.title('branch loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='lower left')
plt.show()

