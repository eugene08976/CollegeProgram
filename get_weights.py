import tensorflow as tf
import os


#layers = [conv1_1, conv1_2, conv2_1,conv2_2,conv3_1,conv3_2,conv3_3,conv4_1,conv4_2,conv4_3,dense1,dense2,logits,bconv1_1,bconv1_2,bconv2_1,bconv2_2,branch_logits]

layers = [conv1_1,conv1_2,conv2_1,conv2_2,local_conv2_1,local_conv2_2,logits]
par_list = []

for layer in layers:
    weights = tf.get_default_graph().get_tensor_by_name(
      os.path.split(layer.name)[0] + '/kernel:0')
    
    bias = tf.get_default_graph().get_tensor_by_name( os.path.split(layer.name)[0] + '/bias:0')
    
    weights1 = weights.eval(session=sess)
    
    bias1 = bias.eval(session = sess)
    par_list.append(weights1)
    par_list.append(bias1)


import pickle

npy = [1,2,3]
#a = {'hello': 'world'}
a = par_list
with open('0802_local_all.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

#check if save correctly
with open('0802_local_all.pickle', 'rb') as handle:
    b = pickle.load(handle)
