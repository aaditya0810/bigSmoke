import tensorflow as tf
from numpy import genfromtxt 
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

n = 10
tmp = genfromtxt("bitcoin-close.csv")
dat = np.zeros(tmp.shape[0],dtype=np.float32)
dat[:] = tmp

train_x = np.zeros([n,dat.shape[0]-n], dtype=np.float32)
train_y = np.zeros([dat.shape[0]-n,1], dtype=np.float32)
for i in range(n):
  train_x[i,:] = dat[i:i+dat.shape[0]-n]
train_x = np.transpose(train_x)
train_y[:,0] = dat[n:]

n_l1 = 20
n_cls = 1
bt_size = 100

x = tf.placeholder('float', [None, n])
y = tf.placeholder('float')

def model(data):
  h_1_l = {'weights':tf.Variable(tf.random_normal([n, n_l1])), 'biases':tf.Variable(tf.random_normal([n_l1]))}
  output_layer = {'weights':tf.Variable(tf.random_normal([n_l1, n_cls])), 'biases':tf.Variable(tf.random_normal([n_cls]))}
  l1 = tf.add(tf.matmul(data,h_1_l['weights']), h_1_l['biases'])
  l1 = tf.nn.sigmoid(l1)
  output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']
  return output

def train(x):
  pred = model(x)
  cost = tf.reduce_mean(tf.log(tf.abs(pred - y)))
  opt = tf.train.AdamOptimizer().minimize(cost)
  ep_t = 10
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(ep_t):
      ep_loss = 0
      i = 0
      while i < len(train_x):
        start = i
        end = i + bt_size
        batch_x = np.array(train_x[start:end])
        batch_y = np.array(train_y[start:end])
        print(batch_x)
        _, c = sess.run([opt, cost], feed_dict={x: batch_x, y: batch_y})
        ep_loss += c
        i += bt_size
      print('Epoch', ep, ' completed out of ', ep_t, ' loss:', ep_loss)
    correct = tf.greater(0.1, tf.abs(pred - y))        
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:',accuracy.eval({x:train_x, y:train_y}))

train(train_x)