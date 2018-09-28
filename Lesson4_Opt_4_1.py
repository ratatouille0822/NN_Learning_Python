#coding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = 2
import tensorflow as tf
import numpy as ny
BATCH_SIZE = 8
seed = 23455

rdm = tf.random.RandomState(seed)
X = rdm.rand(32,2)
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

x = tf.placeholder(tf.float32,shape(None,2))
y_= tf.placeholder(tf.float32,shape(None,1))

w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))

y=tf.matmul(x,w1)

loss_mse = tf.reduce_mean(tf.squared_difference(y,y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

with tf.Session as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    STEPS = 20000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(train_step,feed_dict={x:X,y_:Y})
            print("After %d train steps, total loss is %g"%(i,total_loss))
    print("w1 is : \n",sess.run(w1))

