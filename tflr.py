from __future__ import print_function

import tensorflow as tf
import numpy as np
import h5py

#Loading saved file

h5f1 = h5py.File('train_label.h5','r')
train_output = h5f1['d1'][:]
h5f2 = h5py.File('test_label.h5','r')
test_output = h5f2['d2'][:]
h5f3 = h5py.File('train_word.h5','r')
train_word = h5f3['d3'][:]
h5f4 = h5py.File('test_word.h5','r')
test_word = h5f4['d4'][:]

#test_word = np.load(test_word).astype(np.float32)
#set parameters

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2
beta = 0.01

x = tf.placeholder("float", [None, train_word.shape[1]])
y = tf.placeholder("float", [None, 49])

print(train_word.shape)
print(train_output.shape)
print(test_word.shape)
print(test_output.shape)

# Set model weights
W = tf.Variable(tf.zeros([train_word.shape[1], 49]))
b = tf.Variable(tf.zeros([49]), dtype = tf.float32)


# Create a model

prediction = tf.nn.softmax(tf.matmul(x, W) + b)

#cross entropy

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))


# Loss function using L2 Regularization by ritchie ng
regularizer = tf.nn.l2_loss(W)
loss = tf.reduce_mean(loss + beta * regularizer)

def accuracy_of_model(predict, actual):
    corr=0
    k=predict.shape[0]
    for i in range(predict.shape[0]):
        if (actual[i,np.argmax(predict[i,:])]!=0):
           corr=corr+1
        
    return (100.0 * corr/ test_output.shape[0])


# Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
test_prediction = tf.nn.softmax(tf.matmul(test_word,W) + b)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_iteration):
        avg_cost = 0.
        total_batch = int(train_word.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = train_word[i*batch_size:i*batch_size+batch_size,:]
	    batch_ys = train_output[i*batch_size:i*batch_size+batch_size,:]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:  
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
	print(accuracy_of_model(test_prediction.eval(),test_output))
