
from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
import h5py

# cluster specificationlocalhost:2222
parameter_servers = ["10.24.1.201:2225","10.24.1.202:2225"]
workers = [ "10.24.1.203:2225", "10.24.1.204:2225",
      "10.24.1.205:2225"]

h5f1 = h5py.File('train_label.h5','r')
label_train = h5f1['d1'][:]
h5f2 = h5py.File('test_label.h5','r')
label_test = h5f2['d2'][:]
h5f3 = h5py.File('train_word.h5','r')
tfidf_documents_train = h5f3['d3'][:]
h5f4 = h5py.File('test_word.h5','r')
tfidf_documents_test= h5f4['d4'][:]


cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster, 
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.001
training_epochs = 100
logs_path = "/home/prachi.sharma92/1"


if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):
        global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 

	xent = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
	loss = tf.reduce_mean(xent) #shape 1
	regularizer = tf.nn.l2_loss(W)
	cost = tf.reduce_mean(loss + 0.01 * regularizer)
	# Gradient Descent
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	prediction = tf.nn.softmax(tf.matmul(x, W) + b)
        test_prediction = tf.nn.softmax(tf.matmul(tfidf_documents_test, W) + b)
	def accuracy(predictions, labels):
	    p=0
	    k=predictions.shape[0]
	    for i in range(predictions.shape[0]):
		if (labels[i,np.argmax(predictions[i,:])]!=0):
		   p=p+1
	    return (100.0 * p/ labels.shape[0])
	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
        tf.summary.scalar("cost", cost)
        tf.summary.scalar("accuracy", accuracy(test_prediction,label_test))
	# merge all summaries into a single "operation" which we can execute in a session 
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
	print("Variables initialized ...")
    

    
  save_cost1=np.zeros((training_epochs,1))
  save_cost2=np.zeros((training_epochs,1))
  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

  begin_time = time.time()

  with sv.prepare_or_wait_for_session(server.target) as sess:
    
	    # create log writer object (this will log on every machine)
	    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
		
	    # perform training cycles
	    start_time = time.time()
	    for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(tfidf_documents_train.shape[0]/batch_size)
		# Loop over all batches
		for i in range(total_batch):
		    batch_xs=tfidf_documents_train[i*batch_size:i*batch_size+batch_size,:]
		    batch_ys = label_train[i*batch_size:i*batch_size+batch_size,:]
		    _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
		                                                  y: batch_ys})
		    # Compute average loss
		    avg_cost += c / total_batch
		# Display logs per epoch step
		if (epoch+1) % display_step == 0:
		    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
		print (accuracy(sess.run(prediction, feed_dict={x: tfidf_documents_test}),label_test))
                if FLAGS.task_index == 0:
                   save_cost1[epoch,0]=avg_cost
                else:
                   save_cost2[epoch,0]=avg_cost
	    
	    print ("train_accuracy=",accuracy(sess.run(prediction, feed_dict={x: tfidf_documents_train}),label_train))
            begin_time = time.time()
	    print ("test_accuracy=",accuracy(sess.run(prediction, feed_dict={x: tfidf_documents_test}),label_test))
            np.save("asyn1",save_cost1)
            np.save("asyn2",save_cost2)

  sv.stop()
