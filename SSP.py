from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
import h5py


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
training_epochs = 3
logs_path = "/home/prachi.sharma92/1"

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

 # Build model...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):
        global_step = tf.get_variable('global_step', [], 
                                initializer = tf.constant_initializer(0), 
                                trainable = False, dtype=tf.int32)

	# tf Graph Input
	x = tf.placeholder(tf.float32, [None, tfidf_documents_train.shape[1]]) 
	y = tf.placeholder(tf.float32, [None, 50]) 
	# Set model weights
	W = tf.Variable(tf.random_normal([tfidf_documents_train.shape[1], 50]))
	b = tf.Variable(tf.random_normal([50]))

	weight=tf.reshape(tf.reduce_sum(y,0)/tf.reduce_sum(tf.reduce_sum(y,0)),[1,50])

	# Construct model
	pred = (tf.matmul(x, W) + b) # Softmax
	weight_per_label = tf.transpose( tf.matmul(y
		                   , tf.transpose(weight)) ) 

	xent = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
	loss = tf.reduce_mean(xent) #shape 1
	regularizer = tf.nn.l2_loss(W)
	cost = tf.reduce_mean(loss + 0.01 * regularizer)
	# Gradient Descent
        grad_op = tf.train.AdamOptimizer(learning_rate)
      
        rep_op = tf.contrib.opt.DropStaleGradientOptimizer(grad_op, 
                                          staleness=10,use_locking=True
                                          )
        optimizer = rep_op.minimize(cost, global_step=global_step)
      
        
      
   
       

	test_prediction = tf.nn.softmax(tf.matmul(tfidf_documents_test, W) + b)
	def accuracy(predictions, labels):
	    p=0
	    k=predictions.shape[0]
	    for i in range(predictions.shape[0]):
		if (labels[i,np.argmax(predictions[i,:])]!=0):
		   p=p+1
	    return (100.0 * p/ label_test.shape[0])
	init = tf.global_variables_initializer()
        tf.summary.scalar("cost", cost)
        tf.summary.scalar("accuracy", accuracy(test_prediction,label_test))
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
    

    

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
		print (accuracy(test_prediction.eval(session=sess),label_test))
	    
	    print("Test-Accuracy: %2.2f" % accuracy(test_prediction.eval(session=sess),label_test))
	    print("Total Time: %3.2fs" % float(time.time() - begin_time))
	    #print("Final Cost: %.4f" % cost)

  sv.stop()
