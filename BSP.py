

from __future__ import print_function

import tensorflow as tf
import sys
import time
import numpy as np
import h5py

parameter_servers = ["10.24.1.206:2226","10.24.1.207:2226"]
workers = [ "10.24.1.209:2226", "10.24.1.210:2226",
      "10.24.1.211:2226"]

h5f1 = h5py.File('train_label.h5','r')
label_train = h5f1['d1'][:]
h5f2 = h5py.File('test_label.h5','r')
label_test = h5f2['d2'][:]
h5f3 = h5py.File('train_word.h5','r')
tfidf_documents_train = h5f3['d3'][:]
h5f4 = h5py.File('test_word.h5','r')
tfidf_documents_test= h5f4['d4'][:]
print(label_train.shape)
print(label_test.shape)
print(tfidf_documents_train.shape)
print(tfidf_documents_test.shape)



cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})



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

  # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
        
      
   
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
    
	

	test_prediction = tf.nn.softmax(tf.matmul(tfidf_documents_test, W) + b)
	def accuracy(predictions, labels):
	    p=0
	    k=predictions.shape[0]
	    for i in range(predictions.shape[0]):
		if (labels[i,np.argmax(predictions[i,:])]!=0):
		   p=p+1
	   # print (np.argmax(predictions[50,:]),predictions[50,:],labels[50,:]) 
	    return (100.0 * p/ label_test.shape[0])
	# Initialize the variables (i.e. assign their default value)
	init = tf.global_variables_initializer()
        tf.summary.scalar("cost", cost)
        tf.summary.scalar("accuracy", accuracy(test_prediction,label_test))
	# merge all summaries into a single "operation" which we can execute in a session 
	summary_op = tf.summary.merge_all()
	init_op = tf.global_variables_initializer()
    

    

  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

  begin_time = time.time()

  with sv.prepare_or_wait_for_session(server.target) as sess:
          
	    # is chief
	    if FLAGS.task_index == 0:
	      sv.start_queue_runners(sess, [chief_queue_runner])
	      sess.run(init_token_op)
   
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
