import tensorflow as tf
import time
import datetime
import os
from cnn_classifier import CNNClassifier
from utils import load_data, batch_iter
import pickle
import turibolt as bolt


class CreateModel(object):

    def __init__(self,
                 flags='flags'
                 ):

        self.FLAGS = flags

    def train_model(self):

        # Load data
        print("Loading data...")
        x_text, y, vocabulary, vocabulary_inv = load_data()

        # Split train/test set
        dev_sample_index = -1 * int(self.FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_text[:dev_sample_index], x_text[dev_sample_index:]
        y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
        print("Vocabulary Size: {:d}".format(len(vocabulary)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.FLAGS.allow_soft_placement,
                log_device_placement=self.FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = CNNClassifier(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocabulary),
                    embedding_size=self.FLAGS.embedding_dim,
                    filter_sizes=list(map(int, self.FLAGS.filter_sizes.split(","))),
                    num_filters=self.FLAGS.num_filters,
                    l2_reg_lambda=self.FLAGS.l2_reg_lambda)

                writer = tf.summary.FileWriter('tensorlogs')
                writer.add_graph(sess.graph)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.join(bolt.ARTIFACT_DIR, "checkpoints")
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.FLAGS.num_checkpoints)
                vocab_and_shape = os.path.join(checkpoint_dir, "vocab_shape.pickle")
                # Write vocabulary
                print('saving the vocabulary and input shape to file .... ')
                with open(vocab_and_shape, 'wb') as fp:
                    # protocol=2 is for python 2.7 or running on Bolt ( as bolt support only python2.7 ), remove this for python 3.6
                    pickle.dump((vocabulary, x_text.shape[1]), fp, protocol=2)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):

                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: self.FLAGS.dropout_keep_prob
                    }
                    _, step, loss, accuracy = sess.run(
                        [train_op, global_step, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                def test_step(x_batch, y_batch):

                    feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch, cnn.dropout_keep_prob: 1.0}
                    step, loss, accuracy = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
                    print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

                batches = batch_iter(list(zip(x_train, y_train)), self.FLAGS.batch_size, self.FLAGS.num_epochs)

                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % self.FLAGS.checkpoint_every == 0:
                        print("\nEvaluation:")
                        test_step(x_batch, y_batch)
                        print("")

                    if current_step % self.FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
