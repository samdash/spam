import tensorflow as tf
import numpy as np
import os
import utils
import pickle


# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
PARENT_DIR_PATH = os.path.dirname(os.path.realpath(os.path.join(__file__, '..')))
SPAM_MODEL = os.path.join(PARENT_DIR_PATH, "r2d2-spam","checkpoints")
VOCAB_FILE = os.path.join(PARENT_DIR_PATH, "r2d2-spam", "checkpoints","vocab_shape.pickle")


with open (VOCAB_FILE, 'rb') as fp:
    vocabulary,shape = pickle.load(fp)


def eval():
    with tf.device('/cpu:0'):
        x_text, y = utils.load_data_from_disk()

    # Map data into vocabulary
    text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(shape)
    text_vocab_processor.restore(VOCAB_FILE)
    x_text = [" ".join(x) for x in x_text]
    x_eval = np.array(list(text_vocab_processor.fit_transform(x_text)))
    y_eval = np.argmax(y, axis=1)

    checkpoint_file = tf.train.latest_checkpoint(SPAM_MODEL)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = utils.batch_iter(list(x_eval), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []
            for x_batch in batches:
                batch_predictions = sess.run(predictions, {input_text: x_batch,
                                                           dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

            correct_predictions = float(sum(all_predictions == y_eval))
            print("Total number of test examples: {}".format(len(y_eval)))
            print("Accuracy: {:g}".format(correct_predictions / float(len(y_eval))))


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()
