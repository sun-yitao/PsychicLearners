from modules.multihead import *
from utils.model_helper import *
import time
from utils.prepare_data import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class AttentionClassifier(object):
    def __init__(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)

    def build_graph(self):
        print("building graph...")
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, self.x)
        # multi-head attention
        ma = multihead_attention(queries=batch_embedded, keys=batch_embedded)
        # FFN(x) = LN(x + point-wisely NN(x))
        outputs = feedforward(ma, [self.hidden_size, self.embedding_size])
        outputs = tf.reshape(outputs, [-1, self.max_len * self.embedding_size])
        logits = tf.layers.dense(outputs, units=self.n_class)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.label))
        self.prediction = tf.argmax(tf.nn.softmax(logits), 1)

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, 0.9, use_nesterov=True)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        print("graph built successfully!")


if __name__ == '__main__':
    # load data
    data_dir = r"D:\yitao\tcmtf"
    BIG_CATEGORY = 'beauty'
    N_CLASS = 17
    x_train, y_train = load_data(os.path.join(
        data_dir, f'{BIG_CATEGORY}_train_split.csv'), one_hot=False, n_class=N_CLASS, starting_class=0)
    x_dev, y_dev = load_data(os.path.join(
        data_dir, f'{BIG_CATEGORY}_valid_split.csv'), one_hot=False, n_class=N_CLASS, starting_class=0)

    #data preprocessing
    x_train, x_dev, vocab_size = data_preprocessing_v2(
       x_train, x_dev, max_len=16)

    config = {
        "max_len": 16,
        "hidden_size": 64,
        "vocab_size": vocab_size,
        "embedding_size": 128,
        "n_class": N_CLASS,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "train_epoch": 20
    }

    classifier = AttentionClassifier(config)
    classifier.build_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    os.makedirs(f"{BIG_CATEGORY}/ind_rnn", exist_ok=True)
    max_acc = 0
    dev_batch = (x_dev, y_dev)
    start = time.time()
    for e in range(config["train_epoch"]):

        t0 = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
            return_dict = run_train_step(classifier, sess, (x_batch, y_batch))

        t1 = time.time()

        print("Train Epoch time:  %.3f s" % (t1 - t0))
        dev_acc = run_eval_step(classifier, sess, dev_batch)
        print("validation accuracy: %.3f " % dev_acc)
        if dev_acc > max_acc:
            max_acc = dev_acc
            saver.save(
                sess, f"{BIG_CATEGORY}/ind_rnn/{dev_acc}.ckpt", global_step=e)

    print("Training finished, time consumed : ", time.time() - start, " s")
    """
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
        acc = run_eval_step(classifier, sess, (x_batch, y_batch))
        test_acc += acc
        cnt += 1

    print("Test accuracy : %f %%" % (test_acc / cnt * 100))"""

