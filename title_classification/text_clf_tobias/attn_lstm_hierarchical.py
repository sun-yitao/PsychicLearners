from modules.attention import attention
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
from utils.prepare_data import *
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Hyperparameter
MAX_DOCUMENT_LENGTH = 16
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 64
ATTENTION_SIZE = 64
lr = 1e-3
BATCH_SIZE = 2048
KEEP_PROB = 0.5
LAMBDA = 0.0001
MAX_LABEL = 17
epochs = 100


# load data
data_dir = r"D:\yitao\tcmtf"
BIG_CATEGORY = 'beauty'
x_train, y_train = load_data(os.path.join(
    data_dir, f'{BIG_CATEGORY}_train_split.csv'), one_hot=True, n_class=MAX_LABEL, starting_class=0)
x_dev, y_dev = load_data(os.path.join(
    data_dir, f'{BIG_CATEGORY}_valid_split.csv'), one_hot=True,n_class=MAX_LABEL, starting_class=0)

# data preprocessing
x_train, x_dev, vocab_size = data_preprocessing_v2(
    x_train, x_dev, max_len=16)

graph = tf.Graph()
with graph.as_default():

    batch_x = tf.placeholder(tf.int32, [None, MAX_DOCUMENT_LENGTH])
    batch_y = tf.placeholder(tf.float32, [None, MAX_LABEL])
    keep_prob = tf.placeholder(tf.float32)

    embeddings_var = tf.Variable(tf.random_uniform([vocab_size, EMBEDDING_SIZE], -1.0, 1.0), trainable=True)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_x)
    # print(batch_embedded.shape)  # (?, 256, 100)
    rnn_outputs, _ = tf.nn.dynamic_rnn(BasicLSTMCell(HIDDEN_SIZE), batch_embedded, dtype=tf.float32)

    # Attention
    attention_output, alphas = attention(rnn_outputs, ATTENTION_SIZE, return_alphas=True)
    drop = tf.nn.dropout(attention_output, keep_prob)
    shape = drop.get_shape()

    # Fully connected layer（dense layer)
    W = tf.Variable(tf.truncated_normal([shape[1].value, MAX_LABEL], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[MAX_LABEL]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)


    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=batch_y))
    optimizer = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss)

    # Accuracy metric
    prediction = tf.argmax(tf.nn.softmax(y_hat), 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(batch_y, 1)), tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(graph=graph, config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized! ")
    saver = tf.train.Saver(tf.global_variables())
    os.makedirs(f"{BIG_CATEGORY}/attn_lstsm_hier", exist_ok=True)
    max_acc = 0
    print("Start trainning")
    start = time.time()
    for e in range(epochs):

        epoch_start = time.time()
        print("Epoch %d start !" % (e + 1))
        for x_batch, y_batch in fill_feed_dict(x_train, y_train, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: KEEP_PROB}
            l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict=fd)

        epoch_finish = time.time()
        dev_acc,dev_loss = sess.run([accuracy, loss], feed_dict={
            batch_x: x_dev,
            batch_y: y_dev,
            keep_prob: 1.0
        })
        print("Validation accuracy: ", dev_acc)
        print("epoch finished, time consumed : ", time.time() - epoch_start, " s")
        if dev_acc > max_acc:
            max_acc = dev_acc
            saver.save(
                sess, f"{BIG_CATEGORY}/attn_lstsm_hier/{dev_acc}.ckpt", global_step=e)

    print("Training finished, time consumed : ", time.time() - start, " s")
    """
    print("Start evaluating:  \n")
    cnt = 0
    test_acc = 0
    for x_batch, y_batch in fill_feed_dict(x_test, y_test, BATCH_SIZE):
            fd = {batch_x: x_batch, batch_y: y_batch, keep_prob: 1.0}
            acc = sess.run(accuracy, feed_dict=fd)
            test_acc += acc
            cnt += 1        
    
    print("Test accuracy : %f %%" % ( test_acc / cnt * 100))"""



