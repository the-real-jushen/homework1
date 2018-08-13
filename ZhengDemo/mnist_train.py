# -*-coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义训练相关参数
BATCH_SIZE = 100
learning_rate = 0.01
TRAINING_STEPS = 30000

# 定义神经网络结构相关的参数。
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
LAYER2_NODE = 200

def inference(input_tensor):
    with tf.variable_scope('layer1'):
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)


    with tf.variable_scope('layer2'):
        weights = tf.get_variable("weights", [LAYER1_NODE, LAYER2_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope('layerOut'):
        weights = tf.get_variable("weights", [LAYER2_NODE, OUTPUT_NODE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(layer2, weights) + biases

    return out

def inputs():
    # 定义输入输出placeholder。
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    return x, y_

def loss(y_pred, y_real):
    # 定义损失函数, 这里使用softmax_cross_entropy_with_logits, 首先在对inference输出添加一个
    # softmax层归一化概率，然后计算inference的误差
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_real)
    loss = tf.reduce_mean(cross_entropy)
    return loss

def accuracy(y_pred, y_real):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_real, 1))
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return acc

def train_optimizer(loss):
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return train_step

def train(mnist):
    x, y_ = inputs()
    y = inference(x)
    losses = loss(y_pred=y, y_real=y_)
    acc = accuracy(y_pred=y, y_real=y_)
    train_step =train_optimizer(loss=losses)
    
    with tf.Session() as sess:
        valid_acc_last=0.0000001
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_step, losses], feed_dict={x: xs, y_: ys})
            if i % 500 ==0:
                print("After %d training step(s), loss on training batch is %g." % (i, loss_value))
            if i % 1000 == 0:
                valid_acc = sess.run(acc, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print("After %d training step(s), accuracy on validation is %g." % (i, valid_acc))
                valid_acc_last=valid_acc
                if valid_acc<valid_acc:
                    print("After %d training step(s), accuracy on validation is %g.It went up so over fitting quuit training" % (i, valid_acc))
                    break
                    
        test_acc = sess.run(acc, feed_dict={x:mnist.test.images, y_:mnist.test.labels})
        print("After %d training step(s), accuracy on test is %g." % (TRAINING_STEPS, test_acc))

if __name__ == "__main__":
    mnist = input_data.read_data_sets("data/mnist_data", one_hot=True)
    train(mnist)