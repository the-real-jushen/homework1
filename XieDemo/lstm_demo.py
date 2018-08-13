# -*-coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import reader
DATA_PATH = "data/ptbdata"
HIDDEN_SIZE = 200   # lstm隐层单元个数
NUM_LAYERS = 2      # lstm层数
VOCAB_SIZE = 10000  # 词表大小

LEARNING_RATE = 0.1  # 学习率
TRAIN_BATCH_SIZE = 20  # 训练batch大小
TRAIN_NUM_STEP = 35    # 训练时截断的最大序列长度

# 测试阶段，batch设置为1
EVAL_BATCH_SIZE = 1  
EVAL_NUM_STEP = 1
# 运行批次
NUM_EPOCH = 2
KEEP_PROB = 0.5  # dropout率
MAX_GRAD_NORM = 5  # 超参数，控制梯度


class PTBModel(object):

    def __init__(self, is_training, batch_size, num_steps):
        # 使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层。
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])
        # 定义lstm模型, self.cell
        self._build_lstm(is_training)

        # 将LSTM中的状态初始化为全0数组，和其他神经网络类似，每次迭代优化时使用一个
        # batch的数据训练
        self.initial_state = self.cell.zero_state(batch_size, tf.float32)
        # 定义一个词向量矩阵，第一维是词表大小，第二维长度与lstm隐藏单元个数相同
        # [词表大小， 词的向量表示]
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])
        # 将原本单词ID转为单词向量。[batch_size, num_steps, HIDDEN_SIZE]
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)
        # inference过程
        logits = self._inference(inputs, num_steps)
        # 定义误差和训练优化器
        self._add_loss(logits=logits)

    def _build_lstm(self, is_training):
        # 定义一个基本的lstm cell，作为递归网络的基础结构。
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        # 使用dropoutwrapper类实现dropout功能，其通过连个参数，input_keep_prob输入的dropout概率和
        # output_keep_prob来控制输出的dropout概率
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=KEEP_PROB)
        # 使用MultiRNNCell堆叠多个带dropout层的lstm cell
        self.cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

    def _add_loss(self, logits):
        # 定义交叉熵损失函数和平均损失。
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([self.batch_size * self.num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / self.batch_size
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    def _inference(self, inputs, num_steps):
        # 定义输出列表。
        outputs = []
        state = self.initial_state
        with tf.variable_scope("RNN"):
            # 为了避免训练时梯度消失，定义一个最大的序列长度，用num_steps表示。
            for time_step in range(num_steps):
                # 因为递归网络在同层间的不同时刻参数是共享的，所以后续的时刻的计算
                # 会复用第一时刻的lstm结构定义的变量
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # 每一步处理时间序列中的一个时刻，传入当前时刻的输入和前一时刻的状态state，
                # 可以计算得到当前时刻的输出和状态
                cell_output, state = self.cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        self.final_state = state
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias
        return logits


def run_epoch(session, model, data, train_op, output_log):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)    #lstm单元初始状态

    # 训练一个epoch。
    for step,(x,y) in enumerate(reader.ptb_iterator(data, model.batch_size, model.num_steps)):
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                        {model.input_data: x, model.targets: y, model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)

def show_data():
    train_data, valid_data,test_data, _ = reader.ptb_raw_data("data/ptbdata/")
    result = reader.ptb_iterator(train_data, batch_size=16, num_steps=5)
    x, y = next(result)
    print("x:", x)
    print("y:", y)


def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("language_model", reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    #
    with tf.variable_scope("language_model", reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)
    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True)

            valid_perplexity = run_epoch(session, eval_model, valid_data, tf.no_op(), False)
            print("Epoch: %d Validation Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, eval_model, test_data, tf.no_op(), False)
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    main()