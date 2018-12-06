# coding:utf-8
import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import pickle
import pandas as pd
curPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(curPath))

from utility.utilities import one_hot_representation
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)


class FM(object):
    """
    Factorization Machine with FTRL optimization
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        # num of features
        self.p = feature_length

    def add_placeholders(self):
        self.X = tf.placeholder('float32', [None, self.p])
        self.y = tf.placeholder('int64', [None,])
        self.keep_prob = tf.placeholder('float32')

    def inference(self):
        """
        forward propagation
        :return: labels for each sample

        tf.reduce_mean(input_tensor, axis=None, keep_dims=False, name=None, reduction_indices=None)计算张量的各个维度上的元素的平均值
        tf.truncated_normal_initializer()使用稀疏矩阵x乘以稠密矩阵v
        tf.subtract(x, y, name=None)返回一个Tensor，与 x 具有相同的类型
        tf.control_dependencies()判断每个元素是否为真,如果是可训练的变量则训练,不可训练的变量不训练
        """
        with tf.variable_scope('linear_layer'):
            b = tf.get_variable('bias', shape=[2], initializer=tf.zeros_initializer())
            w1 = tf.get_variable('w1', shape=[self.p, 2], initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
            # shape of [None, 2]
            self.linear_terms = tf.add(tf.matmul(self.X, w1), b)

        with tf.variable_scope('interaction_layer'):
            """计算交叉项"""
            # v是交互层的隐向量, self.k是潜变量的个数, 潜变量与潜变量交互
            v = tf.get_variable('v', shape=[self.p, self.k], initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01))
            # shape of [None, 1]
            self.interaction_terms = tf.multiply(0.5,
                                                 tf.reduce_mean(
                                                     tf.subtract(
                                                         tf.pow(tf.matmul(self.X, v), 2),
                                                         tf.matmul(tf.pow(self.X, 2), tf.pow(v, 2))),
                                                     1, keep_dims=True))
        # shape of [None, 2]
        self.y_out = tf.add(self.linear_terms, self.interaction_terms)
        self.y_out_prob = tf.nn.softmax(self.y_out)

    def add_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        # accuracy
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(model.y_out,1), tf.int64), model.y) # 逻辑值
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32)) # 将逻辑值转化为0-1
        # add summary to accuracy
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.FtrlOptimizer(self.lr,
                                           l1_regularization_strength=self.reg_l1,
                                           l2_regularization_strength=self.reg_l2)
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # 获得所有需要更新的变量
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for Model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()

def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoint")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my CNN architectures...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the my Factorization Machine")

def train_model(sess, model, epochs=10, print_every=500):
    """training Model"""
    num_samples = 0
    losses = []
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train_logs', sess.graph)
    for e in range(epochs):
        # get training data, iterable
        train_data = pd.read_csv('/home/zt/Documents/Data/avazu_CTR/train.csv', chunksize=model.batch_size)
        # batch_size data
        for data in train_data:
            actual_batch_size = len(data)
            batch_X = []
            batch_y = []
            batch_idx = []
            for i in range(actual_batch_size):
                sample = data.iloc[i,:] # 一条样本
                array,idx = one_hot_representation(sample,fields_train_dict, train_array_length)
                batch_X.append(array[:-2]) # 离散变量
                batch_y.append(array[-1])  # 标签
                batch_idx.append(idx)      # 每个样本有那些是1
            batch_X = np.array(batch_X)
            batch_y = np.array(batch_y)
            batch_idx = np.array(batch_idx)
            # create a feed dictionary for this batch
            feed_dict = {model.X: batch_X,
                         model.y: batch_y,
                         model.keep_prob:1.0}
            loss, accuracy,  summary, global_step, _ = sess.run([model.loss,
                                                                 model.accuracy,
                                                                 merged,
                                                                 model.global_step,
                                                                 model.train_op],
                                                                feed_dict=feed_dict)
            # 将批损失,展开给每个样本,放入losses中,记录每个样本的损失
            losses.append(loss*actual_batch_size)

            num_samples += actual_batch_size
            # Record summaries and train.csv-set accuracy
            train_writer.add_summary(summary, global_step=global_step)
            # print training loss and accuracy
            if global_step % print_every == 0:
                logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                             .format(global_step, loss, accuracy))
                saver.save(sess, "checkpoints/Model", global_step=global_step)

        # print loss of one epoch
        total_loss = np.sum(losses)/num_samples
        print("Epoch {1}, Overall loss = {0:.3g}".format(total_loss, e+1))

def validation_model(sess, model, print_every=50):
    """testing Model"""
    # num samples
    num_samples = 0
    # num of correct predictions
    num_corrects = 0
    losses = []
    # Merge all the summaries and write them out to train_logs
    merged = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter('test_logs', sess.graph)
    # get testing data, iterable
    validation_data = pd.read_csv('/home/zt/Documents/Data/utility/train.csv', chunksize=model.batch_size)
    # testing step
    valid_step = 1
    # batch_size data
    for data in validation_data:
        actual_batch_size = len(data)
        batch_X = []
        batch_y = []
        batch_idx = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array,idx = one_hot_representation(sample,fields_train_dict, train_array_length)
            batch_X.append(array[:-2])
            batch_y.append(array[-1])
            batch_idx.append(idx)
        batch_X = np.array(batch_X)
        batch_y = np.array(batch_y)
        batch_idx = np.array(batch_idx)
        # create a feed dictionary for this batch,
        feed_dict = {model.X: batch_X, model.y: batch_y,
                 model.feature_inds: batch_idx, model.keep_prob:1}
        loss, accuracy, correct, summary = sess.run([model.loss, model.accuracy,
                                                     model.correct_prediction, merged,],
                                                    feed_dict=feed_dict)
        # aggregate performance stats
        losses.append(loss*actual_batch_size)
        num_corrects += correct
        num_samples += actual_batch_size
        # Record summaries and train.csv-set accuracy
        test_writer.add_summary(summary, global_step=valid_step)
        # print training loss and accuracy
        if valid_step % print_every == 0:
            logging.info("Iteration {0}: with minibatch training loss = {1} and accuracy of {2}"
                         .format(valid_step, loss, accuracy))
        valid_step += 1
    # print loss and accuracy of one epoch
    total_correct = num_corrects/num_samples
    total_loss = np.sum(losses)/num_samples
    print("Overall test loss = {0:.3g} and accuracy of {1:.3g}" \
          .format(total_loss,total_correct))


def test_model(sess, model, print_every = 50):
    """training Model"""
    # get testing data, iterable
    test_data = pd.read_csv('/home/zt/Documents/Data/avazu_CTR/test.csv',
                            chunksize=model.batch_size)
    test_step = 1
    # batch_size data
    for data in test_data:
        actual_batch_size = len(data)
        batch_X = []
        batch_idx = []
        for i in range(actual_batch_size):
            sample = data.iloc[i,:]
            array,idx = one_hot_representation(sample, fields_test_dict, test_array_length)
            batch_X.append(array)
            batch_idx.append(idx)

        batch_X = np.array(batch_X)
        batch_idx = np.array(batch_idx)
        # create a feed dictionary for this batch
        feed_dict = {model.X: batch_X, model.keep_prob:1, model.feature_inds:batch_idx}
        # shape of [None,2]
        y_out_prob = sess.run([model.y_out_prob], feed_dict=feed_dict)
        # write to csv files
        data['click'] = y_out_prob[0][:,-1]
        if test_step == 1:
            data[['id','click']].to_csv('ch18_FM.csv', mode='a', index=False, header=True)
        else:
            data[['id','click']].to_csv('ch18_FM.csv', mode='a', index=False, header=False)

        test_step += 1
        if test_step % 50 == 0:
            logging.info("Iteration {0} has finished".format(test_step))



if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''

    # original fields
    fields_train = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                    'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                    'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
                    'device_conn_type','click']
    fields_test = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                   'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                   'app_id', 'device_id', 'app_category', 'device_model', 'device_type',
                   'device_conn_type']
    # loading dicts
    fields_train_dict = {}
    for field in fields_train:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_train_dict[field] = pickle.load(f)

    fields_test_dict = {}
    for field in fields_test:
        with open('dicts/' + field + '.pkl', 'rb') as f:
            fields_test_dict[field] = pickle.load(f)
    # length of representation
    train_array_length = max(fields_train_dict['click'].values()) + 1
    test_array_length = train_array_length - 2
    # initialize the Model
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 512
    config['reg_l1'] = 2e-2
    config['reg_l2'] = 0
    config['k'] = 40
    # get feature length
    feature_length = test_array_length

    # initialize ch18_FM Model
    model = FM(config)
    # build graph for Model
    model.build_graph()

    saver = tf.train.Saver(max_to_keep=5)

    with tf.Session() as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        # restore trained parameters
        check_restore_parameters(sess, saver)

        print('start training...')
        train_model(sess, model, epochs=20, print_every=500)
        # if mode == 'test':
        #     print('start testing...')
        #     test_model(sess, model)