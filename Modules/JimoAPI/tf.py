import tensorflow as tf
import numpy as np
import pathlib
import socket
import math
import json
import os

import JimoAPI

tf.logging.set_verbosity(tf.logging.ERROR)

max_select_size = 100

default_cluster_port = 3330

cluster = []


def _setup_server():
    global cluster
    global server
    cluster = [i + ':' + str(default_cluster_port) for i in [socket.gethostbyname(socket.gethostname())] + cluster]
    cluster_spec = {'master': [cluster[0]]}
    if len(cluster) > 1:
        cluster_spec['worker'] = [cluster[-1]]
        if len(cluster) > 2:
            cluster_spec['slave'] = cluster[1:-1]
    cluster = tf.train.ClusterSpec(cluster_spec)
    server = tf.train.Server(cluster, job_name='master', task_index=0)


def input_fn(data, return_type):
    if data['type'][0] == 'D':
        return input_fn_dnn(**data, return_type=return_type)
    elif data['type'][0] == 'R':
        return input_fn_rnn(**data, return_type=return_type)
    else:
        raise NotImplementedError('type {} not supported'.format(data['type']))


def input_fn_dnn(table_name, return_type, train_percent=0.7, batch_size=100, **kwargs):
    def return_func(features, labels, param=None):
        with JimoAPI.TempCursor() as cx:
            cx.execute("SELECT count (*) FROM {};", table_name)
            table_size = list(cx.fetchone().values())[0]
            if return_type == 'train':
                start = 0
                num_large_batches = math.ceil(table_size/max_select_size)
                for i in range(1, num_large_batches+1):
                    step_size = round(param / (int(table_size * train_percent) / max_select_size if i < num_large_batches else 1))
                    cx.execute("SELECT {} FROM {} LIMIT {}, {};", (str([i['column'] for i in features] + [labels['column']])[1:-1].replace("'", ""),
                                                                   table_name, start, max_select_size if i < num_large_batches else int(table_size * train_percent)))
                    start += max_select_size
                    data = cx.fetchall()
                    label = data.pop(labels['column'])
                    rand_array = np.random.randint((max_select_size if i < num_large_batches else int(table_size * train_percent)) - batch_size, size=[step_size])
                    for start_point in rand_array:
                        yield {i: j[start_point:start_point+batch_size] for i, j in data.items()}, label[start_point:start_point+batch_size]
            elif return_type == "predict":
                start = 0
                while start < table_size:
                    cx.execute("SELECT {} FROM {} LIMIT {}, {};", (str([i['column'] for i in features] + [param['column']])[1:-1].replace("'", ""), table_name,
                                                                   start, min(table_size-start, max_select_size)))
                    start += max_select_size
                    while True:
                        data = cx.fetchmany(batch_size)
                        if data is None:
                            break
                        primary_key = data.pop(param['column'])
                        yield data, primary_key
            elif return_type == "evaluate":
                start = int(table_size*train_percent)
                while start < table_size:
                    cx.execute("SELECT {} FROM {} LIMIT {}, {};", (str([i['column'] for i in features] + [labels['column']])[1:-1].replace("'", ""),
                                                                   table_name, start, min(table_size-start, max_select_size)))
                    start += max_select_size
                    while True:
                        data = cx.fetchmany(batch_size)
                        if data is None:
                            break
                        label = data.pop(labels['column'])
                        yield data, label
    return return_func


def input_fn_rnn(table_name, return_type):
    pass


class custom_estimator():
    def __init__(self, name, features, labels, network_type, hidden_units, learning_rate, beta1, beta2, epsilon, dropout, hook):
        raise NotImplementedError

    def train(self, batch_fn, steps):
        raise NotImplementedError

    def evaluate(self, batch_fn):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class custom_hook():
    def __init__(self, estimator):
        self.estimator = estimator

    def begin_train(self, *args):
        pass

    def begin_train_step(self, *args):
        pass

    def after_train_step(self, *args):
        pass

    def after_train(self, *args):
        pass

    def begin_evaluate(self, *args):
        pass

    def begin_evaluate_step(self, *args):
        pass

    def after_evaluate_step(self, *args):
        pass

    def after_evaluate(self, *args):
        pass


class deep_neural_network(custom_estimator):
    def __init__(self, name, features, labels, network_type, primary_key=None,
                 hidden_units=[128, 128, 128, 128, 128, 128, 128, 128], learning_rate=.001, beta1=.9, beta2=.999, epsilon=1e-8, dropout=0.,
                 cost_fn=None, hook=custom_hook, **kwargs):
        tf.reset_default_graph()
        self.name = name
        self.features = features
        self.labels = labels
        self.primary_key = primary_key
        self.network_type = network_type
        self._hook = hook(self)
        self._initialize_layers(hidden_units, learning_rate, beta1, beta2, epsilon, dropout, cost_fn)
        with tf.device('job:master'):
            self._saver = tf.train.Saver(max_to_keep=10)
        self._sess = tf.Session(server.target)
        self._merge = tf.summary.merge_all()
        if pathlib.Path('./.model/'+self.name+'/checkpoint').exists():
            self._saver.restore(self._sess, get_checkpoint(self.name))
            self._sess.run(tf.tables_initializer())
        else:
            self._sess.run((tf.global_variables_initializer(), tf.tables_initializer()))
            pathlib.Path('./.model/'+self.name).mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.FileWriter('./.model/'+self.name, self._sess.graph)

    def train(self, batch_fn, steps):
        self._hook.begin_train(steps)
        for feature_batch, label_batch in batch_fn(self.features, self.labels, steps):
            feed_dict = {self._feature_placeholders[i['column']]: feature_batch[i['column']] for i in self.features}
            feed_dict[self._label_placeholder] = label_batch
            self._hook.begin_train_step()
            summary, _, cost = self._sess.run((self._merge, self._train_fn, self._cost_fn), feed_dict)
            self._writer.add_summary(summary, self.step)
            self._hook.after_train_step(cost)
        self._hook.after_train()

    def evaluate(self, batch_fn):
        accuracy = []
        self._hook.begin_evaluate()
        for feature_batch, label_batch in batch_fn(self.features, self.labels):
            feed_dict = {self._feature_placeholders[i['column']]: feature_batch[i['column']] for i in self.features}
            feed_dict[self._label_placeholder] = label_batch
            self._hook.begin_evaluate_step()
            summary, result = self._sess.run((self._merge, self._evaluate_fn), feed_dict)
            self._writer.add_summary(summary, self.step)
            self._hook.after_evaluate_step()
            if self.network_type == 'DNNClassifier':
                accuracy.append(result/len(label_batch))
            elif self.network_type == 'DNNRegressor':
                accuracy.append(abs(result))
            else:
                raise NotImplementedError('unrecognized type')
        accuracy = np.mean(accuracy)
        self._hook.after_evaluate(accuracy)
        return accuracy

    def predict(self, batch_fn):
        for feature_batch, primary_key in batch_fn(self.features, self.labels, self.primary_key):
            feed_dict = {self._feature_placeholders[i['column']]: feature_batch[i['column']] for i in self.features}
            result = self._sess.run(self._predict_fn, feed_dict)
            if self.labels['type'] == 'vocab':
                yield zip(primary_key, [self.labels['vocab'][i] for i in np.argmax(result, axis=1)])
            elif self.labels['type'] == 'indicator':
                yield zip(primary_key, np.argmax(result, axis=1))
            else:
                yield zip(primary_key, result)

    def save(self, checkpoint=True):
        pathlib.Path('./.model/'+self.name).mkdir(parents=True, exist_ok=True)
        if checkpoint:
            self._saver.save(self._sess, './.model/{0}/{0}'.format(self.name), global_step=self.step)
        else:
            self._saver.save(self._sess, './.model/{0}/{0}'.format(self.name))

    def close(self):
        self._sess.close()

    @property
    def step(self):
        return self._sess.run(self._step)

    def _initialize_layers(self, hidden_units, learning_rate, beta1, beta2, epsilon, dropout, cost_fn):
        self._feature_placeholders = {}
        feature_columns = []
        with tf.device('job:master'):
            for i in self.features:
                placeholder, column = self._generate_placeholder(i)
                self._feature_placeholders[i['column']] = placeholder
                feature_columns.append(column)
            feature_columns = tf.feature_column.input_layer(self._feature_placeholders, feature_columns)
            self._label_placeholder, label_column = self._generate_placeholder(self.labels)
            label_column = tf.feature_column.input_layer({self.labels['column']: self._label_placeholder}, label_column)
        y = self._generate_hidden_layers(feature_columns, hidden_units, dropout)
        device = 'job:worker' if 'worker' in cluster.jobs else 'job:master'
        with tf.device(device):
            self._step = tf.Variable(0, name='global_step', trainable=False)
            self._evaluate_fn = self._get_evaluation(y, label_column)
            self._predict_fn = self._get_result(y)
            self._cost_fn = self._get_cost(y, label_column, cost_fn)
            self._train_fn = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(self._cost_fn, self._step, name='Train')
        tf.summary.histogram('cost', self._cost_fn)
        tf.summary.histogram('accuracy', self._evaluate_fn)

    def _generate_placeholder(self, info):
        if info['type'] == 'vocab':
            placeholder = tf.placeholder(tf.string, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(info['column'], info['vocab']))
        elif info['type'] == 'hash':
            placeholder = tf.placeholder(tf.string, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket(info['column'], info['num_buckets']))
        elif info['type'] == 'indicator':
            placeholder = tf.placeholder(tf.int32, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity(info['column'], info['num_buckets']))
        elif info['type'] == 'numeric':
            placeholder = tf.placeholder(tf.float32, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.numeric_column(info['column'])
        else:
            raise NotImplementedError('type {} not supported'.format(info['type']))

    def _generate_hidden_layers(self, feature_columns, hidden_units, dropout):
        if 'slave' in cluster.jobs:
            task_id = np.linspace(0, cluster.num_tasks('slave')-.01, len(hidden_units), dtype=np.int32)
            y = feature_columns
            for idx, i in hidden_units:
                with tf.device('/job:slave/task:'+str(task_id[idx])):
                    y = tf.layers.dropout(tf.layers.batch_normalization(tf.layers.dense(y, i, tf.nn.relu, name='dense_{}'.format(idx)),
                                          name='batch_normalization_{}'.format(idx)), dropout, name='dropout_{}'.format(idx))
        else:
            with tf.device('job:master'):
                y = feature_columns
                for idx, i in enumerate(hidden_units):
                    y = tf.layers.dropout(tf.layers.batch_normalization(tf.layers.dense(y, i, tf.nn.relu, name='dense_{}'.format(idx)),
                                          name='batch_normalization_{}'.format(idx)), dropout, name='dropout_{}'.format(idx))
        if self.network_type == 'DNNClassifier':
            return tf.layers.dense(y, len(self.labels['vocab']) if self.labels['type'] == 'vocab' else self.labels['num_buckets'])
        elif self.network_type == 'DNNRegressor':
            return tf.layers.dense(y, 1)

    def _get_evaluation(self, y, y_true):
        if self.network_type == 'DNNClassifier':
            return tf.count_nonzero(tf.equal(tf.argmax(tf.nn.softmax(y), axis=1), tf.argmax(y_true, axis=1)))
        elif self.network_type == 'DNNRegressor':
            return tf.reduce_mean(y-y_true)
        else:
            raise NotImplementedError('unrecognized type')

    def _get_result(self, y):
        if self.network_type == 'DNNClassifier':
            return tf.nn.softmax(y)
        elif self.network_type == 'DNNRegressor':
            return y
        else:
            raise NotImplementedError('unrecognized type')

    def _get_cost(self, y, y_true, cost_fn):
        if self.network_type == 'DNNClassifier':
            return cost_fn(y_true, y) if cost_fn is not None else tf.losses.softmax_cross_entropy(y_true, y)
        elif self.network_type == 'DNNRegressor':
            return cost_fn(y_true, y) if cost_fn is not None else tf.losses.mean_squared_error(y_true, y)
        else:
            raise NotImplementedError('type {} not supported'.format(self.network_type))


class recurrent_neural_network(custom_estimator):
    def __init__(self, name, features, labels, primary_key=None,
                 hidden_units=[128, 128, 128, 128, 128, 128, 128, 128], learning_rate=.001, beta1=.9, beta2=.999, epsilon=1e-8, dropout=0.,
                 cost_fn=None, hook=custom_hook, cell_type=tf.contrib.rnn.LayerNormBasicLSTMCell, **kwargs):
        tf.reset_default_graph()
        self.name = name
        self.features = features
        self.labels = labels
        self.primary_key = primary_key
        self._hook = hook(self)
        self._initialize_layers(hidden_units, learning_rate, beta1, beta2, epsilon, dropout, cost_fn, cell_type)
        with tf.device('job:master'):
            self._saver = tf.train.Saver(max_to_keep=10)
        self._sess = tf.Session(server.target)
        self._merge = tf.summary.merge_all()
        if pathlib.Path('./.model/'+self.name+'/checkpoint').exists():
            self._saver.restore(self._sess, get_checkpoint(self.name))
        else:
            self._sess.run((tf.global_variables_initializer(), tf.tables_initializer()))
            pathlib.Path('./.model/'+self.name).mkdir(parents=True, exist_ok=True)
        self._writer = tf.summary.FileWriter('./.model/'+self.name, self._sess.graph)

    def train(self, batch_fn, steps):
        self._hook.begin_train(steps)
        for feature_batch, label_batch in batch_fn(self.features, self.labels, steps):
            feed_dict = {self._feature_placeholders[i['column']]: feature_batch[i['column']] for i in self.features}
            feed_dict[self._label_placeholder] = label_batch
            self._hook.begin_train_step()
            summary, _, cost = self._sess.run((self._merge, self._train_fn, self._cost_fn), feed_dict)
            self._writer.add_summary(summary, self.step)
            self._hook.after_train_step(cost)
        self._hook.after_train()

    def evaluate(self, batch_fn):
        accuracy = []
        self._hook.begin_evaluate()
        for feature_batch, label_batch in batch_fn(self.features, self.labels):
            feed_dict = {self._feature_placeholders[i['column']]: feature_batch[i['column']] for i in self.features}
            feed_dict[self._label_placeholder] = label_batch
            self._hook.begin_evaluate_step()
            summary, result = self._sess.run((self._merge, self._evaluate_fn), feed_dict)
            self._writer.add_summary(summary, self.step)
            self._hook.after_evaluate_step()
            accuracy.append(abs(result))
        accuracy = np.mean(accuracy)
        self._hook.after_evaluate(accuracy)
        return accuracy

    def predict(self, batch_fn, predict_length):
        for feature_batch, label_batch, primary_key in batch_fn(self.features, self.labels, self.primary_key):
            feed_dict = {self._feature_placeholders[i['column']]: feature_batch[i['column']] for i in self.features}
            pred_values = np.zeros((label_batch.shape[0], predict_length), np.float32)
            for idx in range(predict_length):
                feed_dict[self._label_placeholder] = np.pad(np.append(label_batch[:, idx:], pred_values[:, max(0, idx-len(label_batch[0])):idx], axis=1),
                                                            ((0, 0), (0, 1)), 'constant')
                summary, result = self._sess.run((self._merge, self._predict_fn), feed_dict)
                for i, j in enumerate(result):
                    pred_values[i, idx] = j[-1]
                self._writer.add_summary(summary, self.step)
            yield zip(primary_key, pred_values)

    def save(self, checkpoint=True):
        pathlib.Path('./.model/'+self.name).mkdir(parents=True, exist_ok=True)
        if checkpoint:
            self._saver.save(self._sess, './.model/{0}/{0}'.format(self.name), global_step=self.step)
        else:
            self._saver.save(self._sess, './.model/{0}/{0}'.format(self.name))

    def close(self):
        self._sess.close()

    @property
    def step(self):
        return self._sess.run(self._step)

    def _initialize_layers(self, hidden_units, learning_rate, beta1, beta2, epsilon, dropout, cost_fn, cell_type):
        self._feature_placeholders = {}
        feature_columns = []
        with tf.device('job:master'):
            for i in self.features:
                placeholder, column = self._generate_placeholder(i)
                self._feature_placeholders[i['column']] = placeholder
                feature_columns.append(column)
            feature_columns = tf.feature_column.input_layer(self._feature_placeholders, feature_columns) if self.features else None
            self._label_placeholder = tf.placeholder(tf.float32, shape=[None, None], name=self.labels['column'])
            label_column = self._label_placeholder
        y = self._generate_hidden_layers(feature_columns, label_column[:, :-1], hidden_units, dropout, cell_type)
        device = 'job:worker' if 'worker' in cluster.jobs else 'job:master'
        with tf.device(device):
            self._step = tf.Variable(0, name='global_step', trainable=False)
            self._evaluate_fn = self._get_evaluation(y, label_column[:, 1:])
            self._predict_fn = self._get_result(y)
            self._cost_fn = self._get_cost(y, label_column[:, 1:], cost_fn)
            self._train_fn = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(self._cost_fn, self._step, name='Train')
        tf.summary.histogram('cost', self._cost_fn)
        tf.summary.histogram('accuracy', self._evaluate_fn)

    def _generate_placeholder(self, info):
        if info['type'] == 'vocab':
            placeholder = tf.placeholder(tf.string, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(info['column'], info['vocab']))
        elif info['type'] == 'hash':
            placeholder = tf.placeholder(tf.string, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_hash_bucket(info['column'], info['num_buckets']))
        elif info['type'] == 'numeric':
            placeholder = tf.placeholder(tf.float32, shape=[None], name=info['column'])
            return placeholder, tf.feature_column.numeric_column(info['column'])
        else:
            raise NotImplementedError('type {} not supported'.format(info['type']))

    def _generate_hidden_layers(self, feature_columns, label_column, hidden_units, dropout, cell_type):
        label_column = tf.expand_dims(label_column, -1)
        device = 'job:slave' if 'slave' in cluster.jobs else 'job:master'
        with tf.device(device):
            result, state = tf.nn.dynamic_rnn(tf.contrib.rnn.OutputProjectionWrapper(
                            tf.nn.rnn_cell.MultiRNNCell([cell_type(num_units=i) for i in hidden_units]), output_size=1), label_column, dtype=label_column.dtype)
        return tf.squeeze(result, [2])

    def _get_evaluation(self, y, y_true):
        return tf.reduce_mean(y-y_true)

    def _get_result(self, y):
        return y

    def _get_cost(self, y, y_true, cost_fn):
        return cost_fn(y_true, y) if cost_fn is not None else tf.losses.mean_squared_error(y_true, y)


class ConsoleLogHook(custom_hook):
    def begin_train(self, *args):
        self.step = 0
        self.num_steps = args[0]
        self.cost = 0.
        self.loading_bar = JimoAPI.core.JimoLoadingBar(pending=self.num_steps)
        self.loading_bar.prefix = 'training |'
        self.loading_bar.postfix = '|{_processed:5} /{_pending:5}, {percentage:7.2%}, cost: {cost:10.4f}'
        self.loading_bar.phases = ' ▏▎▍▌▋▊▉█'
        self.pending = self.estimator.step + self.num_steps

    def begin_train_step(self, *args):
        self.step += 1
        self.loading_bar.update(self.step, postfix_data={'_processed': self.estimator.step, '_pending': self.pending, 'cost': self.cost})

    def after_train_step(self, *args):
        self.loading_bar.update(self.step, postfix_data={'_processed': self.estimator.step, '_pending': self.pending, 'cost': args[0]})
        self.cost = args[0]

    def after_train(self, *args):
        self.step += 1
        self.loading_bar.update(self.step, postfix_data={'_processed': self.estimator.step, '_pending': self.pending, 'cost': self.cost})
        self.estimator.save()
        print()

    def begin_evaluate(self, *args):
        print('\033[1mbegin evaluation...\033[0m', end='', flush=True)

    def after_evaluate(self, *args):
        print('\r\033[1maccuracy: \033[93m{}\033[0m        '.format(args[0]))


def create_estimator_from_name(instance_name):
    with JimoAPI.TempCursor() as cx:
        cx.execute("SELECT info FROM instance_info WHERE name='{}';", instance_name)
        try:
            data = json.loads(list(cx.fetchone().values())[0])
        except TypeError:
            return None, None, False
    _type = data['type']
    if _type[0] == 'D':
        estimator = deep_neural_network(instance_name, network_type=_type, **data, hook=ConsoleLogHook)
    elif _type[0] == 'R':
        estimator = recurrent_neural_network(instance_name, **data, hook=ConsoleLogHook)
    else:
        raise NotImplementedError('type {} not supported'.format(_type))
    return estimator, data, True


def change_checkpoint(instance_name, checkpoint_path):
    with open('./.model/'+instance_name+'/checkpoint') as f:
        lines = f.readlines()
    lines[0] = 'model_checkpoint_path: "' + os.path.basename(checkpoint_path) + '"\n'
    with open('./.model/'+instance_name+'/checkpoint', 'w') as f:
        f.writelines(lines)


def get_checkpoint(instance_name):
    with open('./.model/'+instance_name+'/checkpoint') as f:
        return './.model/'+instance_name+'/'+f.readlines()[0].split('"')[1]


_setup_server()
