import tensorflow as tf
import json
import time
import os

import JimoAPI.connector
import JimoAPI.tf

tf.logging.set_verbosity(tf.logging.ERROR)

module_name = [("create dnn classifier {model_name}", "create_dnn_classifier"),
               ("create dnn regressor {model_name}", "create_dnn_regressor"),
               ("create rnn regressor {model_name}", "create_rnn_regressor"),
               ("delete model {model_name}", "delete_model"),
               ("create instance {instance_name} using template {model_name} with table {table_name}", "create_instance"),
               ("delete instance {instance_name}", "delete_instance"),
               ("train model {instance_name} with {num_steps} steps", "train_model"),
               ("train model {instance_name}", "train_model_auto"),
               ("evaluate model {instance_name}", "evaluate_model"),
               ("predict using model {instance_name} into table {table_name}", "predict_model"),
               ("save instance {instance_name}", "save_instance"),
               ("load instance {instance_name}", "load_instance")]


def create_dnn_classifier(*args, **kwargs):
    kwargs['type'] = 'DNNClassifier'
    return 'DNNClassifier created' if _create_model(kwargs.pop('model_name'), kwargs) else 'DNNClassifier exists'


def create_dnn_regressor(*args, **kwargs):
    kwargs['type'] = 'DNNRegressor'
    return 'DNNRegressor created' if _create_model(kwargs.pop('model_name'), kwargs) else 'DNNRegressor exists'


def create_rnn_regressor(*args, **kwargs):
    kwargs['type'] = 'RNNRegressor'
    return 'DNNRegressor created' if _create_model(kwargs.pop('model_name'), kwargs) else 'DNNRegressor exists'


def delete_model(*args, **kwargs):
    with JimoAPI.TempCursor() as cx:
        try:
            cx.execute("DELETE FROM model_info WHERE name='{}';", kwargs['model_name'])
        except JimoAPI.connector.OperationalError:
            pass
    return 'Model deleted'


def create_instance(*args, **kwargs):
    with JimoAPI.TempCursor() as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS instance_info (name string, info string, filename string);")
        cx.execute("SELECT name FROM instance_info WHERE name='{}';", kwargs['instance_name'])
        if cx.fetchone() is None:
            cx.execute("SELECT data FROM model_info WHERE name='{}';", kwargs['model_name'])
            try:
                data = json.loads(list(cx.fetchone().values())[0])
            except TypeError:
                return "Template {} does not exist".format(kwargs['model_name'])
            data.update(kwargs)
            cx.execute("INSERT INTO instance_info ('name', 'info') VALUES ('{}', '{}');", (kwargs['instance_name'], json.dumps(data)))
            return 'Instance created'
        return 'Instance exists'


def delete_instance(*args, **kwargs):
    with JimoAPI.TempCursor() as cx:
        try:
            cx.execute("DELETE FROM instance_info WHERE name='{}';", kwargs['instance_name'])
        except JimoAPI.connector.OperationalError:
            pass
    os.system("rm -rf ./.model/{}".format(kwargs['instance_name']))
    return 'Instance deleted'


def train_model(*args, **kwargs):
    estimator, data, flag = JimoAPI.tf.create_estimator_from_name(kwargs['instance_name'])
    if not flag:
        return 'instance {} does not exist'.format(kwargs['instance_name'])
    estimator.train(JimoAPI.tf.input_fn(data, 'train'), kwargs['num_steps'])
    estimator.close()
    return 'trained with {} steps'.format(kwargs['num_steps'])


def evaluate_model(*args, **kwargs):
    estimator, data, flag = JimoAPI.tf.create_estimator_from_name(kwargs['instance_name'])
    if not flag:
        return 'instance {} does not exist'.format(kwargs['instance_name'])
    result = estimator.evaluate(JimoAPI.tf.input_fn(data, 'evaluate'))
    estimator.close()
    return 'accuracy: {}'.format(result)


def predict_model(*args, **kwargs):
    estimator, data, flag = JimoAPI.tf.create_estimator_from_name(kwargs['instance_name'])
    if not flag:
        return 'instance {} does not exist'.format(kwargs['instance_name'])
    result = estimator.predict(JimoAPI.tf.input_fn(data, 'predict'))
    for i in result:
        for j in i:
            print(j)
    estimator.close()
    return 'Done predicting'


def save_instance(*args, **kwargs):
    with JimoAPI.TempCursor() as cx:
        cx.execute("UPDATE instance_info SET filename='{}' WHERE name='{}';", ("Temp", kwargs['instance_name']))
    return 'Instance {} saved'.format(kwargs['instance_name'])


def load_instance(*args, **kwargs):
    return 'Instance {} loaded'.format(kwargs['instance_name'])


def train_model_auto(*args, **kwargs):
    estimator, data, flag = JimoAPI.tf.create_estimator_from_name(kwargs['instance_name'])
    if not flag:
        return 'instance {} does not exist'.format(kwargs['instance_name'])
    start_time = time.time()
    try:
        result = estimator.evaluate(JimoAPI.tf.input_fn(data, 'evaluate'))
        best_instance = [result, tf.train.latest_checkpoint('./.model/'+kwargs['instance_name']), 1]
    except ValueError:
        best_instance = None
    while True:
        estimator.train(JimoAPI.tf.input_fn(data, 'train'), 500)
        result = estimator.evaluate(JimoAPI.tf.input_fn(data, 'evaluate'))
        if best_instance is None or (result > best_instance[0] if estimator.network_type == 'DNNClassifier' else abs(result) < abs(best_instance[0])):
            best_instance = [result, tf.train.latest_checkpoint('./.model/'+kwargs['instance_name']), 1]
        else:
            best_instance[2] += 1
            if best_instance[2] == 10:
                JimoAPI.tf.change_checkpoint(kwargs['instance_name'], best_instance[1])
                break
        if (time.time() - start_time) > 600:
            break
    estimator.close()
    return 'stopped with accuracy: {}'.format(best_instance[0])


def create_multiple_instance(*args, **kwargs):
    pass


def _create_model(model_name, kwargs):
    with JimoAPI.TempCursor() as cx:
        cx.execute("CREATE TABLE IF NOT EXISTS model_info (name string, data string);")
        cx.execute("SELECT name FROM model_info WHERE name='{}';", model_name)
        if cx.fetchone() is None:
            cx.execute("INSERT INTO model_info VALUES ('{}', '{}');", (model_name, json.dumps(kwargs)))
            return True
        return False
