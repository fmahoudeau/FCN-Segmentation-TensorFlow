# Copyright 2019 Florent Mahoudeau. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import argparse
import os.path
import itertools
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.framework import ops

from image_utils import *
from pascal_voc_dataset import PascalVOC2012Dataset
from kitty_road_dataset import KittyRoadDataset
from cam_vid_dataset import CamVidDataset
from pascal_plus_dataset import PascalPlusDataset
import fcn_model


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='pascal_voc2012', help='The name of the dataset.')
parser.add_argument('--data_dir', type=str, default='/tmp/pascal_voc_data/',
                    help='Directory where the data is located.')
parser.add_argument('--vgg16_dir', type=str, default='/tmp/vgg16/',
                    help='Directory where the VGG16 pre-trained weights are located.')
parser.add_argument('--save_dir', type=str, default='/tmp/saved_models/',
                    help='Directory where to save model checkpoints and summary data.')
parser.add_argument('--fcn_version', type=str, default='fcn8',
                    help='The FCN version to train, one of fcn8, fcn16 or fcn32.')
parser.add_argument('--model_name', type=str, default='trial_model',
                    help='The name of your model. Used to save checkpoints of model variables and summary data.')
parser.add_argument('--dropout_rate', type=float, default=0.5, help='The dropout rate of VGG16 layers.')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--n_classes', type=int, default=21, help='Number of classes, excluding any void/ignore class.')
parser.add_argument('--batch_size', type=int, default=12, help='Batch size')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='Optimizer weight decay')
parser.add_argument('--image_height', type=int, default=512,
                    help=('Image input height of FCN model. Must be a multiple of 32.\n'
                          'When an input image is smaller than this height, it is center-padded.\n'
                          'When an input image is taller than this height, it is re-sized with interpolation.'))
parser.add_argument('--image_width', type=int, default=512,
                    help=('Image input width of FCN model. Must be a multiple of 32.\n'
                          'When an input image is narrower than this width, it is center-padded.\n'
                          'When an input image is wider than this width, it is re-sized with interpolation.'))


def save_learning_curve(model_name,
                        epoch, metrics,
                        training_loss,
                        training_metrics,
                        validation_loss=None,
                        validation_metrics=None):
    colors = ['r', 'b', 'g', 'm']  # The first color in the list is for the loss. The others are for the metrics.

    # Save a plot of the loss and accuracy
    plt.clf()
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    ax1.plot(training_loss[1:], colors[0], label="Training loss")
    for i, metric in enumerate(metrics.keys()):
        ax2.plot(training_metrics[metric][1:], colors[i + 1], label='Training {}'.format(metric))
    if validation_loss is not None and validation_metrics is not None:
        ax1.plot(validation_loss[1:], '{}--'.format(colors[0]), label="Validation loss")
        for i, metric in enumerate(metrics.keys()):
            ax2.plot(validation_metrics[metric][1:], '{}--'.format(colors[i + 1]), label='Validation {}'.format(metric))
    plt.title('Learning curve - {}\n{} with lr={} and weight_decay={}'.format(model_name, FLAGS.optimizer,
                                                                              FLAGS.learning_rate, FLAGS.weight_decay))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=colors[0])
    ax2.set_ylabel('Score')
    ax1.legend(loc="lower center")
    ax2.legend(loc="center")
    plt.savefig('{}_training_history_{}.png'.format(os.path.join(FLAGS.save_dir, model_name), epoch + 1));

    # Save loss and accuracy in a DataFrame
    df = pd.DataFrame()
    df['train_loss'] = training_loss
    for metric in list(metrics.keys()):
        df['train_' + metric] = training_metrics[metric]
    if validation_loss is not None and validation_metrics is not None:
        df['val_loss'] = validation_loss
        for metric in list(metrics.keys()):
            df['val_' + metric] = validation_metrics[metric]
    df.to_csv('{}_{}.csv'.format(os.path.join(FLAGS.save_dir, model_name), epoch + 1))


def clear_session():
    ops.reset_default_graph()
    if tf.get_default_session() is not None:
        tf.get_default_session().close()
    sess = tf.InteractiveSession(config=tf.ConfigProto(intra_op_parallelism_threads=4))
    sess.as_default()
    return sess


def learning_rate_with_exp_decay(batch_size, n_images, decay_epochs, decay_rate=0.95, staircase=False, base_lr=1e-5):
    """

    :param batch_size: The batch size.
    :param n_images: The number of images in the training set.
    :param decay_epochs: The number of epochs after which the learning rate is decayed by `decay_rate`.
    :param decay_rate: The amount of decay applied after every `decay_epochs`. Defaults to 0.95.
    :param staircase: Defaults to False
    :param base_lr: The default starting learning rate. Default to 1e-5

    :return: The learning rate function
    """

    global_step = tf.Variable(0, name='global_step', trainable=False)
    n_batches = int(n_images / batch_size)

    if decay_rate > 0:
        print('Applying exponential decay every {} epoch(s)'.format(decay_epochs))
    else:
        print('Applying fixed learning rate of {}'.format(base_lr))

    def learning_rate_fn():
        if decay_rate > 0:
            lr = tf.train.exponential_decay(base_lr, global_step, n_batches * decay_epochs,
                                            decay_rate, staircase=staircase, name='exp_decay')
        else:
            lr = tf.constant(base_lr, name='constant_lr')
        return lr

    return learning_rate_fn


def compile_model(model, n_classes, metrics, learning_rate_fn, ignore_label=True, **kwargs):
    """Shared functionality for different FCN model_fns.

    :param model:
    :param n_classes:
    :param metrics:
    :param learning_rate_fn:
    :param ignore_label: Append a void label, defaults to True.
    :param kwargs:
    :return:
    """

    # Reshape 4D tensor (batch, row, column, channel) to 2D, each row represents a pixel, each column a class
    logits = tf.reshape(model.outputs, (-1, n_classes + 1), name="logits")
    # Vector of ground truth labels
    labels = tf.stop_gradient(tf.reshape(model.labels, (-1,)))

    if FLAGS.debug:
        print('logits = {}'.format(str(logits.shape)))
        print('labels = {}'.format(str(model.labels.shape)))

    # Mask pixels to be ignored for datasets with a void class.
    # The void class is assumed to be last label in the color map
    if ignore_label:
        # Pixel mask vector, 1=keep, 0=ignore. Vector dim = batch_size * num_pixels
        ignore_pixels = tf.stop_gradient(tf.subtract(tf.ones((tf.shape(logits)[0],)),
                                                     tf.cast(tf.reshape(tf.equal(model.labels,
                                                                                 tf.constant(n_classes,
                                                                                             dtype=tf.float32)),
                                                                        [-1]),
                                                             dtype=tf.float32)))

        loss = tf.losses.compute_weighted_loss(
            weights=ignore_pixels,
            losses=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32),
                                                                  logits=logits, name="cross_entropy"))
    # Calculate mean distance from actual labels using cross entropy
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32),
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy")

    learning_rate = learning_rate_fn()

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')

    # Create optimizer
    if FLAGS.optimizer.lower() == 'adam':
        if 'weight_decay' in kwargs.keys():
            if kwargs['weight_decay'] > 0:
                opt = tf.contrib.opt.AdamWOptimizer(kwargs['weight_decay'],
                                                    learning_rate=learning_rate)
                if FLAGS.debug:
                    print('Adam optimizer with weight decay = {}'.format(kwargs['weight_decay']))
            else:
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
                if FLAGS.debug:
                    print('Adam optimizer w/o weight decay')
        else:
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            if FLAGS.debug:
                print('Adam optimizer w/o weight decay')
    elif FLAGS.optimizer.lower() == 'sgd':
        momentum = kwargs['momentum'] if 'momentum' in kwargs.keys() else 0.0
        decay = 0.0  # For future use
        nesterov = kwargs['nesterov'] if 'nesterov' in kwargs.keys() else False

        if 'weight_decay' in kwargs.keys():
            if kwargs['weight_decay'] > 0:
                opt = tf.contrib.opt.MomentumWOptimizer(kwargs['weight_decay'], learning_rate=learning_rate,
                                                        momentum=momentum, use_nesterov=nesterov)
                if FLAGS.debug:
                    print('SGD optimizer: weight_decay={}, momentum={}, nestorov={}'.format(
                        kwargs['weight_decay'], momentum, nesterov))
            else:
                opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                 momentum=momentum,
                                                 use_nesterov=nesterov)
                if FLAGS.debug:
                    print('SGD optimizer: momentum={}, nestorov={}'.format(momentum, nesterov))
        else:
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=momentum,
                                             use_nesterov=nesterov)
            if FLAGS.debug:
                print('SGD optimizer: momentum={}, nestorov={}'.format(momentum, nesterov))
    else:
        raise ValueError('{} is an unrecognized optimizer'.format(FLAGS.optimizer))

    """if DOUBLE_BIAS_LR:
        grads_and_vars = opt.compute_gradients(loss)
        grads_and_vars_mult = []
        for grad, var in grads_and_vars:
            if var.op.name in ['fcn8/bilinear_filter', 'fcn8/conv2d_transpose/kernel']:
                grad *= 2
            grads_and_vars_mult.append((grad, var))
        train_op = opt.apply_gradients(grads_and_vars_mult, global_step=global_step, name="train_op")
    else:"""
    train_op = opt.minimize(loss, global_step=tf.train.get_global_step(), name="train_op")

    # Create metrics
    tf_metrics = {}
    predictions = tf.argmax(logits, 1, name="predictions")  # The vector of predicted labels
    if 'acc' in metrics:  # Pixel accuracy
        tf_metrics['acc'] = tf.metrics.accuracy(labels, predictions,
                                                weights=ignore_pixels if ignore_label else None, name='acc')
    if 'mean_acc' in metrics:  # Mean class pixel accuracy
        tf_metrics['mean_acc'] = tf.metrics.mean_per_class_accuracy(labels, predictions, n_classes+1,
                                                                    weights=ignore_pixels if ignore_label else None,
                                                                    name='mean_acc')
    if 'mean_iou' in metrics:  # Mean Intersection-over-Union (mIoU)
        tf_metrics['mean_iou'] = tf.metrics.mean_iou(labels, predictions, n_classes+1,
                                                     weights=ignore_pixels if ignore_label else None,
                                                     name='mean_iou')

    sess = tf.get_default_session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print("Model build successful")
    model.logits = logits
    model.opt = opt
    model.train_op = train_op
    model.loss = loss
    model.metrics = tf_metrics


def fit_model(model, epochs, batch_size, dataset_train, dataset_val, model_name, initial_epoch=0):
    sess = tf.get_default_session()
    training_loss, validation_loss = [], []
    training_metrics, validation_metrics = {}, {}
    for metric in list(model.metrics.keys()):
        training_metrics[metric] = []
        validation_metrics[metric] = []

    best_loss = 0.2  # 0.05 for KittyRoad & CamVid

    # Isolate the variables stored behind the scenes by the metrics operation
    metrics_vars = []
    for metric in list(model.metrics.keys()):
        metrics_vars.append(tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope=metric))
        training_metrics[metric] = []
        validation_metrics[metric] = []
    metrics_vars = list(itertools.chain.from_iterable(metrics_vars))  # Flatten list of lists

    # Define initializer to initialize/reset metrics variable(s)
    metrics_vars_initializer = tf.variables_initializer(var_list=metrics_vars)

    # A reinitializable iterator is defined by its structure. We could use the
    # `output_types` and `output_shapes` properties of either `training_dataset`
    # or `validation_dataset` here, because they are compatible.
    iterator = tf.data.Iterator.from_structure(dataset_train.output_types,
                                               dataset_train.output_shapes)
    next_batch = iterator.get_next()

    train_init_op = iterator.make_initializer(dataset_train)
    val_init_op = iterator.make_initializer(dataset_val)

    print(datetime.now().isoformat())

    for epoch in range(initial_epoch, epochs):
        print("Epoch {}/{}, LR={}".format(epoch + 1, epochs, sess.run(model.opt._lr)))
        training_loss.append(0)
        n_batches = 0

        # Initialize an iterator over the training dataset
        sess.run(train_init_op)

        # Train the model
        while True:
            try:
                im_batch, gt_batch = sess.run(next_batch)
                if len(im_batch) < batch_size:
                    continue
                res = sess.run({**{"loss": [model.loss, model.train_op]}, **model.metrics},
                               feed_dict={model.inputs: im_batch,
                                          model.labels: gt_batch,
                                          model.keep_prob: FLAGS.dropout_rate})
                training_loss[-1] += res["loss"][0]
                n_batches += 1
                # print('Batch {} = {:.3f}'.format(n_batches, res["loss"][0]))
            except tf.errors.OutOfRangeError:
                break

        # Update training loss and metrics
        training_loss[-1] /= n_batches
        current_loss = training_loss[-1]
        message = 'loss = {:.3f}'.format(training_loss[-1])
        for metric in list(model.metrics.keys()):
            # Remove the void/ignore class accuracy in the mean calculation because it's value is 0
            if metric == 'mean_acc':
                val = np.mean(res[metric][1][:model.n_classes])
            # Remove the void/ignore class IoU in the mean calculation because it's value is NaN
            elif metric == 'mean_iou':
                mat = res[metric][1][:model.n_classes, :model.n_classes]
                val = np.mean((np.diag(mat) / (mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat))))
            # No need to adjust other metrics.
            else:
                val = res[metric][0]
            training_metrics[metric].append(val)
            message += ', {} = {:.3f}'.format(metric, val)

        # Reset the metrics variables
        sess.run(metrics_vars_initializer)

        # Evaluate the model if a validation dataset is provided
        if dataset_val is not None:
            validation_loss.append(0)
            n_batches = 0

            # Initialize an iterator over the validation dataset
            sess.run(val_init_op)

            # Train the model
            while True:
                try:
                    im_batch, gt_batch = sess.run(next_batch)
                    if len(im_batch) < batch_size:
                        continue
                    res = sess.run({**{"loss": model.loss}, **model.metrics},
                                   feed_dict={model.inputs: im_batch,
                                              model.labels: gt_batch,
                                              model.keep_prob: 1.0})
                    validation_loss[-1] += res["loss"]
                    n_batches += 1
                except tf.errors.OutOfRangeError:
                    break

            # Save validation metrics results
            validation_loss[-1] /= n_batches
            message += ', val_loss = {:.3f}'.format(validation_loss[-1])
            for metric in list(model.metrics.keys()):
                # Remove the void/ignore class accuracy in the mean calculation because it's value is 0
                if metric == 'mean_acc':
                    val = np.mean(res[metric][1][:model.n_classes])
                # Remove the void/ignore class IoU in the mean calculation because it's value is NaN
                elif metric == 'mean_iou':
                    mat = res[metric][1][:model.n_classes, :model.n_classes]
                    val = np.mean((np.diag(mat) / (mat.sum(axis=0) + mat.sum(axis=1) - np.diag(mat))))
                # No need to adjust other metrics.
                else:
                    val = res[metric][0]
                validation_metrics[metric].append(val)
                message += ', val_{} = {:.3f}'.format(metric, val)

            # Reset the metrics variables
            sess.run(metrics_vars_initializer)

        print(message + '\n' + datetime.now().isoformat())

        model_saved = False
        # Checkpoint every 25 epochs and after the final epoch
        if ((epoch+1) % 25 == 0) | ((epoch+1) == epochs):
            print("Saving learning curve")
            if dataset_val is None:
                save_learning_curve(model_name, epoch, model.metrics,
                                    training_loss, training_metrics)
            else:
                save_learning_curve(model_name, epoch, model.metrics,
                                    training_loss, training_metrics,
                                    validation_loss, validation_metrics)
            print("Saving model checkpoint")
            model.save_variables(os.path.join(FLAGS.save_dir, model_name), global_step=epoch+1)
            model_saved = True

        # Save the best model
        if current_loss <= best_loss:
            if not model_saved:
                print("Saving model checkpoint")
                model.save_variables(os.path.join(FLAGS.save_dir, model_name), global_step=epoch+1)
            best_loss = current_loss
        print()


def staged_training():
    pass


def oneoff_training(fcn_mode, dataset_name, dataset_path, metrics, model_name, saved_variables=None):
    if fcn_mode not in ('FCN32', 'FCN16', 'FCN8'):
        raise ValueError('{} is an invalid model'.format(fcn_mode))

    print('One-off {} end-to-end training using {}'.format(fcn_mode, dataset_name))

    sess = clear_session()
    if dataset_name == 'kitty_road':
        dataset = KittyRoadDataset(FLAGS.augmentation_params)
    elif dataset_name == 'cam_vid':
        dataset = CamVidDataset(FLAGS.augmentation_params)
    elif dataset_name == 'pascal_voc_2012':
        dataset = PascalVOC2012Dataset(FLAGS.augmentation_params)
    elif dataset_name == 'pascal_plus':
        dataset = PascalPlusDataset(FLAGS.augmentation_params)
    else:
        raise ValueError('{} is an invalid dataset'.format(dataset_name))

    dataset_train = dataset.load_dataset(is_training=True, data_dir=dataset_path, batch_size=FLAGS.batch_size)
    dataset_val = dataset.load_dataset(is_training=False, data_dir=dataset_path, batch_size=FLAGS.batch_size)

    model = fcn_model.Model(dataset.image_shape, dataset.n_classes, FLAGS.vgg16_weights_path)
    saved_variables = None if saved_variables is None else os.path.join(FLAGS.save_dir, saved_variables)
    model(fcn_mode, saved_variables=saved_variables)
    learning_rate_fn = learning_rate_with_exp_decay(FLAGS.batch_size, dataset.n_images['train'], 10,
                                                    decay_rate=0, base_lr=FLAGS.learning_rate)
    compile_model(model, dataset.n_classes, metrics, learning_rate_fn, weight_decay=FLAGS.weight_decay)

    fit_model(model, FLAGS.n_epochs, FLAGS.batch_size, dataset_train, dataset_val, model_name)
    print("Total steps = {}".format(tf.train.global_step(sess, tf.train.get_global_step())))
    return model, (dataset_train, dataset_val)


def main(_):
    """
    For future use.
    """
    pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.image_shape = (FLAGS.image_height, FLAGS.image_width)
    FLAGS.metrics = ['acc', 'mean_acc', 'mean_iou']
    tf.app.run(argv=[sys.argv[0]] + unparsed)
