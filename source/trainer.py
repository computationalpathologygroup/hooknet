from random import randint
import sys
import time
import argparse
import random

import numpy as np
import tensorflow as tf

from .model import HookNet

class Trainer:
    """
    Trainer class

    this class trains a keras model

    """

    def __init__(self,
                 epochs,
                 steps,
                 batch_size,
                 model,
                 batch_generator,
                 seed):

        np.random.seed(seed)
        random.seed(seed)

        self._epochs = epochs
        self._steps = steps

        self._batch_generator = batch_generator
        self._batch_functions = self._set_batch_functions()
        self._model = model
        self._model_functions = self._set_model_functions()

        self._graph = tf.get_default_graph()

    def train(self):
        print('start training')
        with self._graph.as_default():
            for epoch in range(self._epochs):
                print(f'epoch: {epoch}')
                for state in ['training','validation']:
                    print(f'{state}....')
                    avg_loss = self._on_epoch(epoch, self._model_functions[state], self._batch_functions[state])
                    print(f'{state} loss: {avg_loss}')

    def _set_model_functions(self):
        return {'training': self._model.train_on_batch,
                'validation': self._model.test_on_batch}

    def _set_batch_functions(self):
        return {'training': self._batch_generator.get_training_batch,
                'validation': self._batch_generator.get_validation_batch}

    def _on_epoch(self, epoch, model_function, batch_function):
        epoch_losses = []
        for step in range(self._steps):
            X_batch, y_batch = batch_function()
            computed_metrics = model_function(x=X_batch, y=y_batch)
            epoch_losses.append(computed_metrics[0])
        return np.mean(epoch_losses)



