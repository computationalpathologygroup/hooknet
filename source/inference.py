import time

from threading import Thread
from collections import deque

import numpy as np
import tensorflow as tf
from multiprocessing import Queue
from .image.deamons import WSIReaderDeamon, WSIWriterDeamon


def normalize(input):
    _type = type(input)
    if _type == np.ndarray:
        return input / 255.0
    return _type(np.array(input) / 255.0)


class Inference:
    """
    Inference class
    """
    def __init__(self,
                 wsi_path,
                 mask_path,
                 output_path,
                 input_shape,
                 output_shape,
                 resolutions,
                 batch_size,
                 cpus,
                 queue_size,
                 model_instance,
                 multi_loss=False):
        
        self._wsi_path = wsi_path
        self._mask_path = mask_path
        self._output_path = output_path
        self._input_shape = input_shape
        self._output_shape = output_shape
        self._tile_size = output_shape[0] - (output_shape[0] % 64)
        self._resolutions = resolutions
        self._cpus = cpus
        self._queue_size = queue_size
        self._multi_loss = multi_loss

        self._batch_size = batch_size

        self._model_instance = model_instance
        self._graph = tf.get_default_graph()

        # queues
        self._reader_queue = Queue(maxsize=self._queue_size)
        self._writer_queue = None

        # Create batch deamon proces
        self._readerdeamon = WSIReaderDeamon(self._wsi_path,
                                             self._mask_path,
                                             self._batch_size,
                                             self._input_shape,
                                             self._output_shape,
                                             self._tile_size,
                                             self._resolutions,
                                             self._queue_size,
                                             self._cpus,
                                             self._reader_queue)
        # Create batch deamon process
        self._writerdeamon = WSIWriterDeamon(self._wsi_path,
                                             self._output_path,
                                             self._resolutions[0],
                                             self._output_shape,
                                             self._tile_size,
                                             self._writer_queue) 

    def _post_process(self, predictions):
        return [np.argmax(prediction, -1).astype('uint8')+1 for prediction in predictions]

    def _test_on_wsi(self):
        t1_read = time.time()
        for data in iter(self._reader_queue.get, 'STOP'):
            t2_read = time.time()
            X_batch, items = data 
            X_batch = normalize(X_batch)
            pred = self._model_instance.predict_on_batch(x=X_batch)
            
            if self._multi_loss: 
                predictions = self._post_process(pred[0])
            else:
                predictions = self._post_process(pred)
                
            self._writerdeamon.put((predictions, items))
            t1_read = time.time()
        # self._writer_queue.put(('STOP'))

    def start(self):
        # start reader
        self._readerdeamon.start()
        with self._graph.as_default():
            self._test_on_wsi()
        self.stop()

    def stop(self):
        self._readerdeamon.join()
        self._writerdeamon.stop()