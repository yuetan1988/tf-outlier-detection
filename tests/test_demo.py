"""
python -m unittest -v tests/test_demo.py
"""

import unittest
import functools
import tensorflow as tf
from tensorflow.keras.layers import Input
from tfod import AutoModel, Trainer
import tfod


def build_model():
    return


class DemoTest(unittest.TestCase):
    def test_demo(self):
        data = tfod.load_data()

        model = AutoModel('lstm')
        model.detect(data, './', plot=True)


