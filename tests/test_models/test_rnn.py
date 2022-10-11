import unittest
import tensorflow as tf
from tfod.models.rnn import RNN


class RNNTest(unittest.TestCase):
    def test_model(self):
        model = RNN()
        x = tf.random.normal([2, 10, 1])

        y = model(x)
        self.assertEqual(y.shape, (2, 3), "incorrect output shape")

    def test_train(self):
        pass


if __name__ == "__main__":
    unittest.main()
