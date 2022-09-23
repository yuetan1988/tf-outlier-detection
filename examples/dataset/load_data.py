
import numpy as np
import tensorflow as tf


class DataReader(object):
    def __init__(self, data, train_sequence_length=10, predict_sequence_length=3) -> None:
        self.data = data
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        train_begin_idx = idx - self.train_sequence_length
        test_end_idx = idx + self.predict_sequence_length
        features = self.data[train_begin_idx: idx]
        target = self.data[idx: test_end_idx]
        return features, target

    def iter(self):
        for i in range(self.train_sequence_length, len(self)-self.predict_sequence_length):
            yield self[i]


class DataLoader(object):
    def __init__(self, data_reader):
        self.data_reader = data_reader
        self.train_sequence_length = data_reader.train_sequence_length
        self.predict_sequence_length = data_reader.predict_sequence_length

    def __call__(self, batch_size, shuffle=False, drop_remainder=False): 
        dataset = tf.data.Dataset.from_generator(
            self.data_reader.iter,
            output_types=((tf.float32, tf.float32)),
            # output_signature=({'inputs': 
            # (tf.TensorSpec(shape=(), dtype=tf.int32), 
            # tf.TensorSpec(shape=(self.train_sequence_length, self.short_feature_size), dtype=tf.float32),
            # tf.TensorSpec(shape=(self.train_sequence_length+self.predict_sequence_length, self.long_feature_size), dtype=tf.float32)), 
            # 'teacher': tf.TensorSpec(shape=(self.predict_sequence_length//self.target_aggs, 1), dtype=tf.float32)}, 
            # tf.TensorSpec(shape=(self.predict_sequence_length//self.target_aggs, self.target_column_size ), dtype=tf.float32)
            # )
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


if __name__ == '__main__':
    from examples.dataset.read_ecg import load_ecg
    train, valid= load_ecg('ecg')
    train_data = DataReader(train)
    valid_data = DataReader(valid)

