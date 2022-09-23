**[Documentation](https://tf-outlier.readthedocs.io)** | **[Tutorials](https://tf-outlier.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://tf-outlier.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](./README_CN.md)**

**tfoutlier** is a python package for outlier detection task, supporting the common deep learning methods on TensorFlow.

## Tutorial

install
```bash
pip install tensorflow
pip isntall tfoutlier
```

usage
```python
import tensorflow as tf
import tfoutlier as tfod

data = tfod.load('ecg')
model = AutoModel('lstm').from_checkpoint('./')
model.detect(data, plot=True)

```

## Examples
