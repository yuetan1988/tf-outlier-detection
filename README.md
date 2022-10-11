[license-image]: https://img.shields.io/badge/License-Apache%202.0-blue.svg
[license-url]: https://opensource.org/licenses/Apache-2.0
[pypi-image]: https://badge.fury.io/py/tfts.svg
[pypi-url]: https://pypi.python.org/pypi/tfts
[build-image]: https://github.com/LongxingTan/python-lekin/actions/workflows/test.yml/badge.svg?branch=master
[build-url]: https://github.com/LongxingTan/python-lekin/actions/workflows/test.yml?query=branch%3Amaster
[lint-image]: https://github.com/LongxingTan/python-lekin/actions/workflows/lint.yml/badge.svg
[lint-url]: https://github.com/LongxingTan/python-lekin/actions/workflows/lint.yml
[docs-image]: https://readthedocs.org/projects/python-lekin/badge/?version=latest
[docs-url]: https://python-lekin.readthedocs.io/en/latest/

<h1 align="center">
<img src="./docs/source/_static/logo.svg" width="490" align=center/>
</h1><br>

-------------------------------------------------------------------------

[![LICENSE][license-image]][license-url]
[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Lint Status][lint-image]][lint-url]
[![Docs Status][docs-image]][docs-url]

**[Documentation](https://tf-outlier.readthedocs.io)** | **[Tutorials](https://tf-outlier.readthedocs.io/en/latest/tutorials.html)** | **[Release Notes](https://tf-outlier.readthedocs.io/en/latest/CHANGELOG.html)** | **[中文](./README_CN.md)**

**tfod** is a python package for outlier detection, supporting the common deep learning methods in TensorFlow.

## Tutorial

Install

```bash
pip install tensorflow
pip isntall tfoutlier
```

Usage

```python
import tensorflow as tf
import tfod
from tfod import AutoModel

data = tfod.load('ecg')
model = AutoModel('lstm').from_checkpoint('./')
model.detect(data, plot=True)
```

## Examples
