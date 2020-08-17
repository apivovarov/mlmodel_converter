import logging
import pytest
from pytest import approx
import tempfile
import sys
import coremltools as ct
from neo_mlmodel_converter import *


def test_get_class_labels():
  imagenet_labels_1000_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'imagenet_labels_1000.txt',
  )
  compiler_options={"class_labels": imagenet_labels_1000_path}
  labels = get_class_labels(compiler_options)
  assert labels is not None and len(labels) == 1000
  assert labels[0] == "tench"
  assert labels[999] == "toilet tissue"


def test_get_shape():
  input_config = {"shape": [1,224,224,3]}
  ct_shape = get_shape(input_config)
  assert isinstance(ct_shape, ct.Shape)
  assert ct_shape.shape == (1,224,224,3)

  input_config = {"shape": ["1..10",224,224,3], "default_shape": [1,224,224,3]}
  ct_shape = get_shape(input_config)
  assert isinstance(ct_shape, ct.Shape)
  assert ct_shape.shape[1:] == (224,224,3)
  assert ct_shape.shape[0].lower_bound, ct_shape.shape[0].upper_bound == (1, 10)
  assert ct_shape.default == (1,224,224,3)

  input_config = {"shape": [[1,224,224,3], [1,448,448,3]], "default_shape": [1,224,224,3]}
  ct_shape = get_shape(input_config)
  assert isinstance(ct_shape, ct.EnumeratedShapes)
  assert isinstance(ct_shape.shapes[0], ct.Shape)
  assert ct_shape.shapes[0].shape == (1,224,224,3)
  assert ct_shape.shapes[1].shape == (1,448,448,3)
  assert ct_shape.default == [1,224,224,3]


def test_get_input_TensorType():
  input_config = {"shape": [1,224,224,3]}
  ct_input = get_input("input_0", input_config)
  assert isinstance(ct_input, ct.TensorType)
  assert ct_input.name == "input_0"
  assert isinstance(ct_input.shape, ct.Shape)
  assert ct_input.shape.shape == (1,224,224,3)


def test_get_input_ImageType():
  input_config = {"shape": [1,224,224,3], "type": "Image", "bias": [-1,-1,-1], "scale": 1.0/256.0}
  ct_input = get_input("input_0", input_config)
  assert isinstance(ct_input, ct.ImageType)
  assert ct_input.name == "input_0"
  assert isinstance(ct_input.shape, ct.Shape)
  assert ct_input.shape.shape == (1,224,224,3)
  assert ct_input.bias == [-1,-1,-1]
  assert ct_input.scale == approx(1.0/256.0)


