import coremltools as ct
import json
import os
import sys

DISPLAY_TO_USER_ERRORS=[
    "provided is not found in given tensorflow graph",
    "C_in / groups",
    "must all be the same length"
    "but expected one of"
]


def user_assert(b: bool, err_msg: str):
    if not b:
        raise ValueError(err_msg)


def inject_range_dim(v):
    if isinstance(v, int):
      return v
    if isinstance(v, str):
      vv = v.split("..")
      user_assert(len(vv) == 2 and vv[0].isnumeric() and vv[1].isnumeric(),
           "Can not parse shape element {}. Excepted RangeDim, e.g. 1..50".format(v))
      return ct.RangeDim(lower_bound=int(vv[0]), upper_bound=int(vv[1]))
    raise ValueError("Can not parse shape element {}, type {}".format(v, type(v)))


def transform_shape(shape: dict):
  return [inject_range_dim(v) for v in shape]


def get_shape(inp: dict):
  if 'shape' not in inp:
    raise ValueError("shape key not found")
  shape = inp['shape']
  default_shape = inp.get('default_shape')
  if not isinstance(shape, list):
    raise ValueError("shape should be list, found {}, type {}".format(shape, type(shape)))

  list_items = 0
  for s in shape:
    if isinstance(s, list):
        list_items += 1
  is_enum_shapes = list_items == len(shape)
  # All or None shapes element shoud be list
  # Regular shape [1,20,20,3]
  # Enumerated Shapes [[1,20,20,3], [1,40,40,3]]
  user_assert(list_items == 0 or is_enum_shapes,
        "Can not parse shape {}".format(shape))
  if is_enum_shapes:
    enum_shapes = [transform_shape(s) for s in shape]
    print("enum_shapes:", enum_shapes)
    return ct.EnumeratedShapes(shapes=enum_shapes, default=default_shape)
  # Replace string elements such as "1..50" with RangeDim(1, 50)
  tx_shape = transform_shape(shape)
  print("tx_shape:", tx_shape)
  return ct.Shape(shape=tx_shape, default=default_shape)


def get_input(name: str, inp: dict):
    shape = get_shape(inp)
    if 'type' in inp and inp['type'].upper() == 'IMAGE':
      return ct.ImageType(name=name, shape=shape,
        bias=inp.get('bias'), scale=inp.get('scale'))
    else:
      return ct.TensorType(name=name, shape=shape)


def get_input_list(input_config):
  inputs = []
  for n in input_config:
    if isinstance(n, str):
      # {'n1': ..., 'n2': ...}
      v = input_config[n]
      if isinstance(v, list):
        # {'n1':[...], 'n2': [...]}
        print("Added TensorType(name='{}', shape={})".format(n, v))
        inputs.append(ct.TensorType(name=n, shape=v))
      elif isinstance(v, dict):
        # {'n1': {...}, 'n2': {...}}
        inputs.append(get_input(name=n, inp=v))
    elif isinstance(n, list):
      # [[...], [...]]
      print("Added TensorType(shape={})".format(n))
      inputs.append(ct.TensorType(shape=n))
    elif isinstance(n, dict):
      # [{...}, {...}]
      inputs.append(get_input(name=None, inp=n))
    else:
      raise ValueError("Invalid input_shape {}".format(input_shape))
  return inputs


def get_class_labels(compiler_options):
  if compiler_options is None or 'class_labels' not in compiler_options:
    return None
  class_labels=[]
  with open(compiler_options['class_labels']) as f:
    for line in f:
      class_labels.append(line.strip())
  print("len(class_labels):", len(class_labels))
  return class_labels


def get_classifier_config(compiler_options):
  class_labels = get_class_labels(compiler_options)
  if class_labels is None or len(class_labels) == 0:
    return None
  return ct.ClassifierConfig(class_labels)


def convert(input_model_path, output_model_path, input_config_json, compiler_options_json):
  input_config = json.loads(input_config_json)
  compiler_options = None
  if compiler_options_json is not None:
    compiler_options=json.loads(compiler_options_json)
  try:
    inputs = get_input_list(input_config)
    classifier_config = get_classifier_config(compiler_options)
    mlmodel = ct.convert(input_model_path, inputs=inputs, classifier_config=classifier_config)
    mlmodel.save(output_model_path)
    print("MLMODEL was saved to ", output_model_path)
  except ValueError as e:
    print('ValueError:', e)
    err_str = str(e)
    print("Error (str) {}".format(e))
    for x in DISPLAY_TO_USER_ERRORS:
        if x in err_str:
            sys.exit(4)
    sys.exit(1)


if __name__ == "__main__":
  input_model_path = os.environ["INPUT_MODEL"]
  output_model_path = os.environ["OUTPUT_MODEL"]
  input_config_json=os.environ['INPUT_CONFIG']
  compiler_options_json = os.environ.get("COMPILER_OPTIONS")

  print("input_config_json", input_config_json)
  print("input_model_path", input_model_path)
  print("output_model_path", output_model_path)
  print("compiler_options", compiler_options_json)
  convert(input_model_path, output_model_path, input_config_json, compiler_options_json)

