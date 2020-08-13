import coremltools as ct
import json
import os
import sys

def user_assert(b: bool, err_msg: str):
    if not b:
        raise ValueError(err_msg)

DISPLAY_TO_USER_ERRORS=[
    "provided is not found in given tensorflow graph",
    "C_in / groups",
    "must all be the same length"
    "but expected one of"
]

input_config_json=os.environ['INPUT_CONFIG']
framework = os.environ['FRAMEWORK']
input_model_path = os.environ["INPUT_MODEL"]
output_model_path = os.environ["OUTPUT_MODEL"]
compiler_options_json = os.environ.get("COMPILER_OPTIONS")

print("framework", framework)
print("input_config_json", input_config_json)
print("input_model_path", input_model_path)
print("output_model_path", output_model_path)
print("compiler_options", compiler_options_json)

inputs=[]
class_labels=[]
if compiler_options_json is not None:
    compiler_options=json.loads(compiler_options_json)
    if 'class_labels' in compiler_options:
        with open(compiler_options['class_labels']) as f:
            for line in f:
                class_labels.append(line.strip())

print("len(class_labels):", len(class_labels))


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


def append_input_config(name: str, inp: dict):
    shape = get_shape(inp)
    if 'type' in inp and inp['type'].upper() == 'IMAGE':
        inputs.append(ct.ImageType(name=name, shape=shape,
            bias=inp.get('bias'), scale=inp.get('scale')))
    else:
        inputs.append(ct.TensorType(name=name, shape=shape))


input_config=json.loads(input_config_json)
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
            append_input_config(name=n, inp=v)
    elif isinstance(n, list):
        # [[...], [...]]
        print("Added TensorType(shape={})".format(n))
        inputs.append(ct.TensorType(shape=n))
    elif isinstance(n, dict):
        # [{...}, {...}]
        append_input_config(name=None, inp=n)

    else:
        raise ValueError("Invalid input_shape {}".format(input_shape))

try:
    classifier_config = None
    if len(class_labels) > 0:
        classifier_config = ct.ClassifierConfig(class_labels)
    mlmodel=ct.convert(input_model_path, inputs=inputs, classifier_config=classifier_config)
except ValueError as e:
    print('ValueError:', e)
    err_str = str(e)
    print("Error (str) {}".format(e))
    for x in DISPLAY_TO_USER_ERRORS:
        if x in err_str:
            sys.exit(4)
    sys.exit(1)

mlmodel.save(output_model_path)
print("MLMODEL was saved to ", output_model_path)
