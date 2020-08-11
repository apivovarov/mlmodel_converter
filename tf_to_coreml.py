import coremltools as ct
import json
import os
import sys

DISPLAY_TO_USER_ERRORS=[
    "provided is not found in given tensorflow graph",
    "C_in / groups",
    "must all be the same length"
]

input_config_json=os.environ['INPUT_CONFIG']
framework = os.environ['FRAMEWORK']
input_model_path = os.environ["INPUT_MODEL"]
output_model_path = os.environ["OUTPUT_MODEL"]
compiler_options_json = os.environ["COMPILER_OPTIONS"]

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


def append_input_config(name: str, inp: dict):
    if 'type' in inp and inp['type'].upper() == 'IMAGE':
        inputs.append(ct.ImageType(name=name, shape=inp.get('shape'), 
            bias=inp.get('bias'), scale=inp.get('scale')))
    else:
        inputs.append(ct.TensorType(name=name, shape=inp.get('shape')))


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
