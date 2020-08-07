import coremltools as ct
import json
import os
import sys

DISPLAY_TO_USER_ERRORS=[
    "provided is not found in given tensorflow graph",
    "C_in / groups",
]

input_shape_json=os.environ['INPUT_SHAPE']
framework = os.environ['FRAMEWORK']
input_model_path = os.environ["INPUT_MODEL"]
output_model_path = os.environ["OUTPUT_MODEL"]

print("framework", framework)
print("input_shape_json", input_shape_json)
print("input_model_path", input_model_path)
print("output_model_path", output_model_path)

inputs=[]
input_shape=json.loads(input_shape_json)
for n in input_shape:
    if isinstance(n, str):
        shape=input_shape[n]
        print("Added TensorType(name='{}', shape={})".format(n, shape))
        inputs.append(ct.TensorType(name=n, shape=shape))
    elif framework in ['PYTORCH'] and isinstance(n, list):
        print("Added ImageType(shape={})".format(n))
        inputs.append(ct.TensorType(shape=n))
    else:
        raise ValueError("Invalid input_shape {}".format(input_shape))

try:
    mlmodel=ct.convert(input_model_path, inputs=inputs)
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
