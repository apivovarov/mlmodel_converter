import coremltools as ct
import json
import os
import sys

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

mlmodel=ct.convert(input_model_path, inputs=inputs)
mlmodel.save(output_model_path)
print("MLMODEL was saved to ", output_model_path)
