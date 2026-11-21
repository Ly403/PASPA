import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

arr = np.load(args.input_path)
# print(arr.keys())
arr = dict(arr)
arr["betas"] = arr["betas"][0]
np.savez(args.output_path, **arr)