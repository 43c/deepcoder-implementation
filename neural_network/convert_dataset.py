#!/usr/bin/env python3
import pickle
import os
from pathlib import Path

DATASET_PATH = "bickle100k.pickle"
OUTPUT_PATH = "enumerative-search/data/example" 

def format_val(val):
    if isinstance(val, (list, tuple)):
        return " ".join(str(x) for x in val)
    else:
        return str(val)

def convert_dataset():
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "rb") as f:
        d = pickle.load(f)
    
    if hasattr(d, "dataset"):
        all_data = d.dataset
    else:
        all_data = d

    print(f"All data loaded: {len(all_data)} entries")

    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    input_file_path = out_dir / "input_values.txt"
    output_file_path = out_dir / "output_values.txt"

    print(f"Writing to {out_dir}...")

    with open(input_file_path, "w") as f_in, open(output_file_path, "w") as f_out:
        
        for idx, entry in enumerate(all_data):
            for example in entry.examples:
                inp = example.inputs
                
                if isinstance(inp, tuple):
                    inputs_strings = []
                    for arg in inp:
                        formatted_arg = format_val(arg)
                        inputs_strings.append(formatted_arg)
                    
                    inp_str = " | ".join(inputs_strings)
                else:
                    inp_str = format_val(inp)
                
                out_str = format_val(example.output)

                f_in.write(inp_str + "\n")
                f_out.write(out_str + "\n")

    print("Conversion complete!")

if __name__ == "__main__":
    convert_dataset()