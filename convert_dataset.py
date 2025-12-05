import pickle
import os
from pathlib import Path

DATASET_PATH = "dataset/eval/eval_set.pickle"
OUTPUT_PATH = "enumerative-search/data/example" 

def format_val(val):
    if isinstance(val, (list, tuple)):
        return " ".join(str(x) for x in val)
    else:
        return str(val)

def get_type_str(val):
    if isinstance(val, (list, tuple)):
        return "Array"
    return "Int"

def convert_dataset():
    print(f"Loading dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "rb") as f:
        d = pickle.load(f)
    
    if hasattr(d, "dataset"):
        all_data = d.dataset
    else:
        all_data = d

    print(f"All data loaded: {len(all_data)} entries")

    out_dir = Path(OUTPUT_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files_map = {
        "in_val":  open(out_dir / "input_values.txt", "w"),
        "in_type": open(out_dir / "input_types.txt", "w"),
        "out_val": open(out_dir / "output_values.txt", "w"),
        "out_type": open(out_dir / "output_types.txt", "w"),
    }

    print(f"Writing to {out_dir} (Overwriting existing files)...")

    try:
        for idx, entry in enumerate(all_data):
            first_ex = entry.examples[0]
            
            in_types = []
            inputs_container = first_ex.inputs
            if not isinstance(inputs_container, (list, tuple)):
                inputs_container = [inputs_container]
            
            for arg in inputs_container:
                in_types.append(get_type_str(arg))
            
            files_map["in_type"].write(" ".join(in_types) + "\n")
            files_map["out_type"].write(get_type_str(first_ex.output) + "\n")

            for example in entry.examples:
                inp = example.inputs
                
                if isinstance(inp, (list, tuple)):
                    inputs_strings = [format_val(arg) for arg in inp]
                    inp_str = " | ".join(inputs_strings)
                else:
                    inp_str = format_val(inp)
                
                out_str = format_val(example.output)

                files_map["in_val"].write(inp_str + "\n")
                files_map["out_val"].write(out_str + "\n")

    finally:
        for f in files_map.values():
            f.close()

    print("Conversion complete!")

if __name__ == "__main__":
    convert_dataset()