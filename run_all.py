#!/usr/bin/env python3
import argparse
import subprocess
import csv
import re
import sys
import os
from pathlib import Path
from tqdm import tqdm

BINARY_NAME = "./search"
TEST_SET = "example"
NUM_EXAMPLES = 5
MAX_PROG_LEN = 5
SAA_CUTOFF = -1 
TIMEOUT_SECONDS = 20 

def get_total_problems(data_file_path):
    if not data_file_path.exists():
        print(f"Error: Data file not found at {data_file_path}")
        sys.exit(1)
    
    with open(data_file_path, 'r') as f:
        line_count = sum(1 for _ in f)
    
    return line_count // NUM_EXAMPLES

def parse_cpp_output(output_str):
    result = {
        "status": "Error",
        "nodes": 0,
        "time": 0.0,
        "raw_output": output_str.strip()
    }
    
    if "Solved!" in output_str:
        result["status"] = "Solved"
    elif "Failed!" in output_str:
        result["status"] = "Failed"
    
    nodes_match = re.search(r"Nodes explored: (\d+)", output_str)
    if nodes_match:
        result["nodes"] = int(nodes_match.group(1))
    
    lines = output_str.strip().split('\n')
    for line in reversed(lines):
        try:
            val = float(line.strip())
            result["time"] = val
            break
        except ValueError:
            continue
            
    return result

def setup_prediction_symlink(data_dir, model_type):
    target_dir_name = f"predictions_{model_type}"
    source_path = data_dir / target_dir_name
    link_path = data_dir / "predictions"

    if not source_path.exists():
        print(f"Error: Prediction directory not found: {source_path}")
        sys.exit(1)

    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            print(f"Error: '{link_path}' is a real directory. Please remove it manually.")
            sys.exit(1)
        link_path.unlink()

    try:
        os.symlink(target_dir_name, link_path)
    except OSError as e:
        print(f"Error creating symlink: {e}")
        sys.exit(1)

def run_experiment_loop(total_problems, search_dir, csv_path, order_type, desc):
    print(f"Starting: {desc}")
    print(f"Saving to: {csv_path}")

    csv_header = ["Problem_ID", "Status", "Nodes_Explored", "Time_Seconds", "Order_Type"]
    
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

        iterator = tqdm(range(total_problems), desc=desc, unit="prob")
        success_count = 0
        timeout_count = 0

        for problem_idx in iterator:
            cmd = [
                str(BINARY_NAME),
                TEST_SET,
                str(NUM_EXAMPLES),
                str(MAX_PROG_LEN),
                str(problem_idx),
                str(order_type),
                str(SAA_CUTOFF)
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=search_dir,
                    timeout=TIMEOUT_SECONDS
                )

                parsed = parse_cpp_output(proc.stdout)
                
                writer.writerow({
                    "Problem_ID": problem_idx,
                    "Status": parsed["status"],
                    "Nodes_Explored": parsed["nodes"],
                    "Time_Seconds": parsed["time"],
                    "Order_Type": order_type
                })
                csvfile.flush()

                if parsed["status"] == "Solved":
                    success_count += 1
                
                iterator.set_postfix(acc=f"{success_count}/{problem_idx+1}", to=timeout_count)

            except subprocess.TimeoutExpired:
                timeout_count += 1
                writer.writerow({
                    "Problem_ID": problem_idx,
                    "Status": "Timeout",
                    "Nodes_Explored": -1,
                    "Time_Seconds": TIMEOUT_SECONDS,
                    "Order_Type": order_type
                })
                csvfile.flush()
                iterator.set_postfix(acc=f"{success_count}/{problem_idx+1}", to=timeout_count)

            except Exception as e:
                tqdm.write(f"Error processing problem {problem_idx}: {e}")

    print(f"Finished {desc}.")
    print(f"Solved: {success_count}/{total_problems} ({success_count/total_problems*100:.2f}%)")
    print(f"Timeouts: {timeout_count}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["transformer", "neural_network"])
    args = parser.parse_args()

    root_dir = Path(__file__).parent.resolve()
    search_dir = root_dir / "enumerative-search"
    binary_path = search_dir / BINARY_NAME
    data_dir = search_dir / "data" / TEST_SET
    input_file = data_dir / "input_values.txt"
    results_dir = root_dir / "results" / args.model

    if not binary_path.exists():
        print(f"Error: Binary not found at {binary_path}")
        sys.exit(1)

    setup_prediction_symlink(data_dir, args.model)
    results_dir.mkdir(parents=True, exist_ok=True)
    total_problems = get_total_problems(input_file)

    """
    run_experiment_loop(
        total_problems=total_problems,
        search_dir=search_dir,
        csv_path=results_dir / "baseline_results.csv",
        order_type=0, 
        desc="Phase 1: Baseline (Prior)"
    )
    """

    run_experiment_loop(
        total_problems=total_problems,
        search_dir=search_dir,
        csv_path=results_dir / "prediction_results.csv",
        order_type=1,
        desc=f"Phase 2: Neural Guided ({args.model})"
    )

if __name__ == "__main__":
    main()