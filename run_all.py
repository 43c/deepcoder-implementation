import argparse
import subprocess
import csv
import re
import sys
import os
from pathlib import Path
from tqdm import tqdm

BINARY_NAME = "./search"

def get_total_problems(data_file_path, num_examples):
    if not data_file_path.exists():
        print(f"data file not found at {data_file_path}")
        sys.exit(1)
    
    with open(data_file_path, 'r') as f:
        line_count = sum(1 for _ in f)
    
    return line_count // num_examples

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

def run_experiment_loop(test_set, total_problems, search_dir, csv_path, order_type, desc, num_examples, max_len, saa_cutoff, timeout):
    print(f"begin {desc}")
    print(f"csv_path = {csv_path}")
    print(f"running loop for total problems = {total_problems}")

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
                test_set,
                str(num_examples),
                str(max_len),
                str(problem_idx),
                str(order_type),
                str(saa_cutoff)
            ]

            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=search_dir,
                    timeout=timeout
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
                    "Time_Seconds": timeout,
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
    parser.add_argument("test_set", choices=["transformer", "neural_network"])
    parser.add_argument("--num-examples", type=int, default=5, help="Number of example I/Os per program.")
    parser.add_argument("--max-len", type=int, default=5, help="Maximum program length to search up to.")
    parser.add_argument("--saa-cutoff", type=int, default=-1, help="I honestly have no idea what this does.")
    parser.add_argument("--timeout", type=int, default=20, help="Timeout in seconds.")
    parser.add_argument("--max-problems", type=int, default=-1, help="Max number of problems to solve.")
    parser.add_argument("--order-type", type=int, default=1, help="1 to use predictions, 0 to use prior.")

    args = parser.parse_args()

    root_dir = Path(__file__).parent.resolve()
    search_dir = root_dir / "enumerative-search"
    binary_path = search_dir / BINARY_NAME
    data_dir = search_dir / "data" / args.test_set
    input_file = data_dir / "input_values.txt"
    results_dir = root_dir / "results"

    if not binary_path.exists():
        print(f"Error: Binary not found at {binary_path}")
        sys.exit(1)

    results_dir.mkdir(parents=True, exist_ok=True)

    total_problems = get_total_problems(input_file, args.num_examples)

    if args.max_problems > 0:
        total_problems = min(total_problems, args.max_problems)

    """
    run_experiment_loop(
        total_problems=total_problems,
        search_dir=search_dir,
        csv_path=results_dir / "baseline_results.csv",
        order_type=0, 
        desc="Phase 1: Baseline (Prior)"
    )
    """

    identifier = f"num_ex_{args.num_examples}_max_len_{args.max_len}"
    csv_name = f"{args.test_set}.csv"
    csv_dir = results_dir / identifier / args.test_set
    try:
        csv_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"error making csv path somehow: {e}")

    csv_path = csv_dir / csv_name

    run_experiment_loop(
        test_set=args.test_set,
        total_problems=total_problems,
        search_dir=search_dir,
        csv_path=csv_path,
        order_type=args.order_type,
        desc=f"evaluating ({args.test_set})",
        num_examples=args.num_examples,
        max_len = args.max_len,
        saa_cutoff=args.saa_cutoff,
        timeout=args.timeout
    )

if __name__ == "__main__":
    main()