import argparse
import subprocess
import csv
import re
import sys
import os
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

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

def process_problem(problem_idx, binary_path, search_dir, test_set, num_examples, max_len, order_type, saa_cutoff, timeout):
    """
    worker to execute individual problems. 
    keeps track of problem idx to feed back to orchestrator
    """
    cmd = [
        str(binary_path),
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
        # prolly not needed
        parsed["Order_Type"] = order_type
        return problem_idx, parsed

    except subprocess.TimeoutExpired:
        return problem_idx, {
            "status": "Timeout",
            "nodes": -1,
            "time": timeout,
            "Order_Type": order_type
        }
    except Exception as e:
        return problem_idx, {
            "status": "Error",
            "error_msg": str(e),
            "nodes": 0,
            "time": 0.0,
            "Order_Type": order_type
        }

def run_experiment_loop(test_set, total_problems, search_dir, binary_path, csv_path, order_type, desc, num_examples, max_len, saa_cutoff, timeout, max_workers):
    print(f"begin {desc}")
    print(f"csv_path = {csv_path}")
    print(f"running loop for total problems = {total_problems} using {max_workers} threads")

    csv_header = ["Problem_ID", "Status", "Nodes_Explored", "Time_Seconds", "Order_Type"]
    
    # maintain buffer for out of order arrivals
    result_buffer = {}
    next_idx_to_write = 0
    
    success_count = 0
    timeout_count = 0

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for problem_idx in range(total_problems):
                future = executor.submit(
                    process_problem,
                    problem_idx,
                    binary_path,
                    search_dir,
                    test_set,
                    num_examples,
                    max_len,
                    order_type,
                    saa_cutoff,
                    timeout
                )
                futures.append(future)

            iterator = tqdm(concurrent.futures.as_completed(futures), total=total_problems, desc=desc, unit="prob")
            
            for future in iterator:
                idx, result = future.result()
                status = result.get("status")
                if status == "Solved":
                    success_count += 1
                elif status == "Timeout":
                    timeout_count += 1
                
                result_buffer[idx] = result

                # so for every iteration, we check if buffer has the next index to 
                # write quickly, this ensures that we don't block the child processes
                # but still maintain csv ordering
                while next_idx_to_write in result_buffer:
                    data = result_buffer.pop(next_idx_to_write)
                    
                    writer.writerow({
                        "Problem_ID": next_idx_to_write,
                        "Status": data["status"],
                        "Nodes_Explored": data.get("nodes", 0),
                        "Time_Seconds": data.get("time", 0.0),
                        "Order_Type": data.get("Order_Type", order_type)
                    })
                    next_idx_to_write += 1
                
                # flush to file every write in case i want to do some progress saving
                csvfile.flush()
                
                # update the tqdm visualizer
                iterator.set_postfix(acc=f"{success_count}/{idx+1}", to=timeout_count)

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
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of threads to use.")
    parser.add_argument("--group", type=str, default=None, help="Group results into dir.")

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

    identifier = f"num_ex_{args.num_examples}_max_len_{args.max_len}"
    if args.group is not None:
        identifier = f"{args.group}/{identifier}"
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
        binary_path=binary_path,
        csv_path=csv_path,
        order_type=args.order_type,
        desc=f"evaluating ({args.test_set})",
        num_examples=args.num_examples,
        max_len = args.max_len,
        saa_cutoff=args.saa_cutoff,
        timeout=args.timeout,
        max_workers=args.workers
    )

if __name__ == "__main__":
    main()