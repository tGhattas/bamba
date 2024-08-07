import os
import yaml
import argparse

def config(num_procs: int):
    # Define the configuration
    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU",
        "num_machines": 1,
        "num_processes": num_procs,  # Set this to the number of GPUs you want to use
        "machine_rank": 0,
        "main_process_ip": None,
        "main_process_port": None,
        "main_training_function": "main",
        "deepspeed_config": {},
        "fsdp_config": {},
        "gpu_ids": "all",  # Use 'all' to utilize all available GPUs, or specify GPU IDs as a list
        "mixed_precision": "no",  # Set to 'fp16' or 'bf16' for mixed precision training
        "use_cpu": False,
    }

    # Create the directory if it does not exist
    config_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
    os.makedirs(config_dir, exist_ok=True)

    # Write the configuration to a file
    config_path = os.path.join(config_dir, "default_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Accelerate configuration file created at {config_path}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_procs", type=int, required=True)
    args = parser.parse_args()
    
    config(args.num_procs)