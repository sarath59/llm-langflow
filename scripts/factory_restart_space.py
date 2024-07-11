import argparse
from huggingface_hub import HfApi, list_models
from rich import print
import agentops
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize AgentOps
AGENTOPS_API_KEY = os.getenv("AGENTOPS_API_KEY")
if not AGENTOPS_API_KEY:
    raise ValueError("AGENTOPS_API_KEY not found in environment variables")

agentops.init(AGENTOPS_API_KEY)

@agentops.record_function('list_models')
def get_models():
    return list_models()

@agentops.record_function('parse_arguments')
def parse_arguments():
    args = argparse.ArgumentParser(description="Restart a space in the Hugging Face Hub.")
    args.add_argument("--space", type=str, help="The space to restart.")
    args.add_argument("--token", type=str, help="The Hugging Face API token.")
    return args.parse_args()

@agentops.record_function('validate_arguments')
def validate_arguments(parsed_args):
    if not parsed_args.space:
        print("Please provide a space to restart.")
        return False
    if not parsed_args.token:
        print("Please provide an API token.")
        return False
    return True

@agentops.record_function('create_hf_api')
def create_hf_api(token):
    return HfApi(
        endpoint="https://huggingface.co",  # Can be a Private Hub endpoint.
        token=token,
    )

@agentops.record_function('restart_space')
def restart_space(hf_api, space):
    return hf_api.restart_space(space, factory_reboot=True)

@agentops.record_function('main')
def main():
    models = get_models()
    parsed_args = parse_arguments()
    
    if not validate_arguments(parsed_args):
        return
    
    hf_api = create_hf_api(parsed_args.token)
    
    try:
        space_runtime = restart_space(hf_api, parsed_args.space)
        print(space_runtime)
    except Exception as e:
        agentops.log_error(f"Error restarting space: {str(e)}")
        print(f"An error occurred while restarting the space: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        agentops.log_error(str(e))
    finally:
        agentops.end_session('Success')