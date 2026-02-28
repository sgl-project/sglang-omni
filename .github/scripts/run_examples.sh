# Run examples
# we only run two stage examples for now cuz the CI only has 2 GPUs
python examples/run_two_stage_demo.py
python examples/run_two_stage_demo.py --relay shm
python examples/run_two_stage_llama_demo.py --prompt "Hello, how are you?" --model-id unsloth/Llama-3.2-1B-Instruct
