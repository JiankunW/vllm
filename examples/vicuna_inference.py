from vllm import LLM, SamplingParams


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(2, temperature=0.8, top_p=0.95)

# Create an LLM.
# llm = LLM(model="/cpfs01/user/sunpeng/Sequence-Scheduling/ckpts/vicuna-7b",
#           tokenizer_mode="slow", tensor_parallel_size=2)

llm = LLM(model="/home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b",
          tokenizer_mode="slow", tensor_parallel_size=2)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
