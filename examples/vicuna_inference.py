from vllm import LLM, SamplingParams


# Sample prompts.
# prompts = [
#     "Hello, my name is",
#     "The president of the United States is",
#     "The capital of France is",
#     "The future of AI is",
# ]
prompts = [
    "Describe a time when you had to make a difficult decision.",
    "Explain why the following fraction is equivalent to 1/4\n\n### Input:\n4/16",
    "Write a short story in third person narration about a protagonist who has to make an important career decision."
]
# PROMPT_PREFIX = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
# PROMPT_QUERY = "\n\n### Response:\nDon't output the response for the above instruction. Instead, you need to predict the number of tokens in your response. Output one number only. ASSISTANT:"

# PROMPT_PREFIX_EVAL = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# follow Seq paper benchmark params
# sampling_params = SamplingParams(temperature=0.8, top_k=4, presence_penalty=0.6, max_tokens=16)

# Create an LLM.
llm = LLM(model="/home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b",
          tokenizer_mode="slow", tensor_parallel_size=1)

# 2 gpu
# llm = LLM(model="/home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter",
#           tokenizer_mode="slow", tensor_parallel_size=2, gpu_memory_utilization=.4)

# # single gpu
# llm = LLM(model="/home/jiankun/randomfun/Sequence-Scheduling/ckpts/vicuna-7b-adapter",
#           tokenizer_mode="slow", tensor_parallel_size=1, gpu_memory_utilization=.8)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
