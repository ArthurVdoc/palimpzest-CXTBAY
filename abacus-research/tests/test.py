import litellm

litellm.model_cost["Qwen/Qwen2.5-7B-Instruct"] = {
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "max_tokens": 32768
}

litellm.model_cost["hosted_vllm/Qwen/Qwen2.5-7B-Instruct"] = {
    "input_cost_per_token": 0,
    "output_cost_per_token": 0,
    "max_tokens": 32768
}

cost = litellm.completion_cost(
    model="Qwen/Qwen2.5-7B-Instruct",
    custom_llm_provider="hosted_vllm",
    messages=[{"role": "user", "content": "Hello world"}],
    completion="This is a test reply"
)

print(cost)


# import sys, litellm
# print(litellm.__file__)
# print(sys.path)