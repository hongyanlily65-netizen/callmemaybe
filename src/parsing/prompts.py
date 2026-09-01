def get_prompts(path: str) -> list[str]:
    with open(path, 'r') as f:
        prompts = json.load(f)
    prompt_list = []
    for prompt in prompts:
        prompt_list.append(prompt['prompt'])
    return prompt_list