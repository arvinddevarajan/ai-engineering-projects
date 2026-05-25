COSTS_PER_MILLION_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen/qwen3-32b": {"input": 0.29, "output": 0.39},
}


def calculate_cost(model, input_tokens, output_tokens):
    pricing = COSTS_PER_MILLION_TOKENS.get(model)
    if not pricing:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)