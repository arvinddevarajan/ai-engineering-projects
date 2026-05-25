import os
import time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from dotenv import load_dotenv
from costs import calculate_cost

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODELS = [
    "gpt-4o-mini",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
]


def call_model(model, question, context):
    prompt = f"Answer based only on this context. If the answer is not in the context, say 'I don't have that information'.\n\nContext:\n{context}\n\nQuestion:\n{question}"

    start = time.time()

    if model.startswith("gpt"):
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    elif model.startswith("claude"):
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    else:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

    latency = round(time.time() - start, 2)
    cost = calculate_cost(model, input_tokens, output_tokens)

    return {
        "model": model,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency": latency,
        "cost": cost,
    }


def safe_call_model(model, question, context):
    try:
        return call_model(model, question, context)
    except Exception as e:
        return {
            "model": model,
            "answer": f"Error: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency": 0.0,
            "cost": 0.0,
        }


def run_all_models(question, context):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(safe_call_model, model, question, context)
            for model in MODELS
        ]
        return [f.result() for f in futures]