import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

JUDGE_PROMPT = """You are evaluating an AI-generated answer against a reference context.

Question: {question}

Context used for answering:
{context}

Answer to evaluate:
{answer}

Score the answer on two dimensions, each from 1-10:
1. Faithfulness: Is the answer grounded in the context? Does it avoid making claims not supported by the context?
2. Relevance: Does the answer actually address the question asked?

Respond with valid JSON only, no explanation:
{{"faithfulness": <score>, "relevance": <score>}}"""


def evaluate(question, context, answer):
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        scores = json.loads(raw)
        return {
            "faithfulness": scores.get("faithfulness", 0),
            "relevance": scores.get("relevance", 0),
        }
    except json.JSONDecodeError:
        return {"faithfulness": 0, "relevance": 0}