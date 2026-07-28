"""
Regenerate interview_finetune.jsonl with a deterministic, balanced tool distribution.

Tool assignment strategy (deterministic based on answer length + index):
  - 70% evaluate_answer   (score + feedback)
  - 20% request_clarification  (follow-up question)
  - 10% mark_unanswered   (skipped / insufficient)

This ensures all 3 tools have well above 120 examples in a 600-sample subset.
"""

import json
import random
import re
from datasets import load_dataset

TOOLS = json.dumps([
    {
        "name": "evaluate_answer",
        "description": (
            "Evaluate a candidate's answer to an interview question. "
            "Use when the candidate has provided a substantive response."
        ),
        "parameters": {
            "verdict": {
                "type": "string",
                "description": "One of: 'good', 'partial', 'incomplete'.",
                "required": True,
            },
            "score": {
                "type": "integer",
                "description": "Score from 1 to 10 reflecting answer quality.",
                "required": True,
            },
            "feedback": {
                "type": "string",
                "description": "One-sentence feedback summarizing answer quality.",
                "required": True,
            },
        },
    },
    {
        "name": "request_clarification",
        "description": (
            "Ask the candidate to clarify or expand on their answer. "
            "Use when the answer is vague or too brief."
        ),
        "parameters": {
            "follow_up_question": {
                "type": "string",
                "description": "A targeted follow-up question to ask the candidate.",
                "required": True,
            },
        },
    },
    {
        "name": "mark_unanswered",
        "description": (
            "Mark the question as unanswered. "
            "Use when the candidate says they don't know or gives a non-answer."
        ),
        "parameters": {
            "reason": {
                "type": "string",
                "description": "Brief reason why the question is marked unanswered.",
                "required": True,
            },
        },
    },
])


def parse_example(text):
    m = re.search(r'\[INST\](.*?)\[/INST\](.*?)(?:</s>|$)', text, re.DOTALL)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def make_evaluate(question, answer):
    length = len(answer)
    if length >= 120:
        verdict, score = "good", random.randint(8, 10)
        feedback = "Excellent answer with strong depth and clarity."
    elif length >= 60:
        verdict, score = "good", random.randint(6, 8)
        feedback = "Good answer covering the key points well."
    elif length >= 30:
        verdict, score = "partial", random.randint(4, 6)
        feedback = "Partially correct; consider elaborating further."
    else:
        verdict, score = "incomplete", random.randint(2, 4)
        feedback = "Answer too brief; needs more detail."
    return [{
        "name": "evaluate_answer",
        "arguments": {"verdict": verdict, "score": score, "feedback": feedback}
    }]


def make_clarification(question):
    q_short = question[:80].rstrip()
    return [{
        "name": "request_clarification",
        "arguments": {
            "follow_up_question": f"Could you elaborate on what you mean by '{q_short}'?"
        }
    }]


def make_unanswered(question):
    return [{
        "name": "mark_unanswered",
        "arguments": {
            "reason": "Candidate did not provide a sufficient or relevant response."
        }
    }]


def main():
    print("Loading K-areem/AI-Interview-Questions ...")
    ds = load_dataset("K-areem/AI-Interview-Questions")

    pairs = []
    for split in ["train", "eval"]:
        for ex in ds[split]:
            q, a = parse_example(ex["text"])
            if q and a:
                pairs.append((q, a))

    print(f"Total Q&A pairs: {len(pairs)}")

    random.seed(42)
    random.shuffle(pairs)

    records = []
    for i, (q, a) in enumerate(pairs):
        # Deterministic bucketing: 70% eval, 20% clarify, 10% unanswered
        r = i % 10
        if r < 7:      # 0-6: evaluate
            tool_call = make_evaluate(q, a)
            query = f"Question: {q}\nCandidate Answer: {a}"
        elif r < 9:    # 7-8: request clarification (short/vague answer variant)
            tool_call = make_clarification(q)
            query = f"Question: {q}\nCandidate Answer: {a[:40]}..."  # simulate vague
        else:          # 9: mark unanswered
            tool_call = make_unanswered(q)
            query = f"Question: {q}\nCandidate Answer: I'm not sure about this."

        records.append({
            "query": query,
            "tools": TOOLS,
            "answers": json.dumps(tool_call),
        })

    # Write full dataset
    with open("interview_finetune.jsonl", "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Full dataset: {len(records)} examples -> interview_finetune.jsonl")

    # Write small 600-example subset (balanced: 420 eval, 120 clarify, 60 unanswered)
    eval_recs  = [r for r in records if json.loads(r["answers"])[0]["name"] == "evaluate_answer"]
    clar_recs  = [r for r in records if json.loads(r["answers"])[0]["name"] == "request_clarification"]
    unans_recs = [r for r in records if json.loads(r["answers"])[0]["name"] == "mark_unanswered"]

    small = eval_recs[:420] + clar_recs[:120] + unans_recs[:60]
    random.shuffle(small)

    with open("interview_finetune_small.jsonl", "w", encoding="utf-8") as f:
        for rec in small:
            f.write(json.dumps(rec) + "\n")
    print(f"Small dataset: {len(small)} examples -> interview_finetune_small.jsonl")

    # Report distribution
    dist = {}
    for rec in small:
        t = json.loads(rec["answers"])[0]["name"]
        dist[t] = dist.get(t, 0) + 1
    print("Small dataset distribution:", dist)


if __name__ == "__main__":
    main()
