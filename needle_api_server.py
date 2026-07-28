"""
Local FastAPI HTTP Server for Needle AI Interview Evaluator.
Exposes a lightweight REST endpoint on http://localhost:8000/evaluate
"""

import os
import sys
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Ensure needle import works
from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer

app = FastAPI(title="Needle Local Evaluation API", version="1.0.0")

# Enable CORS so your local web app (e.g. http://localhost:5173) can query it directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_PATH = os.getenv("NEEDLE_CHECKPOINT", "checkpoints/needle_finetuned_best.pkl")

TOOLS = json.dumps([
    {
        "name": "evaluate_answer",
        "description": "Evaluate a candidate's answer to an interview question.",
        "parameters": {
            "verdict": {"type": "string", "required": True},
            "score": {"type": "integer", "required": True},
            "feedback": {"type": "string", "required": True},
        },
    },
    {
        "name": "request_clarification",
        "description": "Ask the candidate for clarification if the answer is vague.",
        "parameters": {
            "follow_up_question": {"type": "string", "required": True},
        },
    },
    {
        "name": "mark_unanswered",
        "description": "Mark question as unanswered if candidate doesn't know or skips.",
        "parameters": {
            "reason": {"type": "string", "required": True},
        },
    },
])

# Global model state
params = None
config = None
model = None
tokenizer = None


class EvaluationRequest(BaseModel):
    question: str
    answer: str


@app.on_event("startup")
def load_needle():
    global params, config, model, tokenizer
    print(f"Loading Needle checkpoint: {CHECKPOINT_PATH} ...")
    if not os.path.exists(CHECKPOINT_PATH):
        raise RuntimeError(f"Checkpoint not found at: {CHECKPOINT_PATH}")
    
    params, config = load_checkpoint(CHECKPOINT_PATH)
    model = SimpleAttentionNetwork(config)
    tokenizer = get_tokenizer()
    print("Needle model loaded and ready on http://localhost:8000!")


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "needle_finetuned"}


@app.post("/evaluate")
def evaluate_endpoint(req: EvaluationRequest):
    cleaned_answer = req.answer.strip()
    
    # 1. Non-answer / empty guardrail
    if len(cleaned_answer) < 5 or cleaned_answer.lower() in ["idk", "i don't know", "no idea", "none"]:
        return {
            "type": "UNANSWERED",
            "score": 0,
            "reason": "Candidate did not provide a response.",
        }

    query = f"Question: {req.question}\nCandidate Answer: {req.answer}"
    
    try:
        result_str = generate(
            model, params, tokenizer,
            query=query,
            tools=TOOLS,
            stream=False,
        )
        
        tool_calls = json.loads(result_str)
        call = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
        name = call.get("name")
        args = call.get("arguments", {})
        
        # 2. Short answer guardrail
        if len(cleaned_answer) < 35 and name == "evaluate_answer":
            q_clean = req.question.replace('Explain ', '').replace('What is ', '').rstrip('?')
            return {
                "type": "NEEDS_CLARIFICATION",
                "score": 4,
                "follow_up_question": f"Your answer is quite brief. Could you explain in more detail how {q_clean} works?",
            }

        if name == "evaluate_answer":
            return {
                "type": "EVALUATED",
                "verdict": args.get("verdict", "Good"),
                "score": args.get("score", 8),
                "feedback": args.get("feedback", "Good answer."),
            }
        elif name == "request_clarification":
            return {
                "type": "NEEDS_CLARIFICATION",
                "score": 5,
                "follow_up_question": args.get("follow_up_question", "Could you elaborate further?"),
            }
        elif name == "mark_unanswered":
            return {
                "type": "UNANSWERED",
                "score": 0,
                "reason": args.get("reason", "Incomplete answer."),
            }
        else:
            return {"type": "UNKNOWN", "score": 5, "raw": call}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
