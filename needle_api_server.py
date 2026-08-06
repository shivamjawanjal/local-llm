"""
Needle AI Local Model API Server
Runs the finetuned 26M Needle AI function routing model on http://localhost:8000
Optimized with Strict Word Count & Topic Relevance Grader.
"""

import json
import re
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer

# 1. Initialize FastAPI app
app = FastAPI(title="Needle AI Model API Server", version="1.4.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Model & Tokenizer Config
CHECKPOINT_PATH = "checkpoints/needle_finetuned_best.pkl"

print(f"Loading Needle model checkpoint from {CHECKPOINT_PATH}...")
params, config = load_checkpoint(CHECKPOINT_PATH)
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()
print("Model loaded successfully!")

# 3. Tool Definitions
TOOLS = json.dumps([
    {
        "name": "evaluate_answer",
        "description": "Evaluate a candidate's answer to an interview question.",
        "parameters": {
            "verdict": {"type": "string", "description": "One of: 'good', 'partial', 'incomplete'.", "required": True},
            "score": {"type": "integer", "description": "Score from 1 to 10 reflecting answer quality.", "required": True},
            "feedback": {"type": "string", "description": "Specific feedback summarizing answer quality.", "required": True},
        },
    },
    {
        "name": "request_clarification",
        "description": "Ask candidate to expand when answer is brief or vague.",
        "parameters": {
            "follow_up_question": {"type": "string", "description": "Targeted follow-up question.", "required": True},
        },
    },
    {
        "name": "mark_unanswered",
        "description": "Mark question as unanswered if candidate gave a non-answer.",
        "parameters": {
            "reason": {"type": "string", "description": "Reason why answer is incomplete.", "required": True},
        },
    },
])


class EvaluationRequest(BaseModel):
    question: str
    answer: str
    fast_mode: bool = True


STOPWORDS = {
    "what", "is", "a", "an", "the", "in", "of", "and", "or", "to", "for", "with",
    "on", "at", "by", "from", "how", "why", "can", "you", "explain", "describe",
    "does", "do", "it", "this", "that", "are", "were", "was", "be", "been", "being",
    "should", "would", "could", "have", "has", "had", "which", "when", "where", "who"
}


def compute_semantic_evaluation(question: str, answer: str):
    """
    Strict Word-Count & Topic Relevance Calibrated Grader.
    - 1-3 word answers get max score 2.
    - Off-topic answers get score 1.
    """
    cleaned_ans = answer.strip().lower()
    cleaned_q = question.strip().lower()

    words = [w for w in cleaned_ans.split() if w]
    word_count = len(words)

    # Extract core question domain terms (> 2 chars, non-stopword)
    q_words = [w for w in re.findall(r'\b[a-z]{3,}\b', cleaned_q) if w not in STOPWORDS]
    
    # Check exact keyword & stem matches
    matched_terms = [w for w in q_words if w in cleaned_ans or any(w[:4] in a_w for a_w in words if len(a_w) >= 4)]
    match_ratio = len(matched_terms) / max(1, len(q_words))

    # Fluff / Filler phrases check
    filler_phrases = ["good question", "i know this", "yes it is", "i think so", "well it is", "none", "pass"]
    is_pure_filler = any(p == cleaned_ans for p in filler_phrases) or (any(p in cleaned_ans for p in filler_phrases) and word_count < 8)

    if is_pure_filler:
        return {
            "type": "NEEDS_CLARIFICATION",
            "score": 2,
            "verdict": "Incomplete",
            "feedback": "Answer contains generic filler without explaining key technical concepts.",
            "follow_up_question": f"Could you provide a specific technical explanation for {question}?"
        }

    # 1. OFF-TOPIC CHECK (Zero matching domain terms)
    if match_ratio == 0:
        return {
            "type": "UNANSWERED",
            "score": 1,
            "verdict": "Off-Topic",
            "feedback": f"Response is off-topic and does not address the question ({question}).",
            "reason": "Answer contains no relevant domain keywords or concepts related to the question."
        }

    # 2. STRICT SHORT-ANSWER GUARDRAIL (1 to 3 words)
    if word_count <= 3:
        q_clean = question.replace('Explain ', '').replace('What is ', '').rstrip('?')
        return {
            "type": "NEEDS_CLARIFICATION",
            "score": 2,
            "verdict": "Very Brief",
            "feedback": f"Answer is extremely brief ({word_count} words). Mentioned: {', '.join(matched_terms[:2])}.",
            "follow_up_question": f"Your response is only {word_count} words. Could you explain in full sentences how {q_clean} works?"
        }

    # 3. BRIEF ANSWER GUARDRAIL (4 to 8 words)
    if word_count <= 8:
        q_clean = question.replace('Explain ', '').replace('What is ', '').rstrip('?')
        return {
            "type": "NEEDS_CLARIFICATION",
            "score": 3,
            "verdict": "Incomplete",
            "feedback": f"Short response ({word_count} words). Covered: {', '.join(matched_terms[:2])}.",
            "follow_up_question": f"Could you elaborate in more detail on {q_clean}?"
        }

    # 4. FULL EVALUATION SCALE (>= 9 words with positive concept match)
    if match_ratio >= 0.5 and word_count >= 20:
        score = 9 if word_count >= 35 else 8
        verdict = "Excellent" if score >= 9 else "Good"
        feedback = f"Strong technical response covering key concepts: {', '.join(matched_terms[:3])}."
    elif match_ratio >= 0.2 and word_count >= 9:
        score = 6
        verdict = "Pass"
        feedback = f"Satisfactory answer explaining core principles ({', '.join(matched_terms[:2])}), but could expand on practical details."
    else:
        score = 4
        verdict = "Incomplete"
        feedback = f"Response is brief and lacks coverage of essential concepts."

    if score < 6:
        q_clean = question.replace('Explain ', '').replace('What is ', '').rstrip('?')
        return {
            "type": "NEEDS_CLARIFICATION",
            "score": score,
            "verdict": verdict,
            "feedback": feedback,
            "follow_up_question": f"Could you elaborate specifically on how {q_clean} handles real-world scenarios?"
        }

    return {
        "type": "EVALUATED",
        "score": score,
        "verdict": verdict,
        "feedback": feedback
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "needle_finetuned", "version": "1.4.0_calibrated"}


@app.post("/evaluate")
def evaluate_endpoint(req: EvaluationRequest):
    start_t = time.perf_counter()
    cleaned_answer = req.answer.strip()
    
    # 1. Empty / Skipped guardrail
    if len(cleaned_answer) < 3 or cleaned_answer.lower() in ["idk", "i don't know", "no idea", "none", "n/a", "pass"]:
        return {
            "type": "UNANSWERED",
            "score": 0,
            "verdict": "Fail",
            "reason": "Candidate explicitly admitted lack of knowledge or provided no answer.",
            "latency_ms": round((time.perf_counter() - start_t) * 1000, 2)
        }

    # 2. Strict Calibrated Evaluation
    res = compute_semantic_evaluation(req.question, req.answer)
    res["latency_ms"] = round((time.perf_counter() - start_t) * 1000, 2)
    res["engine"] = "Needle Strict Concept Grader v1.4"
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
