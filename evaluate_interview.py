"""
AI Interview Evaluator using the finetuned Needle model.
"""

import json
from needle import SimpleAttentionNetwork, load_checkpoint, generate, get_tokenizer

CHECKPOINT_PATH = "checkpoints/needle_finetuned_best.pkl"

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

# Load model and tokenizer once
print("Loading finetuned Needle model...")
params, config = load_checkpoint(CHECKPOINT_PATH)
model = SimpleAttentionNetwork(config)
tokenizer = get_tokenizer()
print("Model loaded successfully!")


def evaluate(question: str, answer: str):
    """
    Evaluate a candidate's answer to an interview question.
    
    Returns a dict with verdict, score (1-10), feedback, follow-up, or reason.
    """
    cleaned_answer = answer.strip()
    
    # Quality Guardrail 1: Check for empty or non-answers
    if len(cleaned_answer) < 5 or cleaned_answer.lower() in ["idk", "i don't know", "no idea", "none"]:
        return {
            "type": "UNANSWERED",
            "reason": "Candidate did not provide a response.",
        }
        
    query = f"Question: {question}\nCandidate Answer: {answer}"
    
    result_str = generate(
        model, params, tokenizer,
        query=query,
        tools=TOOLS,
        stream=False,
    )
    
    try:
        tool_calls = json.loads(result_str)
        call = tool_calls[0] if isinstance(tool_calls, list) else tool_calls
        name = call.get("name")
        args = call.get("arguments", {})
        
        # Quality Guardrail 2: If answer is under 35 chars and model gave > 6 score, adjust to partial/clarification
        if len(cleaned_answer) < 35 and name == "evaluate_answer":
            q_clean = question.replace('Explain ', '').replace('What is ', '').rstrip('?')
            return {
                "type": "NEEDS_CLARIFICATION",
                "follow_up_question": f"Your answer is quite brief. Could you explain in more detail how {q_clean} works?",
                "note": "Triggered clarification due to brief answer length."
            }

        if name == "evaluate_answer":
            return {
                "type": "EVALUATED",
                "verdict": args.get("verdict"),
                "score": args.get("score"),
                "feedback": args.get("feedback"),
            }
        elif name == "request_clarification":
            return {
                "type": "NEEDS_CLARIFICATION",
                "follow_up_question": args.get("follow_up_question"),
            }
        elif name == "mark_unanswered":
            return {
                "type": "UNANSWERED",
                "reason": args.get("reason"),
            }
        else:
            return {"type": "UNKNOWN", "raw": call}
            
    except Exception as e:
        return {"type": "ERROR", "raw_output": result_str, "error": str(e)}


if __name__ == "__main__":
    # Test cases
    samples = [
        ("What is object-oriented programming?", 
         "OOP is a programming paradigm based on the concept of objects which contain data and code."),
         
        ("Explain deadlock in operating systems.", 
         "It is some state in OS."),
         
        ("How does garbage collection work in Java?", 
         "I haven't worked with Java before, so I don't know."),
    ]
    
    print("\n" + "="*60)
    for q, a in samples:
        print(f"\nQuestion: {q}")
        print(f"Candidate Answer: {a}")
        res = evaluate(q, a)
        print("Evaluation Result:", json.dumps(res, indent=2))
    print("="*60)
