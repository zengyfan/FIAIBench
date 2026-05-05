"""
Generate per-model task books and result templates for sub-agent execution.
For the 7 external models that need to be evaluated externally.
"""
import json, os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_raw_results")

EXTERNAL_MODELS = [
    "MODEL_C",
    "MODEL_K",
    "MODEL_E",
    "MODEL_D",
    "MODEL_J",
    "MODEL_I",
    "MODEL_H"
]

# Count items
fae_items = json.load(open(os.path.join(OUTPUT_DIR, "fae_items.json"), "r", encoding="utf-8"))
aoa_items = json.load(open(os.path.join(OUTPUT_DIR, "aoa_items.json"), "r", encoding="utf-8"))

for model in EXTERNAL_MODELS:
    # Create empty result templates
    fae_template = {
        "experiment": "attribution_bench_v2",
        "model": model,
        "module": "fae",
        "timestamp": "PENDING",
        "parameters": {"temperature": 0, "max_tokens": 512},
        "total_items": len(fae_items),
        "total_repetitions": 1,
        "total_calls": 0,
        "results": [{
            "item_id": item["item_id"],
            "repetition": 0,
            "response": "PENDING",
            "attribution_type": "PENDING",
            "condition": item["condition"]
        } for item in fae_items],
        "errors": []
    }
    
    aoa_template = {
        "experiment": "attribution_bench_v2",
        "model": model,
        "module": "aoa",
        "timestamp": "PENDING",
        "parameters": {"temperature": 0, "max_tokens": 512},
        "total_items": len(aoa_items),
        "total_repetitions": 1,
        "total_calls": 0,
        "results": [{
            "item_id": item["item_id"],
            "repetition": 0,
            "response": "PENDING",
            "attribution_type": "PENDING",
            "explanation": "PENDING",
            "condition": item["condition"]
        } for item in aoa_items],
        "errors": []
    }
    
    with open(os.path.join(OUTPUT_DIR, f"{model}_fae.json"), "w", encoding="utf-8") as f:
        json.dump(fae_template, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, f"{model}_aoa.json"), "w", encoding="utf-8") as f:
        json.dump(aoa_template, f, ensure_ascii=False, indent=2)
    
    print(f"Created templates for {model}")

# Create summary.json with current status
with open(os.path.join(OUTPUT_DIR, "MODEL_A_fae.json"), "r", encoding="utf-8") as f:
    ds_fae = json.load(f)
with open(os.path.join(OUTPUT_DIR, "MODEL_A_aoa.json"), "r", encoding="utf-8") as f:
    ds_aoa = json.load(f)

summary = {
    "experiment": "attribution_bench_v2",
    "status": "partial",
    "completed_models": ["MODEL_A"],
    "pending_models": EXTERNAL_MODELS,
    "models_completed": 1,
    "models_total": 8,
    "data_status": {
        "MODEL_A": {"fae": "done", "aoa": "done"}
    },
    "instructions": f"Pending models need to execute phase4_task.md in sub-Agent or independent LLM pages. {len(EXTERNAL_MODELS)} models remaining."
}

for model in EXTERNAL_MODELS:
    summary["data_status"][model] = {"fae": "pending", "aoa": "pending"}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\nSummary created: {summary['models_completed']}/{summary['models_total']} models completed")
print(f"Output: {OUTPUT_DIR}")
