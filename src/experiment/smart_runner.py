"""
Smart experiment runner with checkpoint/resume.
Skips already-collected items to avoid duplicate API calls.
Merges new results with existing data per model.
"""
import json, os, sys, time, random
random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_runner import MODELS, create_client, call_model

RAW_DIR = "phase4_raw_results"
HARM_PATH = f"{RAW_DIR}/harm20k_new_only.json"

def load_existing(model_name):
    """Load existing results for a model, return {item_id: result_entry}."""
    safe_name = model_name.replace(".", "-").replace("/", "_")
    fname = f"{RAW_DIR}/{safe_name}_harm20k_new.json"
    if not os.path.exists(fname):
        return {}, None

    try:
        data = json.load(open(fname, encoding='utf-8-sig'))
    except:
        data = json.load(open(fname, encoding='utf-8'))

    existing = {}
    for r in data.get("results", []):
        if not r.get("error"):  # Only reuse successful results
            existing[r["item_id"]] = r
    return existing, data

def run_missing(model_name, all_items):
    """Run only items not already collected for this model."""
    existing, old_data = load_existing(model_name)
    existing_ids = set(existing.keys())
    all_ids = [item["item_id"] for item in all_items]

    missing_items = [item for item in all_items if item["item_id"] not in existing_ids]
    n_existing = len(existing_ids)
    n_missing = len(missing_items)

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"  Already collected: {n_existing} items")
    print(f"  Need to run: {n_missing} items")
    print(f"  Total target: {len(all_items)} items")
    print(f"{'='*60}")

    if n_missing == 0:
        print("  All items already collected! Merging existing data...")
        # Build output from existing
        results_list = list(existing.values())
        output = {
            "model": model_name,
            "model_code": MODELS[model_name]["model_code"],
            "provider": MODELS[model_name]["provider"],
            "module": "harm_20k",
            "timestamp": old_data.get("timestamp", "") if old_data else "",
            "total_items": len(results_list),
            "parameters": {"temperature": 0, "max_tokens": 5},
            "results": results_list,
            "errors": [r for r in results_list if r.get("error")],
            "stats": {"note": "merged from existing data, no new API calls"},
        }
        return output

    # Run missing items
    client, cfg = create_client(model_name)
    new_results = []
    errors = []
    total_tokens_p = 0
    total_tokens_c = 0
    t_start = time.time()

    consecutive_errors = 0
    for i, item in enumerate(missing_items):
        result = call_model(client, cfg, item)
        entry = {
            "item_id": item["item_id"],
            "response": result["response"],
            "latency_ms": result["latency_ms"],
            "tokens_used": {"prompt": result["tokens_prompt"], "completion": result["tokens_completion"]},
            "error": result["error"],
        }
        new_results.append(entry)
        if result["error"]:
            errors.append(entry)
            consecutive_errors += 1
        else:
            consecutive_errors = 0
        total_tokens_p += result["tokens_prompt"]
        total_tokens_c += result["tokens_completion"]

        # Abort if too many consecutive errors (quota exhausted, etc.)
        if consecutive_errors >= 20:
            print(f"  ABORT: {consecutive_errors} consecutive errors. Saving {len(new_results)} new items and stopping.", flush=True)
            break

        # Save checkpoint every 100 items (incremental, no data loss)
        if (i + 1) % 100 == 0 or (i + 1) == n_missing:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            eta = (n_missing - i - 1) / max(rate, 0.01)
            total = n_existing + i + 1
            print(f"  [{total}/{len(all_items)}] {100*total/len(all_items):.1f}% | "
                  f"{rate:.1f} items/s | ETA: {eta/60:.0f}min | "
                  f"Errors: {len(errors)} | Saving...", flush=True)
            # Incremental save: merge existing + current new results
            _all = list(existing.values()) + new_results
            _all.sort(key=lambda x: x["item_id"])
            _out = {
                "model": model_name, "model_code": MODELS[model_name]["model_code"],
                "provider": MODELS[model_name]["provider"], "module": "harm_20k",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "total_items": len(_all),
                "parameters": {"temperature": 0, "max_tokens": 5},
                "results": _all,
                "errors": [r for r in _all if r.get("error")],
                "stats": {"reused_items": n_existing, "new_items_so_far": len(new_results)},
            }
            _safe = model_name.replace(".", "-").replace("/", "_")
            with open(f"{RAW_DIR}/{_safe}_harm20k_new.json", 'w', encoding='utf-8') as _f:
                json.dump(_out, _f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start

    # Merge: existing + new
    all_results = list(existing.values()) + new_results
    # Keep consistent order by item_id
    all_results.sort(key=lambda x: x["item_id"])

    output = {
        "model": model_name,
        "model_code": MODELS[model_name]["model_code"],
        "provider": MODELS[model_name]["provider"],
        "module": "harm_20k",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_items": len(all_results),
        "parameters": {"temperature": 0, "max_tokens": 5},
        "results": all_results,
        "errors": [r for r in all_results if r.get("error")],
        "stats": {
            "total_time_seconds": round(elapsed, 1),
            "items_per_second": round(n_missing / max(elapsed, 1), 2),
            "total_tokens_prompt": total_tokens_p,
            "total_tokens_completion": total_tokens_c,
            "error_rate": round(len(errors) / max(n_missing, 1), 4),
            "reused_items": n_existing,
            "new_items": n_missing,
        },
    }
    return output

def save_results(model_name, output):
    """Save merged results."""
    safe_name = model_name.replace(".", "-").replace("/", "_")
    fname = f"{RAW_DIR}/{safe_name}_harm20k_new.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {fname} ({len(output['results'])} items)")

def main():
    items = json.load(open(HARM_PATH, encoding='utf-8'))
    print(f"Loaded {len(items)} items from {HARM_PATH}")

    models = ['deepseek-v4-flash', 'deepseek-v4-pro', 'glm-5.1', 'kimi-k2.6',
              'qwen3.6-35b-a3b']

    t_total = time.time()
    for i, model_name in enumerate(models):
        print(f"\n[Model {i+1}/{len(models)}]")
        try:
            output = run_missing(model_name, items)
            save_results(model_name, output)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"ALL DONE. Total time: {total_elapsed/3600:.1f}h")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
