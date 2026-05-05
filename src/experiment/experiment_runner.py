"""
AttributionBench: Universal LLM Experiment Runner
=================================================
Runs Harm20k benchmark on multiple LLMs via OpenAI-compatible API.

Usage:
  python experiment_runner.py --model MODEL_A --sample 500
  python experiment_runner.py --model all --full
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from openai import OpenAI

# ============================================================
# Model Registry (anonymized for double-blind review)
# ============================================================
MODELS = {
    "MODEL_A": {
        "model_code": "MODEL_A",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model A",
    },
    "MODEL_B": {
        "model_code": "MODEL_B",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model B",
    },
    "MODEL_C": {
        "model_code": "MODEL_C",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model C",
    },
    "MODEL_D": {
        "model_code": "MODEL_D",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model D",
    },
    "MODEL_E": {
        "model_code": "MODEL_E",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model E",
    },
    "MODEL_F": {
        "model_code": "MODEL_F",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model F",
    },
    "MODEL_G": {
        "model_code": "MODEL_G",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model G",
    },
    "MODEL_H": {
        "model_code": "MODEL_H",
        "provider": "PROVIDER_A",
        "base_url": "https://api.provider-a.example.com/v1",
        "env_key": "LLM_API_KEY",
        "description": "LLM Model H",
    },
}


def create_client(model_name):
    """Create OpenAI-compatible client for a model."""
    cfg = MODELS.get(model_name)
    if not cfg:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    api_key = os.getenv(cfg["env_key"])
    if not api_key:
        raise ValueError(
            f"API key not found for {model_name}.\n"
            f"  Required env var: {cfg['env_key']}\n"
            f"  Provider: {cfg['provider']}\n"
            f"  Set with: export {cfg['env_key']}=your_key"
        )

    return OpenAI(api_key=api_key, base_url=cfg["base_url"]), cfg


def call_model(client, cfg, item, system_prompt=None):
    """Call LLM with a single benchmark item. Returns response string."""
    user_prompt = item.get('user_prompt', item.get('framing', '') + '\n\n' + item.get('question', ''))
    if not user_prompt.strip():
        # Harm20k format: framing + question + options
        framing = item.get('framing', '')
        question = item.get('question', '基于以上信息，你的建议是？')
        options = item.get('options', [])
        options_text = '\n'.join(options)
        user_prompt = f"{framing}\n\n{question}\n\n{options_text}"

    messages = []
    sys_prompt = system_prompt or item.get('system_prompt', '')
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    # Force single-letter output for token efficiency
    messages.append({"role": "system", "content": "Output ONLY a single letter (A, B, C, D, or E). No explanation, no punctuation."})
    messages.append({"role": "user", "content": user_prompt})

    t_start = time.time()
    max_retries = 3
    for attempt in range(max_retries):
        try:
            create_kwargs = {
                "model": cfg["model_code"],
                "messages": messages,
                "temperature": 0,
                "max_tokens": 5,
            }
            if not cfg.get("no_extra_body"):
                create_kwargs["extra_body"] = {"enable_thinking": False}
            completion = client.chat.completions.create(**create_kwargs)
            content = completion.choices[0].message.content
            usage = completion.usage
            latency = int((time.time() - t_start) * 1000)
            return {
                "response": content.strip() if content else "",
                "latency_ms": latency,
                "tokens_prompt": usage.prompt_tokens if usage else 0,
                "tokens_completion": usage.completion_tokens if usage else 0,
                "error": None,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/{max_retries} in {wait}s: {str(e)[:80]}")
                time.sleep(wait)
            else:
                return {
                    "response": "ERROR",
                    "latency_ms": 0,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "error": str(e)[:200],
                }


def run_benchmark(model_name, harm_items, sample_size=None, items_to_run=None,
                  start_from=0, output_dir="autoresearch_output/phase4_raw_results"):
    """Run benchmark for a single model."""
    client, cfg = create_client(model_name)
    model_code = cfg["model_code"]

    print(f"\n{'='*60}")
    print(f"Model: {model_name} ({cfg['description']})")
    print(f"Provider: {cfg['provider']}")
    print(f"Model code: {model_code}")
    print(f"{'='*60}")

    # Select items to run
    if items_to_run is not None:
        run_items = items_to_run
    else:
        run_items = harm_items

    if sample_size and sample_size < len(run_items):
        import random
        random.seed(42)
        run_items = random.sample(run_items, sample_size)

    run_items = run_items[start_from:]
    total = len(run_items)
    print(f"Items to run: {total}")

    results = []
    errors = []
    total_tokens_prompt = 0
    total_tokens_completion = 0
    t_start = time.time()

    for i, item in enumerate(run_items):
        result = call_model(client, cfg, item)

        entry = {
            "item_id": item["item_id"],
            "response": result["response"],
            "latency_ms": result["latency_ms"],
            "tokens_used": {
                "prompt": result["tokens_prompt"],
                "completion": result["tokens_completion"],
            },
            "error": result["error"],
        }
        results.append(entry)

        if result["error"]:
            errors.append(entry)

        total_tokens_prompt += result["tokens_prompt"]
        total_tokens_completion += result["tokens_completion"]

        # Progress
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1)
            eta = (total - i - 1) / max(rate, 0.01)
            print(f"  [{i+1}/{total}] {100*(i+1)/total:.1f}% | "
                  f"{rate:.1f} items/s | ETA: {eta:.0f}s | "
                  f"Errors: {len(errors)}")

    elapsed = time.time() - t_start

    # Build output
    output = {
        "model": model_name,
        "model_code": model_code,
        "provider": cfg["provider"],
        "module": "harm_20k",
        "timestamp": datetime.now().isoformat(),
        "total_items": total,
        "parameters": {"temperature": 0, "max_tokens": 5},
        "results": results,
        "errors": [e for e in results if e["error"]],
        "stats": {
            "total_time_seconds": round(elapsed, 1),
            "items_per_second": round(total / max(elapsed, 1), 2),
            "total_tokens_prompt": total_tokens_prompt,
            "total_tokens_completion": total_tokens_completion,
            "error_rate": round(len(errors) / max(total, 1), 4),
        },
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.replace(".", "-").replace("/", "_")
    out_path = os.path.join(output_dir, f"{safe_name}_harm20k.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"  Items: {total}, Errors: {len(errors)} ({output['stats']['error_rate']:.2%})")
    print(f"  Time: {elapsed:.0f}s ({output['stats']['items_per_second']} items/s)")
    print(f"  Tokens: {total_tokens_prompt:,} prompt + {total_tokens_completion:,} completion")

    return output


def main():
    parser = argparse.ArgumentParser(description="AttributionBench Experiment Runner")
    parser.add_argument("--model", type=str, default="MODEL_J",
                        help=f"Model name or 'all'. Available: {list(MODELS.keys())}")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample N items (for quick test). Default: run all.")
    parser.add_argument("--full", action="store_true",
                        help="Run all items (no sampling)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Resume from item index N")
    parser.add_argument("--items-file", type=str,
                        default="autoresearch_output/phase4_raw_results/harm_20k.json",
                        help="Path to benchmark items JSON")
    parser.add_argument("--output-dir", type=str,
                        default="autoresearch_output/phase4_raw_results",
                        help="Output directory for results")
    parser.add_argument("--list-models", action="store_true",
                        help="List available models and exit")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for name, cfg in MODELS.items():
            env_set = "✓" if os.getenv(cfg["env_key"]) else "✗"
            print(f"  {env_set} {name:25s} | {cfg['provider']:10s} | {cfg['description']}")
        return

    # Load items
    items = json.load(open(args.items_file, encoding='utf-8'))
    print(f"Loaded {len(items)} items from {args.items_file}")

    # Determine sample size
    sample_size = args.sample
    if args.full:
        sample_size = None

    # Determine models to run
    if args.model == "all":
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = [args.model]

    for model_name in models_to_run:
        # Check API key
        cfg = MODELS[model_name]
        if not os.getenv(cfg["env_key"]):
            print(f"\n⚠ Skipping {model_name}: {cfg['env_key']} not set")
            continue

        try:
            run_benchmark(
                model_name=model_name,
                harm_items=items,
                sample_size=sample_size,
                start_from=args.start_from,
                output_dir=args.output_dir,
            )
        except Exception as e:
            print(f"\n✗ {model_name} FAILED: {e}")
            continue

    print("\nDone.")


if __name__ == "__main__":
    main()
