"""
Closed-source LLM ComEM-selecting baseline via OpenRouter, evaluated on a
stratified random subsample (default N=2000) of the 114k WDC LSPM eval set
for API-cost reasons.

Reuses:
  - candidates_top50.pkl from build_candidates_vanilla.py  (vanilla DeBERTa
    top-50 candidate pool, identical to all other LLM rerank baselines in
    Table 1)
  - ComEM SELECTING_INSTRUCTION verbatim from qwen36_baseline.py
"""
import os, sys, json, time, pickle, argparse, random
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from eval_utils import load_eval_samples, evaluate_predictions, save_results

SELECTING_INSTRUCTION = (
    "Select a record from the following candidates that refers to the same "
    "real-world entity as the given record. Answer with the corresponding "
    "record number surrounded by \"[]\" or \"[0]\" if there is none."
)

def build_selecting_prompt(cell_text, candidates):
    parts = [SELECTING_INSTRUCTION, "", f"Given entity record: {cell_text}", "",
             "Candidate records:"]
    for i, c in enumerate(candidates, 1):
        parts.append(f"[{i}] {c['name']}")
    return "\n".join(parts)


import re
_BRACKET = re.compile(r"\[(\d+)\]")
def parse_selecting_response(text, num_candidates):
    """Return 0-indexed candidate index, or None for abstain/parse-fail."""
    if not text: return None
    m = _BRACKET.search(text)
    if not m: return None
    idx = int(m.group(1))
    if idx == 0: return None  # abstain
    if 1 <= idx <= num_candidates: return idx - 1
    return None


def call_openrouter(api_key, model, prompt, max_tokens=20, timeout=60):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(5):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=timeout)
            if r.status_code == 429:  # rate limited
                time.sleep(2 ** attempt + random.random())
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return text, usage
        except Exception as e:
            if attempt == 4:
                return f"__ERR__ {type(e).__name__}: {str(e)[:80]}", {}
            time.sleep(1.5 ** attempt)
    return "__ERR__ exhausted", {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api_key", default=os.environ.get("OPENROUTER_API_KEY"),
                   help="OpenRouter API key. If not set, falls back to env OPENROUTER_API_KEY.")
    p.add_argument("--model", required=True, help="e.g. openai/gpt-5.4")
    p.add_argument("--candidates_pkl",
                   default="./experiments/kb_index/vanilla_deberta_cls/candidates_top50.pkl")
    p.add_argument("--n_subsample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    assert args.api_key, "need --api_key or env OPENROUTER_API_KEY"

    eval_samples = load_eval_samples()
    print(f"[eval] full size = {len(eval_samples)}")
    with open(args.candidates_pkl, "rb") as f:
        pkl = pickle.load(f)
    candidates_per_q = pkl["candidates_per_q"]
    id_to_name = pkl["kb_id_to_name"]
    assert len(candidates_per_q) == len(eval_samples), \
        f"len mismatch {len(candidates_per_q)} vs {len(eval_samples)}"

    # Stratified-random subsample (deterministic via seed)
    rng = random.Random(args.seed)
    idxs = list(range(len(eval_samples)))
    rng.shuffle(idxs)
    sub = sorted(idxs[:args.n_subsample])
    print(f"[sub] subsample N = {len(sub)} indices (seed={args.seed})")

    # Build prompts
    prompts = []
    for i in sub:
        cands = candidates_per_q[i][:args.top_k]
        prompts.append(build_selecting_prompt(eval_samples[i]["cell_text"], cands))
    print(f"[llm] model={args.model}, concurrency={args.concurrency}")

    # Concurrent calls
    results = [None] * len(prompts)
    usages = [None] * len(prompts)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(call_openrouter, args.api_key, args.model, p_, 20): k
                for k, p_ in enumerate(prompts)}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="API"):
            k = futs[fut]
            try:
                text, usage = fut.result()
            except Exception as e:
                text, usage = f"__ERR__ {e}", {}
            results[k] = text
            usages[k] = usage
    print(f"[llm] done in {(time.time()-t0):.1f}s")

    # Token totals + cost (best effort)
    in_tok = sum(u.get("prompt_tokens", 0) for u in usages if u)
    out_tok = sum(u.get("completion_tokens", 0) for u in usages if u)
    print(f"[usage] in={in_tok} tokens, out={out_tok} tokens")

    # Parse + assemble
    preds, pred_names, candidates_info, label_names = [], [], [], []
    golds = [str(eval_samples[i]["gold_entity_id"]) for i in sub]
    abstain = 0; err = 0
    for k, text in enumerate(results):
        i = sub[k]
        cands = candidates_per_q[i][:args.top_k]
        if text.startswith("__ERR__"):
            err += 1
            chosen = cands[0]
        else:
            j = parse_selecting_response(text, len(cands))
            if j is None:
                abstain += 1
                chosen = cands[0]
            else:
                chosen = cands[j]
        preds.append(chosen["id"])
        pred_names.append(chosen["name"])
        candidates_info.append(cands)
        label_names.append(id_to_name.get(golds[k], golds[k]))

    print(f"[parse] abstain={abstain}/{len(sub)}  errors={err}/{len(sub)}")

    metrics = evaluate_predictions(
        preds, golds,
        candidates_info=candidates_info,
        pred_names=pred_names, label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)

    # Persist raw outputs for debug / reuse
    out_dir = "./results"
    raw = {
        "model": args.model,
        "n_subsample": len(sub),
        "seed": args.seed,
        "subsample_indices": sub,
        "results": results,
        "usages": usages,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "metrics": metrics,
        "abstain": abstain,
        "errors": err,
    }
    safe_tag = args.tag.replace("/", "_").replace(" ", "_")
    with open(os.path.join(out_dir, f"openrouter_raw_{safe_tag}.json"), "w") as f:
        json.dump(raw, f, default=str)
    print(f"[done] {args.tag}")


if __name__ == "__main__":
    main()
