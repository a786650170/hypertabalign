"""
LLM Zero-shot Entity Alignment Evaluation.
Uses DeBERTa dense retrieval to pre-retrieve candidates, then prompts an LLM to select the best match.
Supports any OpenAI-compatible API (DeepSeek, Qwen3, etc.)
"""
import argparse
import os
import sys
import json
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval_utils import load_eval_samples, load_kb, evaluate_predictions, save_results, KB_PATH


PROMPT_TEMPLATE = """You are an entity matching expert. Given a product mention from a web table, select the most likely matching entity from the candidate list.

Product mention: "{cell_text}"
Row context: "{row_context}"

Candidate entities:
{candidates_block}

Instructions:
- Output ONLY the number (1-{n_candidates}) of the best matching entity.
- If none match, output 0.
- Do not explain, just output the number."""


def build_candidates_block(candidates):
    lines = []
    for i, c in enumerate(candidates):
        lines.append(f"{i + 1}. {c['name']} (ID: {c['id']})")
    return "\n".join(lines)


def call_llm_api(prompt, api_url, api_key, model_name, timeout=30):
    """Call an OpenAI-compatible chat API."""
    import requests
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16,
    }

    try:
        resp = requests.post(
            f"{api_url}/chat/completions",
            headers=headers, json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return content
    except Exception as e:
        return f"ERROR: {e}"


def parse_llm_response(response, candidates):
    """Extract selected candidate index from LLM response."""
    nums = re.findall(r"\d+", response)
    if not nums:
        return None
    idx = int(nums[0])
    if idx == 0:
        return None
    if 1 <= idx <= len(candidates):
        return idx - 1
    return None


def main():
    parser = argparse.ArgumentParser(description="LLM Zero-shot Entity Alignment")
    parser.add_argument("--api_url", type=str, required=True,
                        help="OpenAI-compatible API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--api_key", type=str, default="",
                        help="API key (if required)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name for the API (e.g. deepseek-chat, qwen3-72b)")
    parser.add_argument("--tag", type=str, default="",
                        help="Display name for results (default: model_name)")
    parser.add_argument("--num_candidates", type=int, default=10,
                        help="Number of candidates to present to the LLM")
    parser.add_argument("--retrieval_top_k", type=int, default=50,
                        help="Dense retrieval top-K before selecting best N for prompt")
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--sample_size", type=int, default=5000,
                        help="Number of eval samples to use (0=all, default=5000)")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Parallel API calls")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tag = args.tag or args.model_name

    print("=" * 60)
    print(f"LLM Zero-shot Baseline: {tag}")
    print("=" * 60)

    samples = load_eval_samples()

    if args.sample_size > 0 and args.sample_size < len(samples):
        random.seed(args.seed)
        samples = random.sample(samples, args.sample_size)
        print(f"[LLM] Sampled {len(samples)} eval examples (seed={args.seed})")

    print("\n[1/4] Building vanilla DeBERTa dense retrieval index...")
    from dense_retriever import DenseRetriever
    retriever = DenseRetriever(
        device="cuda:0" if __import__('torch').cuda.is_available() else "cpu",
    )
    retriever.build_or_load_index(args.kb_path)

    print(f"\n[2/4] Encoding {len(samples)} query texts...")
    cell_texts = [s["cell_text"] for s in samples]
    query_embeds = retriever.encode_queries(cell_texts)

    print(f"\n[3/4] Dense retrieval + LLM prompting...")
    all_candidates = retriever.retrieve(query_embeds, top_k=args.retrieval_top_k)

    id_to_name = {str(e["id"]): e["name"] for e in retriever.kb_entities}

    tasks = []
    for i, s in enumerate(samples):
        candidates = all_candidates[i][:args.num_candidates]

        prompt = PROMPT_TEMPLATE.format(
            cell_text=s["cell_text"],
            row_context=s["row_context"],
            candidates_block=build_candidates_block(candidates),
            n_candidates=len(candidates),
        )
        tasks.append((s, prompt, candidates))

    preds, labels, pred_names, label_names = [], [], [], []
    completed = 0

    def process_one(item):
        s, prompt, candidates = item
        response = call_llm_api(prompt, args.api_url, args.api_key, args.model_name)
        selected = parse_llm_response(response, candidates)
        if selected is not None:
            pred_id = candidates[selected]["id"]
            pred_name = candidates[selected]["name"]
        else:
            pred_id = candidates[0]["id"] if candidates else "NIL"
            pred_name = candidates[0]["name"] if candidates else "NIL"
        return s, pred_id, pred_name

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for future in as_completed(futures):
            s, pred_id, pred_name = future.result()
            preds.append(pred_id)
            pred_names.append(pred_name)
            labels.append(s["gold_entity_id"])
            label_names.append(id_to_name.get(s["gold_entity_id"],
                                               str(s["gold_entity_id"])))
            completed += 1
            if completed % 500 == 0:
                print(f"  {completed}/{len(tasks)} completed")

    print(f"\n[4/4] Evaluating...")
    metrics = evaluate_predictions(
        preds, labels,
        pred_names=pred_names,
        label_names=label_names,
        tag=f"{tag} (zero-shot)",
    )
    save_results(metrics, f"{tag} (zero-shot)")


if __name__ == "__main__":
    main()
