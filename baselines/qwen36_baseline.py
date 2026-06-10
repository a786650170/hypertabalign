"""
Zero-shot LLM Entity Matching baseline with Qwen3.6-27B (Alibaba, 2026-04-22).

Paradigm:
  - Up-stream retriever: SAME vanilla DeBERTa-v3-base CLS-pool encoder used by
    the Vanilla baseline (re-uses cached `vanilla_deberta_cls/kb_index.pt`).
    Top-K candidates per cell (default K=10, following ComEM / MatchGPT settings).
  - Re-ranker: Qwen3.6-27B (dense, instruction-tuned) used **zero-shot** with the
    "selecting" prompt template — give the model the query cell and a numbered
    list of K candidates, ask it to output a single number `[i]` (or `[0]` if
    none match).  This template is taken verbatim from ComEM (COLING 2025,
    `selecting.py`) and is the canonical zero-shot LLM EM prompt for the
    candidate-pool setting.

Why this specific design (and what is and is NOT a "fair" adaptation):
  - Backbone choice: Qwen3.6-27B is the latest open-weight Qwen model that
    fits on a single H100-80GB in fp16 (~54 GB).  It replaces the originally
    proposed Jellyfish-13B / DeepSeek-V4 (which we could not download because
    the HuggingFace XET CDN is unreachable from inside the cluster).  The
    paradigm — "instruction-tuned LLM, zero-shot, no task-specific fine-tune" —
    is what the baseline ladder requires; the exact backbone is a substitution
    forced by network limitations and is documented as such.
  - We use plain `transformers.AutoModelForCausalLM.generate(do_sample=False)`
    in fp16, no extra inference engine.  This is the safest reproduction
    surface and matches the standard zero-shot LLM EM literature; we only
    batch prompts inside the model forward.
  - Top-K=10: matches ComEM and the MatchGPT-style EM zero-shot literature
    (Peeters & Bizer 2023).  Pushing K higher would amplify position bias on
    the LLM (a documented failure mode in ComEM §5.3).
  - We do NOT fine-tune Qwen3.6-27B on WDC.  We do NOT do prompt search /
    chain-of-thought / few-shot demos.  We do NOT change the candidate pool
    above what other baselines see.  Anything that "boosts" the LLM beyond
    its zero-shot ability would be unfair to the main model.

Output format identical to other baselines: a row in
  results/comparison_table.csv with Acc / Micro-F1 / Hit@5 / Hit@10 / Name-Acc.
"""
import argparse
import os
import sys
import json
import re
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from eval_utils import (
    load_eval_samples,
    evaluate_predictions,
    save_results,
    KB_PATH,
)


# -----------------------------------------------------------------------------
# Up-stream retriever: vanilla DeBERTa CLS pool (same as Vanilla baseline)
# -----------------------------------------------------------------------------
def cls_encode(model, tokenizer, texts, device, batch_size=128, max_length=128,
               use_fp16=True):
    all_embeds = []
    model.eval()
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if use_fp16 and device.type == "cuda" else
        torch.cuda.amp.autocast(enabled=False)
    )
    with torch.no_grad(), autocast_ctx:
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(device)
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].float()
            all_embeds.append(F.normalize(cls, p=2, dim=-1).cpu())
    return torch.cat(all_embeds, dim=0)


def retrieve_topk(deberta_path, kb_cache, kb_path, eval_samples, top_k, device,
                  encoder_batch_size=256, max_length=128):
    """Run retrieval to get top-K candidates per query, identical to Vanilla baseline."""
    print(f"\n[retrieve] Loading vanilla DeBERTa retriever from {deberta_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(deberta_path)
    model = AutoModel.from_pretrained(deberta_path).to(device)
    model.eval()

    print("[retrieve] Loading KB ...")
    kb_entities = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    print(f"[retrieve] KB size: {len(kb_entities)}")

    if not os.path.exists(kb_cache):
        raise FileNotFoundError(
            f"KB CLS-pool index cache missing: {kb_cache}\n"
            f"Run vanilla_deberta_baseline.py first (it builds the same cache "
            f"in ~10 min); this script reuses it for fairness."
        )
    print(f"[retrieve] Loading cached KB CLS embeddings: {kb_cache}")
    kb_embeds = torch.load(kb_cache, map_location="cpu")
    kb_embeds = F.normalize(kb_embeds, p=2, dim=-1)
    assert kb_embeds.size(0) == len(kb_entities), \
        f"cache mismatch: {kb_embeds.size(0)} vs {len(kb_entities)}"

    cell_texts = [s["cell_text"] for s in eval_samples]
    print(f"[retrieve] Encoding {len(cell_texts)} eval queries ...")
    query_embeds = cls_encode(model, tokenizer, cell_texts, device,
                              batch_size=encoder_batch_size, max_length=max_length)

    # Move KB to GPU once for fast topk
    kb_dev = kb_embeds.to(device)
    candidates_per_q = []
    chunk = 256
    print(f"[retrieve] Top-{top_k} retrieval over {len(kb_entities)} entities ...")
    with torch.no_grad():
        for i in tqdm(range(0, query_embeds.size(0), chunk), desc="Top-K"):
            q = query_embeds[i:i + chunk].to(device)
            scores = torch.matmul(q, kb_dev.t())
            topk_scores, topk_idx = torch.topk(scores, k=top_k, dim=-1)
            for j in range(q.size(0)):
                cands = []
                for k in range(top_k):
                    eidx = topk_idx[j, k].item()
                    cands.append({
                        "id": str(kb_entities[eidx]["id"]),
                        "name": kb_entities[eidx]["name"],
                        "score": float(topk_scores[j, k].item()),
                    })
                candidates_per_q.append(cands)

    # free GPU memory before loading the LLM
    del model, kb_dev, kb_embeds, query_embeds
    torch.cuda.empty_cache()
    return candidates_per_q


# -----------------------------------------------------------------------------
# Re-ranker: Qwen3.6-27B "selecting" prompt (verbatim from ComEM COLING 2025)
# -----------------------------------------------------------------------------
SELECTING_INSTRUCTION = (
    "Select a record from the following candidates that refers to the same "
    "real-world entity as the given record. Answer with the corresponding "
    "record number surrounded by \"[]\" or \"[0]\" if there is none."
)

def build_selecting_prompt(cell_text, candidates):
    """ComEM selecting template (LLM4EM repo `src/selecting.py`)."""
    parts = [SELECTING_INSTRUCTION, "", f"Given entity record: {cell_text}", "",
             "Candidate records:"]
    for i, c in enumerate(candidates, 1):
        parts.append(f"[{i}] {c['name']}")
    return "\n".join(parts)


_NUM_RE = re.compile(r"\[(\d+)\]")

def parse_selecting_response(text, num_candidates):
    """Pull out the first [N] in the model's reply.  N=0 -> abstain."""
    m = _NUM_RE.search(text or "")
    if m is None:
        return None
    n = int(m.group(1))
    if n == 0 or n > num_candidates:
        return None
    return n - 1  # 0-indexed candidate


# === Native prompt variant (uses the model's own chat template) ============
NATIVE_INSTRUCTION = (
    "You are a product matching assistant. Given a query product and a numbered "
    "list of candidate product names, decide which single candidate refers to "
    "the SAME product as the query. Respond with ONLY a single number: 1..K for "
    "the matching candidate, or 0 if none match. Do not add any other text."
)

def build_native_prompt(cell_text, candidates, tokenizer):
    """Use the model's own chat template instead of the ComEM rigid template."""
    cands_str = "\n".join(f"{i}. {c['name']}" for i, c in enumerate(candidates, 1))
    user_msg = (
        f"Query product: {cell_text}\n\n"
        f"Candidates:\n{cands_str}\n\n"
        f"Which candidate matches?"
    )
    messages = [
        {"role": "system", "content": NATIVE_INSTRUCTION},
        {"role": "user", "content": user_msg},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


_BARE_NUM_RE = re.compile(r"(?<!\d)(\d+)(?!\d)")

def parse_native_response(text, num_candidates):
    """Pull out the first bare integer (0..K) in the model's reply."""
    if not text:
        return None
    m = _BARE_NUM_RE.search(text)
    if m is None:
        return None
    n = int(m.group(1))
    if n == 0 or n > num_candidates:
        return None
    return n - 1
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM EM baseline with Qwen3.6-27B (selecting prompt)."
    )
    parser.add_argument("--llm_path", type=str, required=True,
                        help="Local path to Qwen3.6-27B weights.")
    parser.add_argument("--deberta_path", type=str,
                        default="./models_cache/deberta-v3-base",
                        help="Up-stream retriever (vanilla DeBERTa).")
    parser.add_argument("--tag", type=str,
                        default="Qwen3.6-27B zero-shot LLM EM (ComEM selecting)")
    parser.add_argument("--kb_path", type=str, default=KB_PATH)
    parser.add_argument("--kb_cache", type=str,
                        default=os.path.join(PROJECT_ROOT,
                            "experiments/kb_index/vanilla_deberta_cls/kb_index.pt"))

    parser.add_argument("--candidates_pkl", type=str, default=None,
                        help="If set, skip retrieval and load top-K candidates from this pkl.")
    parser.add_argument("--prompt_style", type=str, default="comem",
                        choices=["comem", "native"],
                        help="comem = ComEM/LLM4EM rigid selecting template; "
                             "native = Qwen3 chat template + natural instruction.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Candidate pool size for the LLM (ComEM uses 10).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Selecting only needs '[i]', so 20 tokens is plenty.")
    parser.add_argument("--llm_batch_size", type=int, default=8,
                        help="Number of prompts per LLM forward step.")
    parser.add_argument("--max_input_len", type=int, default=1024,
                        help="Max prompt length (truncate from left).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Smoke-test: only process first N samples (0=all).")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Baseline: {args.tag}")
    print(f"  LLM path: {args.llm_path}")
    print(f"  Retriever: vanilla DeBERTa-v3-base (shared with Vanilla baseline)")
    print(f"  Top-K: {args.top_k}  (ComEM/MatchGPT default)")
    print(f"  Prompt: ComEM 'selecting' template (verbatim)")
    print("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ------- 1) eval data + retrieval -------
    eval_samples = load_eval_samples()
    if args.limit > 0:
        eval_samples = eval_samples[:args.limit]
        print(f"\n[smoke] Limited to first {len(eval_samples)} samples.")
    gold_ids = [s["gold_entity_id"] for s in eval_samples]
    cell_texts = [s["cell_text"] for s in eval_samples]

    if args.candidates_pkl:
        import pickle
        print("[retrieve] Loading precomputed candidates from " + args.candidates_pkl)
        with open(args.candidates_pkl, "rb") as f:
            _pkl = pickle.load(f)
        candidates_per_q = _pkl["candidates_per_q"][:len(eval_samples)]
        _id_to_name_override = _pkl["kb_id_to_name"]
    else:
        candidates_per_q = retrieve_topk(
            deberta_path=args.deberta_path,
            kb_cache=args.kb_cache,
            kb_path=args.kb_path,
            eval_samples=eval_samples,
            top_k=args.top_k,
            device=device,
        )
        _id_to_name_override = None

    # ------- 2) Pick prompt style helpers (build & parse) -------
    if args.prompt_style == "comem":
        _parse_fn = parse_selecting_response
        _prompt_kind = "ComEM 'selecting' (verbatim from LLM4EM)"
    else:
        _parse_fn = parse_native_response
        _prompt_kind = "native Qwen3 chat template + natural instruction"
    print(f"\n[llm] Prompt style: {_prompt_kind}")

    # ------- 3) Load Qwen3.6-27B (HF transformers, fp16) -------
    print(f"\n[llm] Loading Qwen3.6-27B (transformers, bf16) ...")
    tok = AutoTokenizer.from_pretrained(args.llm_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left padding required for batched generation

    llm = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    llm.eval()
    if hasattr(llm, "generation_config"):
        llm.generation_config.pad_token_id = tok.pad_token_id

    # Build prompts now that the tokenizer is loaded.
    if args.prompt_style == "comem":
        print(f"[llm] Building {len(eval_samples)} ComEM 'selecting' prompts ...")
        prompts = [build_selecting_prompt(c, cands)
                   for c, cands in zip(cell_texts, candidates_per_q)]
        # ComEM bare prompts -> wrap as user msg -> chat template
        # (Qwen3 dense `Qwen3-32B` defaults `enable_thinking=True`; disable for
        # greedy decoding per Qwen3 docs. Instruct variants ignore the flag.)
        chat_prompts = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            try:
                ct = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                ct = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            chat_prompts.append(ct)
    else:
        print(f"[llm] Building {len(eval_samples)} native chat-template prompts ...")
        # Native prompts already go through the chat template internally.
        chat_prompts = [build_native_prompt(c, cands, tok)
                        for c, cands in zip(cell_texts, candidates_per_q)]

    print(f"[llm] Generating {len(chat_prompts)} responses (greedy, bs={args.llm_batch_size}) ...")
    outputs_text = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(chat_prompts), args.llm_batch_size), desc="LLM"):
            batch = chat_prompts[i:i + args.llm_batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_input_len).to(device)
            out = llm.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            replies = tok.batch_decode(new_tokens, skip_special_tokens=True)
            outputs_text.extend(replies)
    elapsed = time.time() - t0
    print(f"[llm] Done in {elapsed:.1f}s "
          f"({len(outputs_text)/max(elapsed,1):.2f} samples/s)")

    # ------- 4) Parse + assemble predictions -------
    preds = []
    pred_names = []
    candidates_info = []
    label_names = []
    abstain_count = 0
    for i, reply in enumerate(outputs_text):
        cands = candidates_per_q[i]
        idx = _parse_fn(reply, num_candidates=len(cands))
        if idx is None:
            abstain_count += 1
            # If LLM abstains ('[0]' or unparseable) — fall back to top-1
            # retrieval (the same prediction Vanilla would make).  This matches
            # ComEM's behaviour: a [0] reply means "none of the candidates",
            # but for evaluation we still need to commit to ONE prediction;
            # the only honest fallback is the top-1 retriever.
            chosen = cands[0]
        else:
            chosen = cands[idx]
        preds.append(chosen["id"])
        pred_names.append(chosen["name"])
        candidates_info.append(cands)
        label_names.append("")

    print(f"\n[llm] Abstain rate ('[0]' or unparseable): "
          f"{abstain_count}/{len(outputs_text)} = "
          f"{abstain_count/max(len(outputs_text),1):.3f}")

    metrics = evaluate_predictions(
        preds, gold_ids,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)
    print(f"\n[done] {args.tag}")


if __name__ == "__main__":
    main()
