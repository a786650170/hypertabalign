"""
Inference / eval for the Steiner+ Qwen3-8B fine-tuned matcher on WDC LSPM.

Pipeline mirrors the ComEM (Qwen3-32B) baseline so that all numbers in Table 1
remain apples-to-apples:
  1. Re-use the cached vanilla DeBERTa CLS-pool KB index to retrieve top-K=50
     candidates per cell.
  2. For each (cell, candidate) pair, format the TailorMatch binary prompt
     used by Steiner+ et al. (Fine-tuning LLMs for Entity Matching,
     DAIS@ICDE 2025) verbatim, and score it under the fine-tuned LoRA-merged
     Qwen3-8B by computing P(``Yes'') - P(``No'') from a single forward pass
     at the answer position (teacher-forced scoring; no sampling).
  3. Re-rank the K=50 candidates per cell by this score, take the top-1 as
     the prediction, and report the standard Acc / MRR@10 / Hit@5 / Hit@10 /
     Name-Accuracy metrics through the project's Evaluator.

Re-uses the vanilla retriever for the bootstrap retrieval step (same upstream
as Vanilla DeBERTa / ComEM / Ditto) so that Hit@5 and Hit@10 in Table 1 are
upper-bounded by an identical candidate pool across all rerankers.
"""
import argparse
import json
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import transformers
from transformers.utils import import_utils
import_utils.check_torch_load_is_safe = lambda *a, **kw: None
transformers.modeling_utils.check_torch_load_is_safe = lambda *a, **kw: None

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from peft import PeftModel

from eval_utils import load_eval_samples, evaluate_predictions, save_results, KB_PATH


COLVAL_RE = re.compile(r"\[COL\]\s*([^\[]+?)\s*\[VAL\]\s*([^\[]*)")
PROMPT_TEMPLATE = (
    "Do the two product descriptions refer to the same real-world product? "
    "Entity 1: 'Entity 1'. Entity 2: 'Entity 2'."
)


def strip_colval(s: str) -> str:
    parts = COLVAL_RE.findall(s)
    if not parts:
        return s.strip()
    return " | ".join(f"{f.strip()}: {v.strip()}" for f, v in parts)


# ---------------------------------------------------------------------------
# Vanilla DeBERTa retrieval (shared with ComEM/Vanilla baselines)
# ---------------------------------------------------------------------------
def vanilla_retrieve(deberta_path, kb_cache, kb_path, eval_samples, top_k, device,
                     enc_bs=256, max_len=128):
    print(f"[retrieve] loading vanilla DeBERTa from {deberta_path}")
    tok = AutoTokenizer.from_pretrained(deberta_path)
    model = AutoModel.from_pretrained(deberta_path).to(device).eval()

    kb_entities = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    print(f"[retrieve] KB size = {len(kb_entities)}")

    if not os.path.exists(kb_cache):
        raise FileNotFoundError(f"vanilla KB cache missing: {kb_cache}")
    print(f"[retrieve] loading cached KB CLS embeddings: {kb_cache}")
    kb_embeds = torch.load(kb_cache, map_location="cpu").float()
    kb_embeds = F.normalize(kb_embeds, p=2, dim=-1).to(device)

    print(f"[retrieve] encoding {len(eval_samples)} query cells")
    cells = [s["cell_text"] for s in eval_samples]
    q_all = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in tqdm(range(0, len(cells), enc_bs), desc="encode-q"):
            batch = cells[i:i + enc_bs]
            inp = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len).to(device)
            out = model(**inp)
            cls = out.last_hidden_state[:, 0, :].float()
            q_all.append(F.normalize(cls, p=2, dim=-1).cpu())
    q_embeds = torch.cat(q_all, 0).to(device)

    print(f"[retrieve] top-{top_k} cosine search over {kb_embeds.size(0)} entities")
    candidates = []
    chunk = 128
    for i in tqdm(range(0, q_embeds.size(0), chunk), desc="topk"):
        q = q_embeds[i:i + chunk]
        scores = torch.matmul(q, kb_embeds.t())
        _, idx = torch.topk(scores, k=top_k, dim=-1)
        idx_cpu = idx.cpu().tolist()
        for j in range(q.size(0)):
            cands = [{"id": str(kb_entities[k]["id"]),
                      "name": kb_entities[k]["name"]} for k in idx_cpu[j]]
            candidates.append(cands)
    del model, kb_embeds, q_embeds
    torch.cuda.empty_cache()
    return candidates, kb_entities


# ---------------------------------------------------------------------------
# Steiner+ Yes/No teacher-forced scoring
# ---------------------------------------------------------------------------
def build_pair_prompt(cell_text, cand_name):
    return (PROMPT_TEMPLATE
            .replace("'Entity 1'", f"'{strip_colval(cell_text)}'")
            .replace("'Entity 2'", f"'{strip_colval(cand_name)}'"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",
                   default="./models_cache/Qwen3-8B")
    p.add_argument("--lora_path",
                   default="./checkpoints/steiner_qwen3_8b")
    p.add_argument("--deberta_path",
                   default="./models_cache/deberta-v3-base")
    p.add_argument("--kb_cache",
                   default="./experiments/kb_index/vanilla_deberta_cls/kb_index.pt")
    p.add_argument("--kb_path", default=KB_PATH)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--llm_batch_size", type=int, default=64)
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag",
                   default="Steiner+ Qwen3-8B (fine-tuned LLM rerank, K=50)")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    eval_samples = load_eval_samples()
    print(f"[eval] loaded {len(eval_samples)} samples")

    # ----- Retrieval (cached) -----
    candidates_per_q, kb_entities = vanilla_retrieve(
        args.deberta_path, args.kb_cache, args.kb_path,
        eval_samples, top_k=args.top_k, device=device,
    )

    # ----- Load Steiner+ Qwen3-8B + LoRA adapter -----
    print(f"\n[llm] loading Qwen3-8B + LoRA from {args.lora_path}")
    tok = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # left padding for batched generation/scoring
    tok.truncation_side = "left"  # CRITICAL: keep prompt SUFFIX so the last
                                  # token stays at the chat-template's
                                  # "</think>\n\n" position (Yes/No prediction
                                  # point); right-truncation silently lops it
                                  # off and yields all-zero rerank metrics.

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    llm = PeftModel.from_pretrained(base, args.lora_path)
    llm.eval()
    if hasattr(llm, "generation_config"):
        llm.generation_config.pad_token_id = tok.pad_token_id

    # Token IDs for Yes / No. The Qwen3 chat template (even with
    # enable_thinking=False) always emits "<think>\n\n</think>\n\n" before the
    # assistant payload; training thus targets the BARE "Yes" / "No" tokens
    # (no leading space) at position right after "</think>\n\n". Using
    # " Yes" / " No" with leading space matches a different vocab id and
    # produces near-uniform low logits -> degenerate rerank (observed as
    # all-zero metrics in the first run).
    yes_id = tok.encode("Yes", add_special_tokens=False)[-1]
    no_id  = tok.encode("No",  add_special_tokens=False)[-1]
    print(f"[llm] yes_id={yes_id} no_id={no_id}")

    # ----- Build (cell, candidate) prompts -----
    cells = [s["cell_text"] for s in eval_samples]
    labels = [s["gold_entity_id"] for s in eval_samples]

    flat_prompts = []
    flat_indices = []        # (query_idx, cand_pos_in_topk)
    for qi, cands in enumerate(candidates_per_q):
        for cj, cand in enumerate(cands):
            user_msg = build_pair_prompt(cells[qi], cand["name"])
            msgs = [{"role": "user", "content": user_msg}]
            try:
                ct = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                ct = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                )
            flat_prompts.append(ct)
            flat_indices.append((qi, cj))
    print(f"[llm] total (cell, candidate) pairs to score = {len(flat_prompts)}")

    # ----- Teacher-forced Yes vs No scoring at the next-token position -----
    print(f"[llm] scoring at bs={args.llm_batch_size}, max_len={args.max_input_len}")
    yes_minus_no = torch.zeros(len(flat_prompts), dtype=torch.float32)
    import time as _t
    t0 = _t.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(flat_prompts), args.llm_batch_size), desc="LLM"):
            batch = flat_prompts[i:i + args.llm_batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_input_len).to(device)
            out = llm(**enc)
            # Logits of the FINAL prompt token predict the FIRST generated token.
            last_logits = out.logits[:, -1, :]
            diff = (last_logits[:, yes_id] - last_logits[:, no_id]).float().cpu()
            yes_minus_no[i:i + len(batch)] = diff
    print(f"[llm] done in {(_t.time() - t0):.1f}s")

    # ----- Re-rank top-K by Yes-minus-No score; tie-break by original order -----
    preds, pred_names, label_names, candidates_info = [], [], [], []
    id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    cursor = 0
    for qi, cands in enumerate(candidates_per_q):
        K = len(cands)
        scores = yes_minus_no[cursor:cursor + K].tolist()
        cursor += K
        # higher score = more likely match.  Stable sort keeps retriever order on ties.
        order = sorted(range(K), key=lambda j: (-scores[j], j))
        reranked = [cands[j] for j in order]
        candidates_info.append(reranked)
        preds.append(reranked[0]["id"])
        pred_names.append(reranked[0]["name"])
        label_names.append(id_to_name.get(labels[qi], str(labels[qi])))

    metrics = evaluate_predictions(
        preds, labels,
        candidates_info=candidates_info,
        pred_names=pred_names, label_names=label_names,
        tag=args.tag,
    )
    save_results(metrics, args.tag)
    print(f"[done] {args.tag}")


if __name__ == "__main__":
    main()
