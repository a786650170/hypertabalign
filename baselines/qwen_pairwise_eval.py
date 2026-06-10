"""
Zero-shot pairwise Yes/No reranking with any Qwen3-family LLM
(NO LoRA, NO fine-tuning). Same prompt as Steiner+/TailorMatch for a clean
'fine-tuned vs zero-shot pairwise' contrast in Table 1.

Pipeline identical to steiner_qwen3_8b_eval.py:
  1. Vanilla DeBERTa cached CLS top-K=50 retrieval (shared with all baselines)
  2. For each (cell, candidate) build the TailorMatch binary prompt; score
     under the LLM by logits["Yes"] - logits["No"] at the assistant position
     (Qwen3 chat template prefixes "<think>\\n\\n</think>\\n\\n" then the
     answer; bare-word "Yes"/"No" token IDs apply).
  3. Re-rank top-K by that score; emit Acc/MRR@10/Hit@5/Hit@10/Name-Acc.
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


def vanilla_retrieve(deberta_path, kb_cache, kb_path, eval_samples, top_k, device,
                     enc_bs=256, max_len=128):
    """Legacy raw-DeBERTa-CLS retriever. Broken on transformers>=5.8 (CLS
    collapse); kept only for back-compat. Prefer mpnet_retrieve."""
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
    kb_embeds = torch.load(kb_cache, map_location="cpu").float()
    kb_embeds = F.normalize(kb_embeds, p=2, dim=-1).to(device)

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


def mpnet_retrieve(mpnet_path, kb_cache, kb_path, eval_samples, top_k, device,
                   enc_bs=256):
    """MPNet (sentence-transformers/all-mpnet-base-v2) base retriever.
    Used because raw DeBERTa CLS pool collapsed under transformers>=5.8."""
    from sentence_transformers import SentenceTransformer
    print(f"[retrieve] loading MPNet from {mpnet_path}")
    model = SentenceTransformer(mpnet_path, device=str(device))
    model.eval()

    kb_entities = []
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in f:
            kb_entities.append(json.loads(line))
    print(f"[retrieve] KB size = {len(kb_entities)}")

    if not os.path.exists(kb_cache):
        raise FileNotFoundError(f"MPNet KB cache missing: {kb_cache}")
    kb_embeds = torch.load(kb_cache, map_location="cpu").float()
    kb_embeds = F.normalize(kb_embeds, p=2, dim=-1).to(device)

    cells = [s["cell_text"] for s in eval_samples]
    print(f"[retrieve] encoding {len(cells)} queries with MPNet")
    q_embeds = model.encode(cells, batch_size=enc_bs,
                            show_progress_bar=True,
                            convert_to_tensor=True,
                            normalize_embeddings=True).to(device).float()

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


def build_pair_prompt(cell_text, cand_name):
    return (PROMPT_TEMPLATE
            .replace("'Entity 1'", f"'{strip_colval(cell_text)}'")
            .replace("'Entity 2'", f"'{strip_colval(cand_name)}'"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Path to base LLM.")
    p.add_argument("--lora_path", default=None,
                   help="Optional LoRA adapter (e.g. Steiner+ fine-tuned).")
    p.add_argument("--retriever", default="vanilla_cls",
                   choices=["mpnet", "vanilla_cls", "precomputed"],
                   help="Base retriever for top-K candidates. 'precomputed' "
                        "reads from --candidates_pkl (built offline in env "
                        "tyf1 / transformers 5.0, restoring the original "
                        "vanilla DeBERTa CLS protocol used in the paper).")
    p.add_argument("--candidates_pkl",
                   default="./experiments/kb_index/vanilla_deberta_cls/candidates_top50.pkl",
                   help="Precomputed candidates from build_candidates_vanilla.py")
    p.add_argument("--retriever_path",
                   default="./models_cache/_modelscope_cache/"
                           "sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--kb_cache",
                   default="./experiments/kb_index/mpnet/kb_index.pt")
    p.add_argument("--kb_path", default=KB_PATH)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--llm_batch_size", type=int, default=32)
    p.add_argument("--max_input_len", type=int, default=224)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=0,
                   help="0=use all eval samples; >0=quick smoke test on N samples")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    eval_samples = load_eval_samples()
    if args.limit > 0:
        eval_samples = eval_samples[:args.limit]
    print(f"[eval] loaded {len(eval_samples)} samples")

    if args.retriever == "precomputed":
        import pickle
        print(f"[retrieve] loading precomputed candidates {args.candidates_pkl}")
        with open(args.candidates_pkl, "rb") as f:
            pkl = pickle.load(f)
        candidates_per_q = pkl["candidates_per_q"][:len(eval_samples)]
        # Reconstruct minimal kb_entities-like list for evaluator (only needs id->name).
        # Use a lightweight wrapper: list of dicts; id_to_name will be built later.
        id_to_name = pkl["kb_id_to_name"]
        # Build a dummy kb_entities matching the API of downstream code (we
        # only access kb_entities[k]['id'] / ['name'] via candidate dicts, so
        # passing an empty list is OK as long as id_to_name is supplied).
        kb_entities = pkl  # keep ref so id_to_name accessible below
    elif args.retriever == "mpnet":
        candidates_per_q, kb_entities = mpnet_retrieve(
            args.retriever_path, args.kb_cache, args.kb_path,
            eval_samples, top_k=args.top_k, device=device,
        )
    else:
        candidates_per_q, kb_entities = vanilla_retrieve(
            args.retriever_path, args.kb_cache, args.kb_path,
            eval_samples, top_k=args.top_k, device=device,
        )

    print(f"\n[llm] loading {args.model_path}" + (f" + LoRA {args.lora_path}" if args.lora_path else ""))
    tok_path = args.lora_path if args.lora_path else args.model_path
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"  # keep prompt SUFFIX so last token = Yes/No
                                  # prediction position (else metrics are 0).

    llm = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    if args.lora_path:
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, args.lora_path)
    llm.eval()
    if hasattr(llm, "generation_config"):
        llm.generation_config.pad_token_id = tok.pad_token_id

    yes_id = tok.encode("Yes", add_special_tokens=False)[-1]
    no_id  = tok.encode("No",  add_special_tokens=False)[-1]
    print(f"[llm] yes_id={yes_id} no_id={no_id}")

    cells = [s["cell_text"] for s in eval_samples]
    labels = [s["gold_entity_id"] for s in eval_samples]

    # Detect whether tokenizer has a chat template; Jellyfish-13B (Llama-2
    # base, Alpaca-style instruction-tuned) ships WITHOUT one, so fall back
    # to the Alpaca prompt format used in the Jellyfish paper/README.
    has_chat_tmpl = getattr(tok, "chat_template", None) is not None
    ALPACA = ("You are an AI assistant that follows instruction extremely well. "
              "Help as much as you can.\n\n### Instruction:\n{instr}\n\n### Response:\n")
    flat_prompts = []
    for qi, cands in enumerate(candidates_per_q):
        for cj, cand in enumerate(cands):
            user_msg = build_pair_prompt(cells[qi], cand["name"])
            if has_chat_tmpl:
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
            else:
                ct = ALPACA.format(instr=user_msg)
            flat_prompts.append(ct)
    print(f"[llm] total pairs = {len(flat_prompts)}")

    print(f"[llm] scoring bs={args.llm_batch_size}, max_len={args.max_input_len}")
    yes_minus_no = torch.zeros(len(flat_prompts), dtype=torch.float32)
    import time as _t
    t0 = _t.time()
    with torch.no_grad():
        for i in tqdm(range(0, len(flat_prompts), args.llm_batch_size), desc="LLM"):
            batch = flat_prompts[i:i + args.llm_batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_input_len).to(device)
            out = llm(**enc)
            last = out.logits[:, -1, :]
            diff = (last[:, yes_id] - last[:, no_id]).float().cpu()
            yes_minus_no[i:i + len(batch)] = diff
    print(f"[llm] done in {(_t.time() - t0):.1f}s")

    preds, pred_names, label_names, candidates_info = [], [], [], []
    if args.retriever == "precomputed":
        id_to_name = kb_entities["kb_id_to_name"]
    else:
        id_to_name = {str(e["id"]): e["name"] for e in kb_entities}
    cursor = 0
    for qi, cands in enumerate(candidates_per_q):
        K = len(cands)
        scores = yes_minus_no[cursor:cursor + K].tolist()
        cursor += K
        order = sorted(range(K), key=lambda j: (-scores[j], j))
        reranked = [cands[j] for j in order]
        candidates_info.append(reranked)
        preds.append(reranked[0]["id"])
        pred_names.append(reranked[0]["name"])
        label_names.append(id_to_name.get(labels[qi], str(labels[qi])))

    metrics = evaluate_predictions(
        preds, labels, candidates_info=candidates_info,
        pred_names=pred_names, label_names=label_names, tag=args.tag,
    )
    save_results(metrics, args.tag)
    print(f"[done] {args.tag}")


if __name__ == "__main__":
    main()
