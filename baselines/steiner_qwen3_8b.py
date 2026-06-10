"""
Steiner+ 2024 ("Fine-tuning LLMs for Entity Matching", DAIS@ICDE 2025) replication
with Qwen3-8B as the base LLM (in place of Llama-3.1-8B-Instruct in the original).

We re-use the EXACT prompt template, label format, optimisation recipe, and LoRA
configuration of the TailorMatch reference implementation
(github.com/wbsg-uni-mannheim/TailorMatch, src/fine-tuning.py), only swapping:
  * base model     : Llama-3.1-8B-Instruct  ->  Qwen3-8B
  * training data  : WDC Products 80cc20rnd-train_small (4 k pairs)  ->  our
                     200,000-pair WDC LSPM cell-to-KB Ditto preparation
                     (1 pos + 3 random KB negatives per cell, identical to the
                     pair set used by all our other LLM baselines)

Training pairs are read directly from the Ditto preparation files; the
COL/VAL serialisation is stripped to the `field: value | field: value`
form that the TailorMatch prompt expects.

This file does NOT use wandb or HF Hub login.
"""
import os
import re
import sys
import argparse
import json
import random
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_CHECK"] = "1"

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


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


def load_pairs(path: str, limit: int = 0):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            e1 = strip_colval(parts[0])
            e2 = strip_colval(parts[1])
            try:
                label = int(parts[2])
            except ValueError:
                continue
            prompt = (PROMPT_TEMPLATE
                      .replace("'Entity 1'", f"'{e1}'")
                      .replace("'Entity 2'", f"'{e2}'"))
            response = "Yes" if label == 1 else "No"
            rows.append({"prompt": prompt, "response": response, "text": ""})
            if limit and len(rows) >= limit:
                break
    return rows


def build_chat(tokenizer, rows):
    texts = []
    for r in rows:
        msgs = [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"]},
        ]
        try:
            t = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            t = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False,
            )
        texts.append({"text": t})
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="./models_cache/Qwen3-8B")
    p.add_argument("--train_path", default="./baselines/ditto_official/data/wdc_lspm_cell_to_kb/train.txt")
    p.add_argument("--valid_path", default="./baselines/ditto_official/data/wdc_lspm_cell_to_kb/valid.txt")
    p.add_argument("--output_dir", default="./checkpoints/steiner_qwen3_8b")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--limit_train", type=int, default=0,
                   help="Cap training pairs; 0=use all 200k.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"[Steiner+/Qwen3-8B] base    = {args.base_model}")
    print(f"[Steiner+/Qwen3-8B] train   = {args.train_path}")
    print(f"[Steiner+/Qwen3-8B] output  = {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[data] loading train pairs...")
    train_rows = load_pairs(args.train_path, args.limit_train)
    valid_rows = load_pairs(args.valid_path, limit=2000)
    print(f"  train={len(train_rows)}  valid={len(valid_rows)}")
    train_ds = Dataset.from_list(build_chat(tokenizer, train_rows))
    valid_ds = Dataset.from_list(build_chat(tokenizer, valid_rows))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print("[model] loading Qwen3-8B (4-bit NF4 quant)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    lora = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.1, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # Truncate text manually to avoid the TRL `max_seq_length`-arg drift between
    # versions; once the text is short enough the trainer won't need to know
    # the limit explicitly.
    def _truncate(rows):
        out = []
        for r in rows:
            ids = tokenizer(r["text"], add_special_tokens=False,
                            truncation=True, max_length=args.max_seq_length).input_ids
            out.append({"text": tokenizer.decode(ids, skip_special_tokens=False)})
        return out
    train_ds = Dataset.from_list(_truncate(train_ds.to_list()))
    valid_ds = Dataset.from_list(_truncate(valid_ds.to_list()))

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=True,
        packing=False,
        dataset_text_field="text",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=lora,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        args=sft_cfg,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[done] saved LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
