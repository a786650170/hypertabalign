import argparse
import os
import sys
import json
import pandas as pd
import torch

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.parser import TableParser
from models.knowledge.manager import KnowledgeManager
from training.trainer import JointTrainer


def build_web_config(args):
    web_base_url = args.web_base_url
    web_id_prefix = args.web_id_prefix
    web_lang = args.web_lang
    if args.web_provider == "wikipedia":
        web_base_url = web_base_url or f"https://{web_lang}.wikipedia.org/w/api.php"
        web_id_prefix = web_id_prefix or "wiki"
    elif args.web_provider == "wikidata":
        web_base_url = web_base_url or "https://www.wikidata.org/w/api.php"
        web_id_prefix = web_id_prefix or "wd"
    elif args.web_provider == "mediawiki":
        if not web_base_url:
            raise ValueError("provider=mediawiki 需要指定 --web_base_url")
        web_id_prefix = web_id_prefix or "mw"
    return web_base_url, web_id_prefix, web_lang


def main():
    parser = argparse.ArgumentParser(description="Normalize table cells into standard entities")
    parser.add_argument("--input_table", type=str, required=True, help="CSV table to normalize")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset dir with dataset.json (optional)")
    parser.add_argument("--output_table", type=str, default=None, help="Output CSV path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/large_model.pt")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--kb_path", type=str, default="data/datasets/full_kb.jsonl")
    parser.add_argument("--use_local_kb", action="store_true", help="Use local KB index for retrieval")
    parser.add_argument("--use_web_retriever", action="store_true", default=False, help="Enable web retriever")
    parser.add_argument("--disable_web_retriever", action="store_true", help="Disable web retriever")
    parser.add_argument("--hybrid_retrieval", action="store_true", help="Combine local KB with web retrieval")
    parser.add_argument("--kb_fallback", action="store_true", help="Enable local KB fallback when web is unavailable")
    parser.add_argument("--web_provider", type=str, default="wikidata",
                        choices=["wikipedia", "wikidata", "mediawiki"])
    parser.add_argument("--web_base_url", type=str, default=None)
    parser.add_argument("--web_id_prefix", type=str, default=None)
    parser.add_argument("--web_lang", type=str, default="en")
    parser.add_argument("--web_top_k", type=int, default=5)
    parser.add_argument("--web_timeout", type=int, default=5)
    parser.add_argument("--web_max_retries", type=int, default=2)
    parser.add_argument("--web_backoff", type=float, default=0.5)
    parser.add_argument("--web_cache_dir", type=str, default="data/cache/wiki")
    parser.add_argument("--web_domain_keywords", type=str, default="")
    parser.add_argument("--web_min_query_len", type=int, default=2)
    parser.add_argument("--web_min_title_len", type=int, default=2)
    parser.add_argument("--web_score_boost", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.5, help="NER entity threshold")
    parser.add_argument("--max_candidates", type=int, default=5)
    parser.add_argument("--include_headers", action="store_true", help="Also normalize header cells")
    parser.add_argument("--replace_with", type=str, default="name", choices=["name", "id"])
    parser.add_argument("--no_header", action="store_true", help="Read CSV without header row")
    args = parser.parse_args()

    # dataset manifest (optional)
    if args.dataset_dir and os.path.isdir(args.dataset_dir):
        manifest_path = os.path.join(args.dataset_dir, "dataset.json")
        if os.path.exists(manifest_path) and args.kb_path == "data/datasets/full_kb.jsonl":
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                kb_hint = manifest.get("kb_path") or manifest.get("kb_hint")
                if kb_hint:
                    candidate = kb_hint if os.path.isabs(kb_hint) else os.path.join(args.dataset_dir, kb_hint)
                    if os.path.exists(candidate):
                        args.kb_path = candidate
            except Exception:
                pass

    use_web = args.use_web_retriever and not args.disable_web_retriever
    use_kb = args.use_local_kb or args.hybrid_retrieval or (not use_web)

    # 1) 读取表格
    if args.no_header:
        df = pd.read_csv(args.input_table, header=None)
    else:
        df = pd.read_csv(args.input_table)

    # 2) 构建表格图
    parser_table = TableParser()
    graph = parser_table.df_to_graph(df)

    # 3) KB 管理器
    kb_manager = None
    if use_kb:
        kb_manager = KnowledgeManager(working_dir="experiments/kb_index", model_name=args.model_name)

    # 4) 初始化训练器
    web_base_url, web_id_prefix, web_lang = build_web_config(args)
    web_keywords = [k.strip() for k in args.web_domain_keywords.split(",") if k.strip()]
    trainer = JointTrainer(config={
        "model_name": args.model_name,
        "multi_gpu": False,
        "use_web_retriever": use_web,
        "hybrid_retrieval": args.hybrid_retrieval,
        "kb_fallback": args.kb_fallback,
        "rag_in_train": False,
        "web_top_k": args.web_top_k,
        "web_timeout": args.web_timeout,
        "web_max_retries": args.web_max_retries,
        "web_backoff": args.web_backoff,
        "web_cache_dir": args.web_cache_dir,
        "web_lang": web_lang,
        "web_base_url": web_base_url,
        "web_id_prefix": web_id_prefix,
        "web_domain_keywords": web_keywords,
        "web_min_query_len": args.web_min_query_len,
        "web_min_title_len": args.web_min_title_len,
        "web_score_boost": args.web_score_boost
    }, kb_manager=kb_manager)

    trainer.load_checkpoint(args.checkpoint)

    if use_kb and kb_manager and (not kb_manager.is_built):
        if not os.path.exists(args.kb_path):
            raise FileNotFoundError(f"未找到 KB 文件: {args.kb_path}")
        print(f"🚀 正在构建向量索引: {args.kb_path}...")
        kb_manager.build_index(args.kb_path, encoder=trainer.kb_encoder)

    # 5) 推理与归一化
    results = trainer.infer_table(
        graph,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        include_headers=args.include_headers
    )

    if not results:
        print("⚠️ 未检测到可归一的实体单元格。")

    for item in results:
        r, c = item["coord"]
        if r < 0 or c < 0:
            continue
        if r >= df.shape[0] or c >= df.shape[1]:
            continue
        value = item["pred_name"] if args.replace_with == "name" else item["pred_id"]
        if not value:
            continue
        df.iat[r, c] = value

    # 6) 保存输出
    out_path = args.output_table
    if not out_path:
        base, ext = os.path.splitext(args.input_table)
        out_path = f"{base}.normalized{ext or '.csv'}"

    df.to_csv(out_path, index=False, header=not args.no_header)
    print(f"✅ 已输出归一化表格: {out_path}")


if __name__ == "__main__":
    main()
