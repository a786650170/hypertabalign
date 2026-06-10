import os
import json
import argparse
import pandas as pd

def build_pure_ontology_kb(output_path="data/datasets/full_kb.jsonl"):
    """
    仅构建纯净的标准知识库（Ontology + Standard Entities）。
    不再从 CSV 文件中扫描任何脏数据。
    """
    print("🧹 正在构建纯净版标准知识库 (Standard Ontology Only)...")
    
    unique_entities = {}

    # 1. 注入核心本体 (Common DBpedia/Schema.org Properties)
    # 这些是表格归一化/实体对齐任务里常见的标准属性
    common_props = [
        "type", "name", "description", "id", "code", "date", "year", 
        "location", "position", "model", "project", "status", "notes",
        "title", "artist", "album", "genre", "language", "country",
        "city", "state", "address", "phone", "email", "url", "website",
        "company", "organization", "person", "author", "creator",
        "publisher", "format", "version", "size", "weight", "height",
        "width", "depth", "color", "price", "currency", "cost",
        "duration", "time", "day", "month", "season", "rank", "score",
        "value", "count", "number", "quantity", "percentage", "rate",
        "ratio", "area", "volume", "capacity", "speed", "velocity",
        "force", "power", "energy", "frequency", "temperature",
        "pressure", "density", "viscosity", "voltage", "current",
        "resistance", "inductance", "capacitance", "charge", "field",
        "flux", "intensity", "level", "magnitude", "amplitude",
        "phase", "angle", "slope", "gradient", "curvature", "variance",
        "deviation", "error", "accuracy", "precision", "recall",
        "f1", "sensitivity", "specificity", "selectivity", "latency",
        "throughput", "bandwidth", "bitrate", "framerate", "samplerate",
        "resolution", "definition", "quality", "fidelity", "integrity",
        "reliability", "availability", "maintainability", "usability",
        "serviceability", "portability", "scalability", "extensibility",
        "interoperability", "compatibility", "compliance", "conformance",
        "certification", "accreditation", "validation", "verification",
        "testing", "debugging", "profiling", "monitoring", "logging",
        "auditing", "tracking", "tracing", "reporting", "alerting",
        "notifying", "messaging", "signaling", "routing", "switching",
        "bridging", "gatewaying", "proxying", "tunneling", "encapsulating",
        "encrypting", "decrypting", "signing", "verifying", "authenticating",
        "authorizing", "accounting", "billing", "charging", "clearing",
        "settling", "reconciling", "balancing", "closing", "opening"
    ]

    # 添加标准形式
    for prop in common_props:
        # 形式 1: 简写
        unique_entities[prop] = {
            "id": prop,
            "name": prop,
            "type": "OntologyProperty",
            "path": f"Standard > {prop}"
        }
        # 形式 2: DBpedia URI 风格
        uri = f"http://dbpedia.org/ontology/{prop}"
        unique_entities[uri] = {
            "id": uri,
            "name": prop,
            "type": "OntologyProperty",
            "path": f"DBpedia > {prop}"
        }
        # 形式 3: Schema.org 风格
        schema_uri = f"http://schema.org/{prop}"
        unique_entities[schema_uri] = {
            "id": schema_uri,
            "name": prop,
            "type": "OntologyProperty",
            "path": f"Schema.org > {prop}"
        }

    return unique_entities


def add_from_csv(unique_entities, csv_path, id_col, name_col, type_name="Entity", path_prefix="Custom"):
    df = pd.read_csv(csv_path)
    if id_col not in df.columns or name_col not in df.columns:
        raise ValueError(f"CSV 缺少列: {id_col} 或 {name_col}")
    for _, row in df.iterrows():
        rid = str(row[id_col]).strip()
        rname = str(row[name_col]).strip()
        if not rid or not rname:
            continue
        unique_entities[rid] = {
            "id": rid,
            "name": rname,
            "type": type_name,
            "path": f"{path_prefix} > {rname}"
        }


def write_kb(unique_entities, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ent in unique_entities.values():
            f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    print(f"✅ KB 构建完成，共 {len(unique_entities)} 条实体。")
    print(f"📂 保存路径: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build KB from ontology and optional CSV sources")
    parser.add_argument("--output_path", type=str, default="data/datasets/full_kb.jsonl")
    parser.add_argument("--csv_path", type=str, action="append", default=[],
                        help="CSV source path (repeatable)")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--name_col", type=str, default="name")
    parser.add_argument("--type_name", type=str, default="Entity")
    parser.add_argument("--path_prefix", type=str, default="Custom")
    args = parser.parse_args()

    entities = build_pure_ontology_kb(output_path=args.output_path)
    for csv_path in args.csv_path:
        print(f"➕ 合并 CSV: {csv_path}")
        add_from_csv(
            entities,
            csv_path=csv_path,
            id_col=args.id_col,
            name_col=args.name_col,
            type_name=args.type_name,
            path_prefix=args.path_prefix
        )
    write_kb(entities, args.output_path)
