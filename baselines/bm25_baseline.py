"""
BM25 Sparse Retrieval Baseline.
Builds an inverted index over 8M KB entity names, then retrieves top-K for each eval cell.
"""
import argparse
import math
import re
import time
from collections import defaultdict, Counter
from eval_utils import (
    load_eval_samples, load_kb, evaluate_predictions, save_results,
)


def tokenize(text):
    return re.findall(r"\w+", str(text).lower())


class BM25Index:
    """Memory-efficient BM25 with inverted index for large-scale retrieval."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lens = []
        self.inverted_index = defaultdict(list)
        self.df = Counter()

    def build(self, corpus_tokens):
        """Build index from pre-tokenized corpus. corpus_tokens[i] = list of tokens for doc i."""
        self.doc_count = len(corpus_tokens)
        total_len = 0

        for doc_id, tokens in enumerate(corpus_tokens):
            dl = len(tokens)
            self.doc_lens.append(dl)
            total_len += dl
            seen = set()
            tf = Counter(tokens)
            for token, freq in tf.items():
                self.inverted_index[token].append((doc_id, freq))
                if token not in seen:
                    self.df[token] += 1
                    seen.add(token)

            if (doc_id + 1) % 1_000_000 == 0:
                print(f"  Indexed {doc_id + 1}/{self.doc_count}...")

        self.avg_dl = total_len / max(self.doc_count, 1)
        print(f"  BM25 index built: {self.doc_count} docs, vocab={len(self.inverted_index)}, avg_dl={self.avg_dl:.1f}")

    def query(self, query_tokens, top_k=10):
        """Retrieve top-K docs for a query."""
        scores = defaultdict(float)
        for token in set(query_tokens):
            if token not in self.inverted_index:
                continue
            n_t = self.df[token]
            idf = math.log((self.doc_count - n_t + 0.5) / (n_t + 0.5) + 1.0)
            for doc_id, tf in self.inverted_index[token]:
                dl = self.doc_lens[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[doc_id] += idf * numerator / denominator

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return ranked


def main():
    parser = argparse.ArgumentParser(description="BM25 Baseline")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_kb", type=int, default=0, help="Limit KB size for testing (0=all)")
    args = parser.parse_args()

    print("=" * 60)
    print("BM25 Sparse Retrieval Baseline")
    print("=" * 60)

    samples = load_eval_samples()
    kb = load_kb(max_entities=args.max_kb if args.max_kb > 0 else None)

    print("\n[1/3] Tokenizing KB entity names...")
    t0 = time.time()
    corpus_tokens = [tokenize(ent["name"]) for ent in kb]
    print(f"  Tokenized {len(corpus_tokens)} entities in {time.time() - t0:.1f}s")

    print("\n[2/3] Building BM25 index...")
    t0 = time.time()
    index = BM25Index()
    index.build(corpus_tokens)
    print(f"  Index built in {time.time() - t0:.1f}s")

    print(f"\n[3/3] Retrieving top-{args.top_k} for {len(samples)} eval cells...")
    t0 = time.time()
    preds = []
    labels = []
    pred_names = []
    label_names = []
    candidates_info = []

    id_to_name = {str(ent["id"]): ent["name"] for ent in kb}

    for i, s in enumerate(samples):
        query_tokens = tokenize(s["cell_text"])
        results = index.query(query_tokens, top_k=args.top_k)

        cands = [{"id": str(kb[doc_id]["id"]), "name": kb[doc_id]["name"]} for doc_id, _ in results]
        candidates_info.append(cands)

        if cands:
            pred_id = cands[0]["id"]
            pred_name = cands[0]["name"]
        else:
            pred_id = "NIL"
            pred_name = "NIL"

        preds.append(pred_id)
        pred_names.append(pred_name)
        labels.append(s["gold_entity_id"])
        label_names.append(id_to_name.get(s["gold_entity_id"], str(s["gold_entity_id"])))

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            qps = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / qps
            print(f"  {i + 1}/{len(samples)} queries ({qps:.0f} q/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(samples) / elapsed:.0f} q/s)")

    metrics = evaluate_predictions(
        preds, labels,
        candidates_info=candidates_info,
        pred_names=pred_names,
        label_names=label_names,
        tag="BM25",
    )
    save_results(metrics, "BM25")


if __name__ == "__main__":
    main()
