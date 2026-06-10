from sklearn.metrics import precision_recall_fscore_support
from models.knowledge.normalizer import normalize_entity_id


class Evaluator:
    def compute_alignment_metrics(self, preds, labels, candidates_info=None,
                                   debug=False, pred_names=None, label_names=None,
                                   cell_texts=None, **kwargs):
        norm_preds = [normalize_entity_id(p) for p in preds]
        norm_labels = [normalize_entity_id(l) for l in labels]

        if debug:
            print("\n[DEBUG] Eval samples (top 20):")
            for i in range(min(20, len(norm_preds))):
                match = "✓" if norm_preds[i] == norm_labels[i] else "✗"
                src = cell_texts[i] if cell_texts and i < len(cell_texts) else "?"
                pn = pred_names[i] if pred_names and i < len(pred_names) else norm_preds[i]
                ln = label_names[i] if label_names and i < len(label_names) else norm_labels[i]
                print(f"  {match} [{i}] \"{src}\"")
                print(f"        → pred: \"{pn}\" (ID={norm_preds[i]})")
                print(f"        → gold: \"{ln}\" (ID={norm_labels[i]})")
                if candidates_info and i < len(candidates_info):
                    top3 = candidates_info[i][:3]
                    names = [c.get('name', c.get('id', '?')) for c in top3]
                    print(f"        Top-3: {names}")

        correct_count = 0
        hits_at_5 = 0
        hits_at_10 = 0
        rr_sum = 0.0  # Sum of reciprocal ranks for MRR@10.

        for i, (p, l) in enumerate(zip(norm_preds, norm_labels)):
            if p == l:
                correct_count += 1

            if candidates_info and i < len(candidates_info):
                cand_ids = [normalize_entity_id(c.get('id')) for c in candidates_info[i]]
                if l in cand_ids[:5]:
                    hits_at_5 += 1
                if l in cand_ids[:10]:
                    hits_at_10 += 1
                # Reciprocal rank within the top-K candidate list. If gold is
                # not present, contribute 0 (i.e. MRR is computed at the same
                # cutoff as Hit@10 — MRR@10).
                try:
                    rank = cand_ids[:10].index(l) + 1
                    rr_sum += 1.0 / rank
                except ValueError:
                    pass
            else:
                if p == l:
                    hits_at_5 += 1
                    hits_at_10 += 1
                    rr_sum += 1.0

        n = max(len(preds), 1)
        accuracy = correct_count / n
        recall_5 = hits_at_5 / n
        recall_10 = hits_at_10 / n
        mrr_10 = rr_sum / n

        _, _, f1, _ = precision_recall_fscore_support(
            norm_labels, norm_preds, average='micro', zero_division=0,
        )

        return {
            "Accuracy": accuracy,
            "Micro-F1": f1,
            "MRR@10": mrr_10,
            "Hit@5": recall_5,
            "Hit@10": recall_10,
        }

    def compute_name_metrics(self, pred_names, label_names):
        if not pred_names or not label_names:
            return {"Name-Accuracy": 0.0, "Name-Micro-F1": 0.0}

        norm_preds = [normalize_entity_id(p) for p in pred_names]
        norm_labels = [normalize_entity_id(l) for l in label_names]
        correct = sum(1 for p, l in zip(norm_preds, norm_labels) if p == l)
        accuracy = correct / max(len(norm_preds), 1)
        _, _, f1, _ = precision_recall_fscore_support(
            norm_labels, norm_preds, average='micro', zero_division=0,
        )
        return {"Name-Accuracy": accuracy, "Name-Micro-F1": f1}

    def compute(self, preds, labels, candidates_info=None,
                pred_names=None, label_names=None, cell_texts=None, **kwargs):
        metrics = self.compute_alignment_metrics(
            preds, labels, candidates_info=candidates_info,
            pred_names=pred_names, label_names=label_names,
            cell_texts=cell_texts, **kwargs,
        )
        if pred_names is not None and label_names is not None:
            metrics.update(self.compute_name_metrics(pred_names, label_names))
        return metrics
