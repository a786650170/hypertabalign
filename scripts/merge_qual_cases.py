"""Merge HyperTab / R-SupCon / MPNet per-cell top-1 dumps into a qualitative case table.

For each labeled eval cell, pair the three models' predictions by (table_idx, cell_idx).
Then sample illustrative cases:
  - HyperTab correct AND both baselines wrong  (paper's core narrative)
  - All three correct                           (consistency example)
  - All three wrong                             (honest failure)

Output: LaTeX-ready table snippet + JSON of selected cases.
"""
import json, random, re

random.seed(42)

H = "C:/Users/Administrator/qual_dump_hypertab.jsonl"
R = "C:/Users/Administrator/qual_dump_rsupcon.jsonl"
M = "C:/Users/Administrator/qual_dump_mpnet.jsonl"
OUT_TEX = "C:/Users/Administrator/_case_table.tex"
OUT_JSON = "C:/Users/Administrator/_selected_cases.json"


def load(p):
    d = {}
    for line in open(p, 'r', encoding='utf-8'):
        if not line.strip(): continue
        r = json.loads(line)
        key = (r['table_idx'], r['cell_idx'])
        d[key] = r
    return d


def normalize(s):
    if s is None: return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def name_match(a, b):
    """Normalized-name equality match — treats KB-ID-cluster noise as correct."""
    return normalize(a) == normalize(b) if a and b else False


hd = load(H); rd = load(R); md = load(M)
print(f"loaded: H={len(hd)}  R={len(rd)}  M={len(md)}")

shared = set(hd) & set(rd) & set(md)
print(f"shared keys: {len(shared)}")

# classify each shared cell
buckets = {'hyper_only': [], 'all_correct': [], 'all_wrong': [], 'mixed': []}
for k in shared:
    h, r, m = hd[k], rd[k], md[k]
    gold = h['gold_name']
    if not gold: continue
    h_ok = name_match(h['top1_name'], gold)
    r_ok = name_match(r['top1_name'], gold)
    m_ok = name_match(m['top1_name'], gold)

    rec = {
        'table_idx': k[0], 'cell_idx': k[1],
        'cell_text': h['cell_text'],
        'gold_name': gold,
        'hyper_top1': h['top1_name'], 'hyper_ok': h_ok,
        'rsupcon_top1': r['top1_name'], 'rsupcon_ok': r_ok,
        'mpnet_top1': m['top1_name'], 'mpnet_ok': m_ok,
        'cell_len': len(h['cell_text']),
        'row_context': h.get('row_context'),
    }
    if h_ok and not r_ok and not m_ok:
        buckets['hyper_only'].append(rec)
    elif h_ok and r_ok and m_ok:
        buckets['all_correct'].append(rec)
    elif not h_ok and not r_ok and not m_ok:
        buckets['all_wrong'].append(rec)
    else:
        buckets['mixed'].append(rec)

for b, lst in buckets.items():
    print(f"  {b:15s}: {len(lst)}")

# pick examples — prefer cells that are SHORT (illustrative for paper)
def pick(lst, k, max_cell_len=40):
    short = [r for r in lst if r['cell_len'] <= max_cell_len]
    pool = short if short else lst
    random.shuffle(pool)
    return pool[:k]


picks = {
    'hyper_only': pick(buckets['hyper_only'], 3),
    'all_correct': pick(buckets['all_correct'], 1),
    'all_wrong': pick(buckets['all_wrong'], 1),
}

with open(OUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(picks, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT_JSON}")


def truncate(s, n=40):
    if s is None: return "---"
    s = str(s).strip()
    return s if len(s) <= n else s[:n - 2] + ".."


# emit LaTeX table
lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\caption{Qualitative comparison of top-$1$ predictions on WDC LSPM eval cells. Each row is one labeled cell; \cmark\ / \xmark\ indicates whether the model's top-$1$ matches the gold canonical name after string normalization. HyperTabAlign-direct correctly resolves cells where both R-SupCon and zero-shot MPNet fail (Group 1), agrees with both baselines on easy cells (Group 2), and shares failure modes on inherently ambiguous cells (Group 3).}")
lines.append(r"\label{tab:cases}")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{p{0.18\textwidth}|p{0.20\textwidth}|p{0.16\textwidth}|p{0.16\textwidth}|p{0.16\textwidth}}")
lines.append(r"\hline")
lines.append(r"\textbf{Cell text} & \textbf{Gold canonical name} & \textbf{HyperTab (ours)} & \textbf{R-SupCon} & \textbf{MPNet (zero-shot)} \\")
lines.append(r"\hline")
lines.append(r"\multicolumn{5}{l}{\textit{Group 1: HyperTab correct, both baselines wrong}} \\")
lines.append(r"\hline")
for c in picks['hyper_only']:
    lines.append(
        f"\\texttt{{{truncate(c['cell_text'],40)}}} & "
        f"{truncate(c['gold_name'],45)} & "
        f"\\cmark\\ {truncate(c['hyper_top1'],32)} & "
        f"\\xmark\\ {truncate(c['rsupcon_top1'],32)} & "
        f"\\xmark\\ {truncate(c['mpnet_top1'],32)} \\\\"
    )
    lines.append(r"\hline")
lines.append(r"\multicolumn{5}{l}{\textit{Group 2: all three correct}} \\")
lines.append(r"\hline")
for c in picks['all_correct']:
    lines.append(
        f"\\texttt{{{truncate(c['cell_text'],40)}}} & "
        f"{truncate(c['gold_name'],45)} & "
        f"\\cmark\\ {truncate(c['hyper_top1'],32)} & "
        f"\\cmark\\ {truncate(c['rsupcon_top1'],32)} & "
        f"\\cmark\\ {truncate(c['mpnet_top1'],32)} \\\\"
    )
    lines.append(r"\hline")
lines.append(r"\multicolumn{5}{l}{\textit{Group 3: all three wrong (honest failure case)}} \\")
lines.append(r"\hline")
for c in picks['all_wrong']:
    lines.append(
        f"\\texttt{{{truncate(c['cell_text'],40)}}} & "
        f"{truncate(c['gold_name'],45)} & "
        f"\\xmark\\ {truncate(c['hyper_top1'],32)} & "
        f"\\xmark\\ {truncate(c['rsupcon_top1'],32)} & "
        f"\\xmark\\ {truncate(c['mpnet_top1'],32)} \\\\"
    )
    lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table*}")

# note: requires \usepackage{pifont}\newcommand{\cmark}{\ding{51}}\newcommand{\xmark}{\ding{55}}
with open(OUT_TEX, 'w', encoding='utf-8') as f:
    f.write("\n".join(lines))
print(f"wrote {OUT_TEX}")

print("\n=== preview ===")
for label, lst in picks.items():
    print(f"\n--- {label} ---")
    for c in lst:
        print(f"  cell:   {c['cell_text'][:60]}")
        print(f"  gold:   {c['gold_name'][:60]}")
        print(f"  hyper:  {'OK ' if c['hyper_ok'] else 'WR '}{(c['hyper_top1'] or '---')[:60]}")
        print(f"  rsup:   {'OK ' if c['rsupcon_ok'] else 'WR '}{(c['rsupcon_top1'] or '---')[:60]}")
        print(f"  mpnet:  {'OK ' if c['mpnet_ok'] else 'WR '}{(c['mpnet_top1'] or '---')[:60]}")
