"""On 1078: dump 2000-query subsample as small JSON for offline API eval."""
import os, sys, json, pickle, random
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from eval_utils import load_eval_samples

N = 2000; SEED = 42
samples = load_eval_samples()
print(f'eval n={len(samples)}')
with open('./experiments/kb_index/vanilla_deberta_cls/candidates_top50.pkl','rb') as f:
    pkl = pickle.load(f)
cands = pkl['candidates_per_q']
id2name = pkl['kb_id_to_name']

rng = random.Random(SEED)
idxs = list(range(len(samples))); rng.shuffle(idxs)
sub = sorted(idxs[:N])

out = {'seed': SEED, 'n_sub': N, 'records': []}
seen_ids = set()
for i in sub:
    out['records'].append({
        'idx': i,
        'cell': samples[i]['cell_text'],
        'gold': str(samples[i]['gold_entity_id']),
        'cands': [{'id': c['id'], 'name': c['name']} for c in cands[i][:50]],
    })
    seen_ids.add(str(samples[i]['gold_entity_id']))

# Add id->name for gold ids (so we can compute Name-Acc locally)
out['id_to_name'] = {k: id2name.get(k, k) for k in seen_ids}

with open('./results/subsample_2000.json','w') as f:
    json.dump(out, f, ensure_ascii=False)
print('wrote subsample with', len(out['records']), 'records')
