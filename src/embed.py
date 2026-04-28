import json
import re
import time
import numpy as np
import requests
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MAX_RESIDUES = 1022  # ESM2 context limit minus BOS/EOS
CANONICAL_AA = re.compile(r'^[ACDEFGHIKLMNPQRSTVWY]+$')


def fetch_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None
    lines = r.text.strip().split('\n')
    seq = ''.join(lines[1:])
    return seq if seq else None


def fetch_all_sequences(uniprot_ids, cache_path="uniprot_sequences.json"):
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    missing = [uid for uid in uniprot_ids if uid not in cache]
    print(f"Fetching {len(missing)} sequences (cached: {len(cache)})")

    for i, uid in enumerate(missing):
        seq = fetch_sequence(uid)
        if seq is not None:
            cache[uid] = seq
        if i % 50 == 0:
            with open(cache_path, 'w') as f:
                json.dump(cache, f)
        time.sleep(1.0)

    with open(cache_path, 'w') as f:
        json.dump(cache, f)

    return cache


def filter_sequences(seq_map):
    valid = {}
    for uid, seq in seq_map.items():
        if len(seq) < 50:
            continue
        if not CANONICAL_AA.match(seq):
            continue
        valid[uid] = seq
    dropped = len(seq_map) - len(valid)
    print(f"Sequences after filter: {len(valid)} (dropped {dropped})")
    return valid


def embed_sequences(seq_map, embed_path="protein_embeddings.npy", ids_path="protein_ids.json", batch_size=16):
    embed_path = Path(embed_path)
    ids_path = Path(ids_path)
    ckpt_path = embed_path.parent / (embed_path.stem + '_ckpt.npy')
    ckpt_ids_path = ids_path.parent / (ids_path.stem + '_ckpt.json')

    if embed_path.exists() and ids_path.exists():
        embeddings = np.load(embed_path)
        with open(ids_path) as f:
            ids = json.load(f)
        print(f"Loaded cached embeddings: {embeddings.shape}")
        return dict(zip(ids, embeddings))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.eval()

    uids = list(seq_map.keys())
    seqs = [seq_map[u] for u in uids]

    truncated = sum(1 for s in seqs if len(s) > MAX_RESIDUES)
    print(f"Sequences truncated to {MAX_RESIDUES} residues: {truncated}/{len(seqs)} ({100*truncated/len(seqs):.1f}%)")
    if truncated / len(seqs) > 0.10:
        print("WARNING: >10% truncation — C-terminal domains discarded, embedding quality may degrade")

    seqs = [s[:MAX_RESIDUES] for s in seqs]

    start_idx = 0
    batch_chunks = []

    if ckpt_path.exists() and ckpt_ids_path.exists():
        ckpt_embs = np.load(ckpt_path)
        with open(ckpt_ids_path) as f:
            done_ids = json.load(f)
        start_idx = len(done_ids)
        batch_chunks.append(ckpt_embs)
        print(f"Resuming from checkpoint: {start_idx}/{len(seqs)} already embedded")

    CKPT_EVERY = 5

    with torch.no_grad():
        for batch_num, i in enumerate(range(start_idx, len(seqs), batch_size)):
            batch_seqs = seqs[i:i + batch_size]
            inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=MAX_RESIDUES + 2)
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state  # (B, L, 640)
            # BOS at position 0 removed by [:, 1:-1].
            # EOS for the longest sequence in the batch sits at the final column and is
            # removed by the -1 cut. For shorter sequences EOS sits at an interior column
            # with attention_mask=1; zero it explicitly so it is excluded from every pool.
            B_local, L_full = inputs['attention_mask'].shape
            seq_lens = inputs['attention_mask'].sum(dim=1)           # total tokens incl BOS+EOS
            positions = torch.arange(L_full - 2, device=hidden.device).unsqueeze(0)
            eos_in_slice = (seq_lens - 2).unsqueeze(1)               # EOS index in sliced tensor
            residue_mask = inputs['attention_mask'][:, 1:-1] * (positions != eos_in_slice)
            mask = residue_mask.unsqueeze(-1).float()
            pooled = (hidden[:, 1:-1, :] * mask).sum(dim=1) / mask.sum(dim=1)
            batch_chunks.append(pooled.cpu().float().numpy())

            if (batch_num + 1) % CKPT_EVERY == 0:
                ckpt_embs = np.concatenate(batch_chunks, axis=0)
                np.save(ckpt_path, ckpt_embs)
                with open(ckpt_ids_path, 'w') as f:
                    json.dump(uids[:len(ckpt_embs)], f)

            if i % (batch_size * 10) == 0:
                print(f"  Embedded {i}/{len(seqs)}")

    embeddings = np.concatenate(batch_chunks, axis=0)
    np.save(embed_path, embeddings)
    with open(ids_path, 'w') as f:
        json.dump(uids, f)

    if ckpt_path.exists():
        ckpt_path.unlink()
    if ckpt_ids_path.exists():
        ckpt_ids_path.unlink()

    print(f"Saved embeddings: {embeddings.shape}")
    return dict(zip(uids, embeddings))
