import subprocess
import pandas as pd
import numpy as np
from rdkit import Chem

# cut before pandas: 633+ cols; 2=SMILES,7=Target,8=Org,9=Ki,10=IC50,11=Kd,12=EC50,45=UniProt
_CUT_COLS = '2,7,8,9,10,11,12,45'

SMILES_COL = 'Ligand SMILES'
UNIPROT_COL = 'UniProt (SwissProt) Primary ID of Target Chain 1'
ORGANISM_COL = 'Target Source Organism According to Curator or DataSource'
TARGET_NAME_COL = 'Target Name'
KI_COL = 'Ki (nM)'
IC50_COL = 'IC50 (nM)'
KD_COL = 'Kd (nM)'
EC50_COL = 'EC50 (nM)'


def inspect_columns(path):
    cols = pd.read_csv(path, sep='\t', nrows=5, low_memory=False).columns.tolist()
    print(cols)
    return cols


def parse_affinity(val):
    if pd.isna(val):
        return None
    s = str(val).strip().replace(' ', '')
    bump = s.startswith('>')
    s = s.replace('>', '').replace('<', '')
    try:
        return float(s) + bump
    except ValueError:
        return None


def load_and_filter(path):
    proc = subprocess.Popen(['cut', '-f', _CUT_COLS, path], stdout=subprocess.PIPE)
    df = pd.read_csv(proc.stdout, sep='\t', low_memory=False)
    proc.wait()
    assert set([SMILES_COL, UNIPROT_COL, ORGANISM_COL, TARGET_NAME_COL,
                KI_COL, IC50_COL, KD_COL, EC50_COL]).issubset(df.columns), \
        f"Column mismatch — got: {df.columns.tolist()}"

    df = df[df[ORGANISM_COL].str.lower().str.contains('homo sapiens', na=False)]
    df = df[df[TARGET_NAME_COL].str.lower().str.contains('kinase', na=False)]
    df = df[df[SMILES_COL].notna() & (df[SMILES_COL] != '')]
    df = df[df[UNIPROT_COL].notna()]

    df = df[df[SMILES_COL].apply(lambda s: Chem.MolFromSmiles(str(s)) is not None)]

    for col in [KI_COL, IC50_COL, KD_COL, EC50_COL]:
        if col is not None and col in df.columns:
            df[col + '_nM'] = df[col].apply(parse_affinity)

    priority = [c + '_nM' for c in [KI_COL, KD_COL, IC50_COL, EC50_COL]
                if c is not None and c + '_nM' in df.columns]

    def assign_label(row):
        for col in priority:
            v = row.get(col)
            if v is not None and not np.isnan(v):
                if v <= 1000.0:
                    return 1
                elif v > 10000.0:
                    return 0
                else:
                    return -1  # ambiguous band — drop
        return None

    df['label'] = df.apply(assign_label, axis=1)
    df = df[df['label'].isin([0, 1])]

    pos = (df['label'] == 1).sum()
    neg = (df['label'] == 0).sum()
    print(f"Positives: {pos}, Negatives: {neg}, ratio: {pos/max(neg,1):.2f}:1")

    return df[[SMILES_COL, UNIPROT_COL, TARGET_NAME_COL, 'label']].reset_index(drop=True)
