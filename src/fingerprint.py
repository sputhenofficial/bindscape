import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def morgan_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def physchem_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
    ], dtype=np.float32)


def compute_all(smiles_list):
    fps = {}
    descs = {}
    failed = 0
    for smi in smiles_list:
        fp = morgan_fp(smi)
        if fp is None:
            failed += 1
            continue
        fps[smi] = fp
        descs[smi] = physchem_descriptors(smi)
    print(f"Fingerprints computed: {len(fps)} (failed: {failed})")
    return fps, descs
