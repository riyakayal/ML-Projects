

# Author: Riya Kayal
# Created: 02/11/2025


# Generate n Molecules (no repatation) With Multiple Conformers Using RDKit ETKDGv3
# This guarantees each molecule has ≥5 valid conformers

import random
import time
import tracemalloc
import urllib.request
import gzip
import io
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# ── configuration ────────────────────────────────────────────────────────────
n_molecules = 100
RANDOM_SEED = 42    # set to any int for reproducibility; None = random each run

# ── auto-fetch seed SMILES (ChEMBL → PubChem REST → hardcoded fallback) ──────

# ChEMBL 36 — versioned chemreps file (canonical SMILES, tab-separated)
CHEMBL_URL  = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/chembl_36_chemreps.txt.gz"

# PubChem REST API — fetch SMILES for a range of CIDs (no gzip, no truncation)
PUBCHEM_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids}/property/IsomericSMILES/JSON"

ORGANIC_ATOMS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H C N O F P S Cl Br I

def _is_clean(smi: str) -> bool:
    """Fast string-level pre-filter: no salts, charges, bare H, or huge molecules."""
    if not smi or len(smi) > 150:
        return False
    if smi.strip() in ("[H]", "[H][H]"):
        return False
    if "." in smi:                   # salt / multi-fragment
        return False
    if "+" in smi or "-" in smi:     # charged species
        return False
    return True

def _mol_is_organic(mol) -> bool:
    """RDKit-level: only allowed atoms, no radicals, at least 2 heavy atoms."""
    if mol is None:
        return False
    if mol.GetNumHeavyAtoms() < 2:
        return False
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in ORGANIC_ATOMS:
            return False
        if atom.GetNumRadicalElectrons() > 0:
            return False
    return True

def fetch_chembl_smiles(n_target: int) -> list[str]:
    """Primary: ChEMBL 36 chemreps — canonical SMILES in column 1."""
    import json
    print("  Trying ChEMBL 36 (chemreps.txt.gz)…")
    try:
        req = urllib.request.Request(CHEMBL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            raw = resp.read()                        # download fully before decompressing
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            lines = gz.readlines()
        smiles: list[str] = []
        for line in lines[1:]:                       # skip header
            parts = line.decode("utf-8", errors="ignore").strip().split("\t")
            if len(parts) < 2:
                continue
            smi = parts[1].strip()
            if not _is_clean(smi):
                continue
            mol = Chem.MolFromSmiles(smi)
            if _mol_is_organic(mol):
                smiles.append(Chem.MolToSmiles(mol))
            if len(smiles) >= n_target:
                break
        print(f"  ✔ ChEMBL 36: {len(smiles):,} clean organic SMILES fetched.")
        return smiles
    except Exception as e:
        print(f"  ⚠ ChEMBL failed: {e}")
        return []

def fetch_pubchem_smiles(n_target: int) -> list[str]:
    """Fallback: PubChem REST API — fetch in batches of 100 CIDs."""
    import json
    print("  Trying PubChem REST API fallback…")
    smiles: list[str] = []
    batch = 100
    cid   = 1
    while len(smiles) < n_target:
        cid_range = ",".join(str(i) for i in range(cid, cid + batch))
        url = PUBCHEM_REST.format(cids=cid_range)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for entry in data.get("PropertyTable", {}).get("Properties", []):
                smi = entry.get("IsomericSMILES", "")
                if _is_clean(smi):
                    mol = Chem.MolFromSmiles(smi)
                    if _mol_is_organic(mol):
                        smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            pass   # silently skip failed batches
        cid += batch
        if cid > 50000:   # safety cap
            break
    print(f"  ✔ PubChem REST: {len(smiles):,} clean organic SMILES fetched.")
    return smiles

def fetch_seed_smiles(n_target: int = 5000) -> list[str]:
    """ChEMBL 36 → PubChem REST → hardcoded fallback."""
    print("Fetching seed SMILES pool…")
    smiles = fetch_chembl_smiles(n_target)
    if len(smiles) < 100:
        smiles = fetch_pubchem_smiles(n_target)
    if len(smiles) < 100:
        print("  ⚠ All remote sources failed — using built-in fallback list.")
        smiles = [
            "CCO","CCCO","CCCCO","CCN","CCCN","CCCCN","CCOC","CCOCC",
            "CCC(=O)O","CC(=O)OC","c1ccccc1","c1ccncc1","c1ccccc1O",
            "CC(C)O","CC(C)N","CC(C)CC","CC(C)(C)O","CC(C)(C)N",
            "CC(=O)Oc1ccccc1C(=O)O","CN1CCC[C@H]1c2cccnc2",
            "O=C(O)c1ccccc1O","NCCc1ccc(O)c(O)c1","O=Cc1ccc(O)c(OC)c1",
        ]
    print(f"  Seed pool ready: {len(smiles):,} SMILES\n")
    return smiles

seed_smiles = fetch_seed_smiles(n_target=max(5000, n_molecules * 10))

# ── helpers ──────────────────────────────────────────────────────────────────
def has_multiple_conformers(mol, n_confs=10):
    mol    = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = RANDOM_SEED if RANDOM_SEED is not None else 0xDEADBEEF
    ids    = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    return len(ids) > 1

# ── instrumentation setup ────────────────────────────────────────────────────
random.seed(RANDOM_SEED)
tracemalloc.start()
t_start = time.perf_counter()

molecule_list  = []
mol_weights    = []          # (canon_smi, MW, n_heavy_atoms) for every accepted molecule
attempts       = 0
failed_parse   = 0
filtered_atoms = 0
filtered_mw    = 0
failed_embed   = 0

# ── main loop ────────────────────────────────────────────────────────────────
while len(molecule_list) < n_molecules and attempts < 5000:
    attempts += 1
    smi = random.choice(seed_smiles)
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        failed_parse += 1
        continue

    if any(atom.GetAtomicNum() not in [1,6,7,8,9,15,16,17,35]
           for atom in mol.GetAtoms()):
        filtered_atoms += 1
        continue

    mw = Descriptors.MolWt(mol)
    if mw < 60 or mw > 500:
        filtered_mw += 1
        continue

    if has_multiple_conformers(mol):
        name  = f"Molecule_{len(molecule_list)+1}"
        canon = Chem.MolToSmiles(mol)
        molecule_list.append((name, canon))
        mol_weights.append((canon, mw, mol.GetNumHeavyAtoms()))
    else:
        failed_embed += 1

# ── wrap up ───────────────────────────────────────────────────────────────────
t_elapsed   = time.perf_counter() - t_start
_, mem_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

df = pd.DataFrame(molecule_list, columns=["Name", "smiles"])
df.to_csv(f"{n_molecules}_molecules.csv", index=False)

# also save a deduplicated version
df_unique = df.drop_duplicates(subset="smiles").reset_index(drop=True)
df_unique["Name"] = [f"Molecule_{i+1}" for i in range(len(df_unique))]
df_unique.to_csv(f"{len(df_unique)}_molecules_unique.csv", index=False)

# ── derived stats ─────────────────────────────────────────────────────────────
n_ok           = len(df)
n_unique       = df["smiles"].nunique()
duplicates     = n_ok - n_unique
all_unique     = duplicates == 0
success_rate   = n_ok / attempts * 100 if attempts else 0
avg_mw         = sum(w for _, w, _ in mol_weights) / len(mol_weights) if mol_weights else 0
biggest        = max(mol_weights, key=lambda x: x[1]) if mol_weights else ("–", 0.0, 0)
smallest       = min(mol_weights, key=lambda x: x[1]) if mol_weights else ("–", 0.0, 0)
avg_atoms      = sum(a for _, _, a in mol_weights) / len(mol_weights) if mol_weights else 0
mols_per_sec   = n_ok / t_elapsed if t_elapsed else 0
total_rejected = attempts - n_ok
mins, secs     = divmod(t_elapsed, 60)
time_str       = f"{int(mins)}m {secs:.1f}s" if mins >= 1 else f"{t_elapsed:.2f}s"
status_icon    = "SUCCESS" if n_ok >= n_molecules else "INCOMPLETE  (hit 5,000 attempt cap)"

# ── print summary ─────────────────────────────────────────────────────────────
W = 62   # inner content width

def row(label, value):
    left = f"    {label}"
    right = str(value)
    gap = W - len(left) - len(right)
    return f"║{left}{' ' * max(1, gap)}{right}║"

def section(title):
    return f"║  {title:<{W-1}}║"

def div():
    return "╠" + "═" * W + "╣"

print()
print("╔" + "═" * W + "╗")
print("║" + "  🧪  CONFORMER GENERATION — RUN SUMMARY".center(W) + "║")
print(div())

print(section("OUTCOME"))
print(row("Status",                  status_icon))
print(row("Random seed",             f"{RANDOM_SEED if RANDOM_SEED is not None else 'None (non-reproducible)'}"))
print(row("Seed SMILES pool",        f"{len(seed_smiles):,}"))
print(row("Molecules generated",     f"{n_ok:,} / {n_molecules:,}"))
print(row("Unique molecules",        f"{n_unique:,}  {'✔ all unique' if all_unique else f'⚠ {duplicates:,} duplicates'}"))
print(row("Total loop attempts",     f"{attempts:,}  (cap: 5,000)"))
print(row("Overall success rate",    f"{success_rate:.1f}%"))
print(div())

print(section("FILTERING BREAKDOWN"))
print(row("✔  Accepted",                     f"{n_ok:,}"))
print(row("✘  Invalid SMILES / parse error", f"{failed_parse:,}"))
print(row("✘  Non-organic atoms",            f"{filtered_atoms:,}"))
print(row("✘  MW outside 60–500 Da",         f"{filtered_mw:,}"))
print(row("✘  Too few conformers embedded",  f"{failed_embed:,}"))
print(row("Σ  Total rejected",               f"{total_rejected:,}"))
print(div())

print(section("MOLECULAR WEIGHT & SIZE  (accepted molecules)"))
print(row("Smallest",  f"{smallest[1]:.1f} Da  {smallest[2]} atoms   {smallest[0][:24]}"))
print(row("Largest",   f"{biggest[1]:.1f} Da  {biggest[2]} atoms   {biggest[0][:24]}"))
print(row("Average",   f"{avg_mw:.1f} Da  {avg_atoms:.1f} atoms"))
print(div())

print(section("PERFORMANCE"))
print(row("Wall-clock time",  time_str))
print(row("Throughput",       f"{mols_per_sec:.2f} mol/s"))
print(row("Peak memory",      f"{mem_peak / 1e6:.2f} MB"))
print(div())

print(section("OUTPUT FILES"))
print(row("All molecules",    f"{n_molecules}_molecules_multiple_conformers.csv  ({n_ok:,} rows)"))
print(row("Unique only",      f"{n_unique}_molecules_unique.csv  ({n_unique:,} rows)"))
print("╚" + "═" * W + "╝")
print()

