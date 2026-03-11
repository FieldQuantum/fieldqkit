# WSL Chemistry Workflow

Use WSL for chemistry-heavy calculations (PySCF/OpenFermion), and keep Windows for the quantum control framework runtime.

## 1. WSL environment

```bash
sudo apt update
sudo apt install -y build-essential cmake gfortran python3-venv python3-dev libopenblas-dev liblapack-dev
python3 -m venv ~/venvs/chem
source ~/venvs/chem/bin/activate
pip install -U pip
pip install pyscf openfermion openfermionpyscf numpy
```

## 2. Export H2 Hamiltonian JSON

```bash
cd /mnt/d/OneDrive/work/research/code/Quantum_control
source ~/venvs/chem/bin/activate
./scripts/run_wsl_export_h2.sh 2.6 angstrom
```

Equivalent direct command:

```bash
python scripts/export_h2_terms_wsl.py --R 2.6 --unit angstrom --output examples/data/chemistry/h2_R2.6_sto-3g.json
```

For 2-qubit tapered encoding (SCBK):

```bash
./scripts/run_wsl_export_h2_2q.sh 2.6 angstrom
```

This creates:

`examples/data/chemistry/h2_R2.6_angstrom_sto-3g_scbk2.json`

## 3. Use in Windows notebook

Notebook cells should load JSON from:

`examples/data/chemistry/h2_R2.6_sto-3g.json`

Fields used:
- `constant`
- `terms`
- `nqubits`
- `fci_energy`

This removes the need to install PySCF/OpenFermion in Windows.