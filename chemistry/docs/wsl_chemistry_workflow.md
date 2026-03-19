# WSL Chemistry Workflow

For a short Chinese summary of the full F2 three-step automation, see [f2_auto_workflow.md](f2_auto_workflow.md).

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
python chemistry/scripts/export_h2_terms_wsl.py --R 2.6 --unit angstrom --output chemistry/data/h2_R2.6_sto-3g.json
```

For 2-qubit tapered encoding (SCBK):

```bash
python chemistry/scripts/export_h2_terms_wsl.py --R 2.6 --unit angstrom --output chemistry/data/h2_R2.6_sto-3g_scbk2.json --encoding scbk2
```

This creates:

`chemistry/data/h2_R2.6_angstrom_sto-3g_scbk2.json`

## 3. Export LiH Hamiltonian JSON (JW + reduction)

Run with the LiH reduction preset:

```bash
cd /mnt/d/OneDrive/work/research/code/Quantum_control
source ~/venvs/chem/bin/activate
python chemistry/scripts/export_lih_terms_wsl.py \
	--R 1.6 \
	--unit angstrom \
	--reduction paper \
	--output chemistry/data/lih_R1.6_angstrom_sto-3g_6q.json
```

This creates:

`chemistry/data/lih_R1.6_angstrom_sto-3g_6q.json`

Notes:
- `paper` (default) maps to `occupied_indices=[0]`, `active_indices=[1,2,5]` (typically 6 qubits).
- You can override with `--occupied-indices` and `--active-indices`.

## 4. Export F2 Hamiltonian JSON (JW + 12-qubit active space)

Run with the F2 12-qubit preset:

```bash
cd /mnt/d/OneDrive/work/research/code/Quantum_control
source ~/venvs/chem/bin/activate
python chemistry/scripts/export_f2_terms_wsl.py \
	--R 1.4 \
	--unit angstrom \
	--reduction paper12 \
	--output chemistry/data/f2_R1.4_angstrom_sto-3g_12q.json
```

This creates:

`chemistry/data/f2_R1.4_angstrom_sto-3g_12q.json`

Notes:
- `paper12` (default) maps to `occupied_indices=[0,1,2,3]`, `active_indices=[4,5,6,7,8,9]` (typically 12 qubits).
- This corresponds to the requested orbital choice: freeze `1a1,2a1,3a1,4a1`; active `1e1,2e1,3e1,4e1,5a1,6a1`.
- You can override with `--occupied-indices` and `--active-indices`.

## 5. Use in Windows notebook

Notebook cells should load JSON from:

`chemistry/data/h2_R2.6_sto-3g.json`

Fields used:
- `constant`
- `terms`
- `nqubits`
- `fci_energy`

This removes the need to install PySCF/OpenFermion in Windows.
