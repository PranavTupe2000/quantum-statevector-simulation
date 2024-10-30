# Quantum Statevector Simulator

## Overview

This project is a Python-based quantum circuit simulator implementing **matrix-based** and **tensor-based** approaches for quantum state simulations. The simulator supports operations on multi-qubit systems, including the application of single-qubit and multi-qubit gates, with the flexibility to simulate quantum states using both matrix multiplication and tensor contraction.

It allows users to analyze quantum gates' performance on various qubit counts and examine the runtime between these simulation techniques. This project provides a foundational understanding of quantum computations on classical hardware.

## Quantum Open Source Foundation

![QOSF logo](https://qosf.org/assets/img/logos/qosf_colour_logo.svg)

This project is a part of screening task for Quantum Computing Mentorship Program by QOSF. The project implements *Task 1 Statevector simulation of quantum circuits* of Cohort 10 screening.

---

## Table of Contents

- [Project Setup](#project-setup)
- [Features](##Features)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Bonus](#bonus)

---

## Project setup

Ensure you have Python 3 installed on your system.

Clone this repository and then install the required dependencies:

`pip install -r requirements.txt`

---

## Features

- **Matrix-based Simulation**: Utilizes the Kronecker product and matrix multiplication for state evolution.
- **Tensor-based Simulation**: Leverages tensor contraction for a more memory-efficient approach.
- **Gate Operations**: Supports single-qubit gates (X, H, Y, Z) and the two-qubit CNOT gate.
- **Runtime Analysis**: Measured and plotted the time taken to run simulations for various qubit counts.

---

## Project Structure

```python
quantum-statevector-simulation/
├── README.md                   # Project overview, setup, and instructions
├── requirements.txt            # Dependencies
├── src/                        # Main codebase
│   ├── __init__.py             # Makes src a package
│   ├── gates.py                # Quantum gate definitions
│   ├── matrix_simulation.py    # Main code of matrix multiplication simulator
│   ├── tensor_simulation.py    # Main code of tensor multiplication simulator
│   └── utils.py                # Helper functions
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb          # Notebook for analysing runtimes and qubit limit
├── tests/                         # Unit tests
│   ├── test_matrix_simulation.py  # Tests for matrix simulator accuracy
│   └── test_tensor_simulation.py  # Tests for tensor simulator accuracy
└── docs/                       # Documentation
    └── analysis.md             # Documentation of analysis

```

---

## Implementation Details

1. **Matrix-based Simulation**:

   - Implements a statevector as a vector of size 2^n, where n is the number of qubits.
   - Applies gates using Kronecker products for expanding single-qubit gates to multi-qubit states.
2. **Tensor-based Simulation**:

   - Uses tensors of shape (2, 2, ..., 2) for the n-qubit system.
   - Utilizes `np.tensordot` for efficient gate application via tensor contractions.
3. **Gates**:

   - Single-qubit and two-qubit gates are defined in `gates.py` and applied by mapping them onto the full state representation.

---

## Usage

### Running the Simulations

1. **Matrix-based Simulation**

   ```python
   from matrix_simulation import NaiveQuantumCircuit
   from gates import Gates
   circuit = NaiveQuantumCircuit(num_qubits=2)

   # Applying the gates
   circuit.apply_single_qubit_gate(Gates.X, target_qubit=0)
   circuit.apply_cnot_gate(control_qubit=0, target_qubit=1)
   ```
2. **Tensor-based Simulation**

   ```python
   from tensor_simulation import AdvanceQuantumCircuit
   from gates import Gates
   circuit = AdvanceQuantumCircuit(num_qubits=2)

   # Applying the gates
   circuit.apply_single_qubit_gate(Gates.H, qubit=0)
   circuit.apply_cnot_gate(control_qubit=0, target_qubit=1)
   ```

---

## Performance Analysis

The analysis has been carried out in **[notebooks/analysis.ipynb](./notebooks/analysis.ipynb)** notebook. The finds of analysis are then documented in **[docs/analysis.md](./docs/analysis.md)** file.

### Key Insights

- Tensor-based simulations runs faster and can handle a larger number of qubits than matrix-based.
- The matrix simulator can simulate upto **14 qubits**.
- The tensor simulation can simulate upto **29 qubits**.


## Bonus

The bonus questions of the screening task has been implemented in **[notebooks/bonus_questions.ipynb](./notebooks/bonus_questions.ipynb)** notebook. The notebook contains:

- Sampling from Final States
- Expectation Values

The main functions of these tasks i.e. `measure()` and `expectation_value(gate)` are updated in both the files **[src/matrix_simulation.py](./src/matrix_simulation.py)** and **[src/tensor_simulation.py](./src/tensor_simulation.py)**.