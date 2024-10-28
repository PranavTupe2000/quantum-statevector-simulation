import numpy as np
from gates import Gates
from utils import compute_kronecker_product

class NaiveQuantumCircuit:
    def __init__(self, num_qubits):
        # Initialize the state vector for n qubits in the |0...0⟩ state
        self.state = np.array([1] + [0] * (2**num_qubits - 1))
        self.num_qubits = num_qubits
        
    # Apply single-qubit gate to the entire state vector
    def apply_single_qubit_gate(self, gate, target_qubit):
        # Expand the single-qubit gate to the full system using Kronecker product
        full_gate = compute_kronecker_product(gate, target_qubit, self.num_qubits)
        # Apply the full gate via matrix multiplication
        self.state =  np.dot(full_gate, self.state)
        
    # Apply CNOT gate to two specific qubits in a state vector
    def apply_cnot_gate(self, control_qubit, target_qubit):
        # Create the full matrix for the n-qubit system
        # full_gate = 1
        # for i in range(self.num_qubits - 1):
        #     full_gate = np.kron(full_gate, Gates.I)
        if control_qubit < target_qubit:
            cnot_expanded = np.kron(np.eye(2**control_qubit), np.kron(Gates.CNOT, np.eye(2**(self.num_qubits - target_qubit - 1))))
        else:
            cnot_expanded = np.kron(np.eye(2**target_qubit), np.kron(Gates.CNOT, np.eye(2**(self.num_qubits - control_qubit - 1))))
        
        
        self.state =  np.dot(cnot_expanded, self.state)
        
    def get_probabilities(self, round_off=4):
        """Return probabilities of each quantum state."""
        probabilities = np.abs(self.state) ** 2
        return {f"|{i:0{self.num_qubits}b}>": round(probabilities[i], round_off) for i in range(len(probabilities))}
        


if __name__ == "__main__":
    quantumCircuit =  NaiveQuantumCircuit(3)
    quantumCircuit.apply_single_qubit_gate(Gates.H,0)
    quantumCircuit.apply_cnot_gate(0, 1)
    print(quantumCircuit.get_probabilities())