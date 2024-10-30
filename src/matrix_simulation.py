import numpy as np
import matplotlib.pyplot as plt
from gates import Gates
from utils import compute_kronecker_product, compute_kronecker_product_with_itself

class NaiveQuantumCircuit:
    """ Initialize the state vector for n qubits in the |0...0⟩ state """
    def __init__(self, num_qubits):
        self.state = np.array([1] + [0] * (2**num_qubits - 1))
        self.num_qubits = num_qubits
        
    """ Apply single-qubit gate to the entire state vector """
    def apply_single_qubit_gate(self, gate, target_qubit):
        # Expand the single-qubit gate to the full system using Kronecker product
        full_gate = compute_kronecker_product(gate, target_qubit, self.num_qubits)
        
        # Apply the full gate via matrix multiplication
        self.state =  np.dot(full_gate, self.state)
        
    """ Apply CNOT gate to two specific qubits in a state vector """
    def apply_cnot_gate(self, control_qubit, target_qubit):
        # Expand the CNOT gate to the full system
        cnot_expanded = np.kron(np.eye(2**control_qubit), np.kron(Gates.CNOT, np.eye(2**(self.num_qubits - target_qubit - 1))))
        
        # Apply the expanded gate via matrix multiplication
        self.state =  np.dot(cnot_expanded, self.state)
     
    """ Calculate the probabilities of each quantum state """
    def get_probabilities(self, round_off=6):
        # Squaring the amplitudes of the states
        probabilities = np.abs(self.state) ** 2
        
        # Returning a dictinory containing the probability distribution each quantum state
        return {f"|{i:0{self.num_qubits}b}>": round(probabilities[i], round_off) for i in range(len(probabilities))}
    
    """Measures the qubits, returning a probabilistic result."""
    def measure(self):
        # Squaring the amplitudes of the states
        probabilities = np.abs(self.state) ** 2
        
        # Selecting randomly based on probabilities
        measurement_result = np.random.choice(2 ** self.num_qubits, p=probabilities)
        return format(measurement_result, f'0{self.num_qubits}b')
    
    """Visualizes a quantum state vector as a histogram of probabilities."""
    def visualize_state(self):
        # Squaring the amplitudes of the states
        probabilities = np.abs(self.state) ** 2
        
        # Generate state labels in binary format, e.g., |00>, |01>, etc.
        state_labels = [f"|{i:0{self.num_qubits}b}⟩" for i in range(len(probabilities))]
        
        # Create the bar plot with state labels
        plt.bar(state_labels, probabilities)
        plt.xlabel('State')
        plt.ylabel('Probability')
        plt.title('Quantum State Probabilities')
        plt.show()
        
    """Compute the expectation value of the operator with respect to the given state."""
    def expectation_value(self, gate):
        # Expand the single-qubit gate to the operator (full gate)
        operator = compute_kronecker_product_with_itself(gate, self.num_qubits)
        
        # Apply operator on the state
        # Compute Op|Ψ⟩
        operator_dot_state = np.dot(operator, self.state)
        
        # Take conjugate transpose of the state
        # Compute ⟨Ψ|
        state_bra = np.conj(self.state).T
        
        # Compute ⟨Ψ|Op|Ψ⟩
        return round(np.real(np.dot(state_bra, operator_dot_state)), 2)          
        
