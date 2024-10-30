import numpy as np
import matplotlib.pyplot as plt
from gates import Gates

class AdvanceQuantumCircuit:
    """ Initialize the state vector for n qubits in the |0...0⟩ state"""
    def __init__(self, num_qubits):
        state = np.zeros([2] * num_qubits)
        state[(0,) * num_qubits] = 1.0
        
        self.state = state
        self.num_qubits = num_qubits
        
    """ Apply single-qubit gate to a specific qubit """
    def apply_single_qubit_gate(self, gate, qubit):
        # Performing tensor contraction
        axes = [[1], [qubit]]
        self.state = np.tensordot(gate, self.state, axes=axes)
        
        # Move the qubit axis back to its original position
        self.state = np.moveaxis(self.state, 0, qubit)
    
    """ Apply CNOT gate to two specific qubits """
    def apply_cnot_gate(self, control_qubit, target_qubit):
        # Create the CNOT tensor of shape (2, 2, 2, 2)
        cnot_tensor = np.reshape(Gates.CNOT, [2, 2, 2, 2])
        axes = ([2, 3], [control_qubit, target_qubit])
        self.state = np.tensordot(cnot_tensor, self.state, axes=axes)
        
        # Move axes to restore original qubit positions
        self.state = np.moveaxis(self.state, [0, 1], [control_qubit, target_qubit])
    
    
    """ Calculate the probabilities of each quantum state from the state tensor. """
    def get_probabilities(self, round_off=6):
        # Flatten the tensor
        state = self.state.flatten()

        # Squaring the amplitudes of the states
        probabilities = np.abs(state) ** 2
        
        # Returning a dictinory containing the probability distribution each quantum state
        return {f"|{i:0{self.num_qubits}b}>": round(probabilities[i], round_off) for i in range(len(probabilities))}

    """Measures the qubits, returning a probabilistic result."""
    def measure(self):
        # Flatten the tensor
        state = self.state.flatten()
        
        # Squaring the amplitudes of the states
        probabilities = np.abs(state) ** 2
        
        # Selecting randomly based on probabilities
        measurement_result = np.random.choice(2 ** self.num_qubits, p=probabilities)
        return format(measurement_result, f'0{self.num_qubits}b')
    
    """Visualizes a quantum state vector as a histogram of probabilities."""
    def visualize_state(self):
        # Flatten the tensor
        state = self.state.flatten()
        
        # Squaring the amplitudes of the states
        probabilities = np.abs(state) ** 2
        
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
        # Initialize the contracted state tensor to use for contractions
        contracted_state = self.state
        
        # Loop over each qubit to apply the operator with tensor contraction
        for qubit in range(self.num_qubits):
            # Contract the operator with the state along the specified qubit's axis
            contracted_state = np.tensordot(gate, contracted_state, axes=([1], [qubit]))
            # Move the contracted axis back to the original position
            contracted_state = np.moveaxis(contracted_state, 0, qubit)
        
        # Calculate the expectation value as a contraction between the original and contracted state tensors
        expectation_value = np.tensordot(np.conj(self.state), contracted_state, axes=self.num_qubits)
        return round(np.real(expectation_value.item()), 2)
        
