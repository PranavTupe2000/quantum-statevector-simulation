import numpy as np
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
        # Reshape gate into a tensor and apply using tensordot
        gate_tensor = np.reshape(gate, [2, 2])
        axes = [[1], [qubit]]
        self.state = np.tensordot(gate_tensor, self.state, axes=axes)
        
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

