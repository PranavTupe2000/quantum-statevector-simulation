import numpy as np
from gates import Gates
from utils import compute_kronecker_product
from matrix_simulation import NaiveQuantumCircuit

class AdvanceQuantumCircuit:
    def __init__(self, num_qubits):
        state = np.zeros([2] * num_qubits)
        state[(0,) * num_qubits] = 1.0
        self.state = state
        self.num_qubits = num_qubits
        
    # Apply single-qubit gate to a specific qubit in a state tensor
    def apply_single_qubit_gate(self, gate, qubit):
        # Reshape gate into a tensor and apply using tensordot
        gate_tensor = np.reshape(gate, [2, 2])
        axes = [[1], [qubit]]
        self.state = np.tensordot(gate_tensor, self.state, axes=axes)
        # Move the qubit axis back to its original position
        self.state = np.moveaxis(self.state, 0, qubit)
    
    # Apply CNOT gate to two specific qubits in a state tensor
    def apply_cnot_gate(self, control_qubit, target_qubit):
        # Create the CNOT tensor of shape (2, 2, 2, 2)
        cnot_tensor = np.reshape(Gates.CNOT, [2, 2, 2, 2])
        axes = ([2, 3], [control_qubit, target_qubit])
        self.state = np.tensordot(cnot_tensor, self.state, axes=axes)
        # Move axes to restore original qubit positions
        if control_qubit < target_qubit:
            self.state = np.moveaxis(self.state, [0, 1], [control_qubit, target_qubit])
        else:
            self.state = np.moveaxis(self.state, [1, 0], [control_qubit, target_qubit])
    
    
    """
    Calculate the probabilities of each quantum state from the state tensor.
    """
    def get_probabilities(self, round_off=4):
        # Flatten the tensor to get a 1D array of amplitudes
        state = self.state.flatten()

        # Calculate probabilities by squaring the magnitude of each amplitude
        probabilities = np.abs(state) ** 2
        
        return {f"|{i:0{self.num_qubits}b}>": round(probabilities[i], round_off) for i in range(len(probabilities))}

        
if __name__ == "__main__":
    # nqc =  NaiveQuantumCircuit(3)
    # print(nqc.state)
    aqc = AdvanceQuantumCircuit(2)
    # aqc.apply_single_qubit_gate(Gates.H,0)
    # aqc.apply_single_qubit_gate(Gates.H,2)
    # aqc.apply_cnot_gate(0,1)
    # print("------")
    # print(aqc.get_probabilities())
    print(aqc.state.shape)
        
        
    
        
    