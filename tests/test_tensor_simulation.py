import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from tensor_simulation import AdvanceQuantumCircuit
from gates import Gates

class TestAdvanceQuantumCircuit(unittest.TestCase):
    
    """Set up a 2-qubit circuit for testing."""
    def setUp(self):
        self.circuit = AdvanceQuantumCircuit(num_qubits=2)

    """Test that the initial state is |00>."""
    def test_initial_state(self):
        expected_state = np.array([[1, 0], [0, 0]] )
        np.testing.assert_array_equal(self.circuit.state, expected_state)

    """Test applying a single-qubit gate (X gate) on the first qubit."""
    def test_apply_single_qubit_gate(self):
        # Apply X gate on the first qubit
        self.circuit.apply_single_qubit_gate(Gates.X, qubit=0)
        
        # Verify that the result is |10>
        expected_state = np.array([[0, 0], [1, 0]] )
        np.testing.assert_array_equal(self.circuit.state, expected_state)

    """Test applying a CNOT gate with qubit 0 as control and qubit 1 as target."""
    def test_apply_cnot_gate(self):
        # Apply X gate on the first qubit to prepare |10> state
        self.circuit.apply_single_qubit_gate(Gates.X, qubit=0)
        
        # Apply CNOT gate to make the state |11>
        self.circuit.apply_cnot_gate(control_qubit=0, target_qubit=1)
        
        # Verify that the result is |11>
        expected_state = np.array([[0, 0], [0, 1]] )
        np.testing.assert_array_equal(self.circuit.state, expected_state)

    """Test probability distribution for bell's state"""
    def test_get_probabilities(self):
        # Create bell's state
        self.circuit.apply_single_qubit_gate(Gates.H, 0)
        self.circuit.apply_cnot_gate(0, 1)
        
        # Expected probablities (should be |00>: 0.5 & |11>: 0.5)
        expected_probabilities = {'|00>': 0.5, '|01>': 0.0, '|10>': 0.0, '|11>': 0.5}
        self.assertEqual(self.circuit.get_probabilities(), expected_probabilities)

if __name__ == '__main__':
    unittest.main()
