import sys
import os

# Add the src directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from matrix_simulation import NaiveQuantumCircuit
from gates import Gates

class TestNaiveQuantumCircuit(unittest.TestCase):
    
    """Set up a 2-qubit circuit for testing."""
    def setUp(self):
        self.num_qubits = 2
        self.circuit = NaiveQuantumCircuit(self.num_qubits)
    
    """Test that the initial state is |00>"""
    def test_initial_state(self):
        expected_state = np.array([1, 0, 0, 0])
        np.testing.assert_array_equal(self.circuit.state, expected_state)
        
    """Test applying an X gate to the first qubit."""
    def test_apply_single_qubit_gate(self):
        # Apply X gate to the first qubit
        self.circuit.apply_single_qubit_gate(Gates.X, 0)
        
        # Expected state after applying X gate to first qubit
        expected_state = np.array([0, 0, 1, 0])
        np.testing.assert_array_almost_equal(self.circuit.state, expected_state)
    
    """Test applying CNOT gate with control qubit 0 and target qubit 1."""
    def test_apply_cnot_gate(self):
        # Set initial state to |11> by applying X to both qubits
        self.circuit.apply_single_qubit_gate(Gates.X, target_qubit=0)
        self.circuit.apply_single_qubit_gate(Gates.X, target_qubit=1)
        
        # Apply CNOT gate with control qubit 0, target qubit 1
        self.circuit.apply_cnot_gate(0, 1)
        
        # Expected state after applying CNOT to |11> (should be |10>)
        expected_state = np.array([0, 0, 1, 0])
        np.testing.assert_array_almost_equal(self.circuit.state, expected_state)
        
    """Test probability distribution for bell's state"""
    def test_get_probabilities(self):
        # Create bell's state
        self.circuit.apply_single_qubit_gate(Gates.H, 0)
        self.circuit.apply_cnot_gate(0, 1)
        
        # Expected probablities (should be |00>: 0.5 & |11>: 0.5)
        expected_probabilities = {'|00>': 0.5, '|01>': 0.0, '|10>': 0.0, '|11>': 0.5}
        self.assertEqual(self.circuit.get_probabilities(), expected_probabilities)
      
    """Test final test measurement for bell's state"""  
    def test_measure_probability_distribution(self):
        # Create bell's state
        self.circuit.apply_single_qubit_gate(Gates.H, 0)
        self.circuit.apply_cnot_gate(0, 1)
        
        # Run measurement multiple times to check frequency distribution
        num_trials = 1000
        counts = {'00': 0, '01': 0, '10': 0, '11': 0}
        
        for _ in range(num_trials):
            result = self.circuit.measure()
            counts[result] += 1
                
        # Check that |00⟩ and |01⟩ are approximately 0.5 each, with a small tolerance for randomness
        self.assertAlmostEqual(counts['00'], 500, delta=25)
        self.assertEqual(counts['01'], 0)
        self.assertEqual(counts['10'], 0)
        self.assertAlmostEqual(counts['11'], 500, delta=25)
        
    """Test expectation values for X and Z operators""" 
    def test_expectation_value(self):
        # Apply Hadamard gate to both qubits
        self.circuit.apply_single_qubit_gate(Gates.H, 0)
        self.circuit.apply_single_qubit_gate(Gates.H, 1)
        
        # Expectation value for X = 1 & Z = 0
        excepted_expectation_value_X = 1.0
        excepted_expectation_value_Z = 0.0
        
        self.assertEqual(self.circuit.expectation_value(Gates.X), excepted_expectation_value_X)
        self.assertEqual(self.circuit.expectation_value(Gates.Z), excepted_expectation_value_Z)

if __name__ == '__main__':
    unittest.main()
