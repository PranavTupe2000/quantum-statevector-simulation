import numpy as np
from gates import Gates
 

""" Convert gate matrix to a full matrix by using kronecker product. """    
def compute_kronecker_product(gate, target_qubit, num_qubits):
    full_gate = 1
    
    # Loop over each qubit position in the circuit
    for i in range(num_qubits):
        # Apply the specified gate on the target qubit and identity on others
        full_gate = np.kron(full_gate, gate if i == target_qubit else Gates.I)
    
    return full_gate



















def compute_kronecker_product_n(matrices):
    """
    Compute the Kronecker product of a list of matrices.
    Args:
        matrices (list): List of numpy arrays (matrices) to multiply.
    Returns:
        numpy array: Resulting matrix after performing the Kronecker product.
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

def get_operator_matrices(gate_matrix, qubit, num_qubits):
    return [Gates.I if i != qubit else gate_matrix for i in range(num_qubits)]

# def initialize_statevector(n_qubits):
#     """
#     Initialize the statevector for an n-qubit system.
#     Args:
#         n_qubits (int): Number of qubits in the quantum system.
#     Returns:
#         numpy array: Initial statevector, representing the |0> state for all qubits.
#     """
#     statevector = np.zeros(2**n_qubits, dtype=complex)
#     statevector[0] = 1  # Start in the |0> state
#     return statevector

# def apply_gate(statevector, gate, target_qubit, n_qubits):
#     """
#     Apply a gate to a specific qubit within a statevector.
#     Args:
#         statevector (numpy array): The statevector to modify.
#         gate (numpy array): The gate matrix to apply.
#         target_qubit (int): Index of the qubit to which the gate is applied.
#         n_qubits (int): Total number of qubits in the system.
#     Returns:
#         numpy array: The statevector after the gate application.
#     """
#     gate_expanded = kron_n([gate if i == target_qubit else np.eye(2) for i in range(n_qubits)])
#     return gate_expanded @ statevector
