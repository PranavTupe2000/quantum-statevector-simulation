import numpy as np

class Gates:
    """
    Contains definitions for standard quantum gates using matrix representations.
    """
    # Define basic quantum gates as class variables
    I = np.array([[1, 0], [0, 1]] )  # Identity Gate
    X = np.array([[0, 1], [1, 0]] )  # Pauli-X (NOT) Gate
    Y = np.array([[0, -1j], [1j, 0]] )  # Pauli-Y Gate
    Z = np.array([[1, 0], [0, -1]] )  # Pauli-Z Gate
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]] )  # Hadamard Gate
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]] )  # CNOT Gate (2-qubit gate)
    
    @staticmethod
    def getGate(var_name):
        # Check if the variable exists in the class
        if hasattr(Gates, var_name.upper()):
            return getattr(Gates, var_name.upper())
        else:
            raise ValueError(f"Gate '{var_name}' does not exist.")

# Example usage for importing and using the gates
if __name__ == "__main__":
    print(Gates.getGate('H'))





















# import numpy as np

# def x_gate():
#     return np.array([[0, 1],
#                      [1, 0]], dtype=complex)

# def h_gate():
#     return (1 / np.sqrt(2)) * np.array([[1,  1],
#                                         [1, -1]], dtype=complex)

# def i_gate():
#     return np.array([[1, 0],
#                      [0, 1]], dtype=complex)

# #Returns the matrix representation of the CNOT gate. (2-qubit gate)
# def cx_gate():
#     return np.array([[1, 0, 0, 0],
#                      [0, 1, 0, 0],
#                      [0, 0, 0, 1],
#                      [0, 0, 1, 0]], dtype=complex)

# # Utility functions for multi-qubit gates
# def kronecker_product(*matrices):
#     result = matrices[0]
#     for matrix in matrices[1:]:
#         result = np.kron(result, matrix)
#     return result
