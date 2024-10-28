import numpy as np

""" Contains definitions for standard quantum gates using matrix representations """
class Gates:
    # Define basic quantum gates as class variables
    I = np.array([[1, 0], [0, 1]] )
    X = np.array([[0, 1], [1, 0]] )
    Y = np.array([[0, -1j], [1j, 0]] )
    Z = np.array([[1, 0], [0, -1]] )
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]] )  # Hadamard Gate
    CNOT = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]] )





















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
