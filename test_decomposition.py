import numpy as np

I2 = np.eye(2, dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Z = np.array([[1, 0],
              [0,-1]], dtype=complex)

def Rx(theta):
    return np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*X

def Rz(theta):
    return np.cos(theta/2)*I2 - 1j*np.sin(theta/2)*Z

def kron(A, B):
    return np.kron(A, B)

# 2-qubit gates
iSWAP = np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]], dtype=complex)

CZ = np.diag([1, 1, 1, -1]).astype(complex)

# Simplified single-qubit blocks (up to global phases)
A = Rx(np.pi/2) @ Rz(np.pi/2)
B = Rz(-np.pi/2) @ Rx(np.pi/2)
C = Rz(np.pi/2)

U = kron(I2, A) @ iSWAP @ kron(B, C) @ iSWAP @ kron(I2, A)

def equal_up_to_global_phase(U, V, tol=1e-12):
    idx = np.argmax(np.abs(V))
    phase = U.flatten()[idx] / V.flatten()[idx]
    phase /= np.abs(phase)
    err = np.linalg.norm(U - phase * V)
    return phase, err

phase, err = equal_up_to_global_phase(U, CZ)

print("global phase =", phase)
print("Frobenius error ||U - phase*CZ|| =", err)
assert err < 1e-10
print("OK: equal to CZ up to global phase.")
