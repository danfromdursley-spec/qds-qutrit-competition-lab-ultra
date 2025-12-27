#!/data/data/com.termux/files/usr/bin/python
"""
QDS Qutrit/Qubit Competition Lab — Core v0

- Generic d-level (qudit) register
- Supports d=2 (qubit) and d=3 (qutrit) out of the box
- Statevector and density-matrix modes
- Single-site gate embedding
- Z-basis measurement with collapse
"""

import numpy as np


class QuditSystem:
    def __init__(self, d: int, n: int, mode: str = "statevector"):
        assert d >= 2, "Dimension d must be >= 2"
        assert n >= 1, "Need at least one qudit"
        assert mode in ("statevector", "density"), "mode must be 'statevector' or 'density'"

        self.d = d
        self.n = n
        self.mode = mode

        dim = d ** n
        if mode == "statevector":
            # |0...0>
            self.state = np.zeros(dim, dtype=complex)
            self.state[0] = 1.0 + 0.0j
        else:
            # ρ = |0...0><0...0|
            self.state = np.zeros((dim, dim), dtype=complex)
            self.state[0, 0] = 1.0 + 0.0j

    @property
    def dim(self) -> int:
        return self.d ** self.n

    # ---- Gate plumbing -------------------------------------------------

    def _embed_local(self, op: np.ndarray, site: int) -> np.ndarray:
        """
        Embed a single-qudit operator 'op' acting on qudit 'site'
        (0-based index) into the full n-qudit Hilbert space.
        """
        assert 0 <= site < self.n, "site index out of range"

        eye = np.eye(self.d, dtype=complex)
        mats = []
        for i in range(self.n):
            mats.append(op if i == site else eye)

        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)

        return result

    def apply_gate(self, op: np.ndarray, site: int) -> None:
        """Apply a single-site gate to the register."""
        U = self._embed_local(op, site)
        if self.mode == "statevector":
            self.state = U @ self.state
        else:
            self.state = U @ self.state @ U.conj().T

    # ---- Probabilities and measurements --------------------------------

    def probs(self) -> np.ndarray:
        """Return normalized outcome probabilities in the computational basis."""
        if self.mode == "statevector":
            p = np.abs(self.state) ** 2
        else:
            p = np.real(np.diag(self.state))

        p = np.clip(p, 0.0, 1.0)
        total = p.sum()
        if total <= 0:
            # Failsafe: uniform if something went numerically weird
            return np.ones_like(p) / len(p)
        return p / total

    def measure_z(self, collapse: bool = True):
        """
        Measure in the computational (Z) basis.

        Returns:
            (outcome_string, index)
        """
        p = self.probs()
        idx = np.random.choice(len(p), p=p)

        # Represent index in base-d with n digits (left-padded with zeros)
        outcome_digits = np.base_repr(idx, base=self.d).zfill(self.n)

        if collapse:
            if self.mode == "statevector":
                new_state = np.zeros_like(self.state)
                new_state[idx] = 1.0 + 0.0j
                self.state = new_state
            else:
                proj = np.zeros_like(self.state)
                proj[idx, idx] = 1.0 + 0.0j
                self.state = proj

        return outcome_digits, idx


# ---- Gate libraries ----------------------------------------------------


def qubit_gates():
    """Standard 2-level gates: I, X, Z, H."""
    X = np.array([[0, 1],
                  [1, 0]], dtype=complex)
    Z = np.array([[1, 0],
                  [0, -1]], dtype=complex)
    H = (1 / np.sqrt(2)) * np.array([[1, 1],
                                     [1, -1]], dtype=complex)
    I = np.eye(2, dtype=complex)
    return {"I": I, "X": X, "Z": Z, "H": H}


def qutrit_gates():
    """
    Basic 3-level gates:
    - I: identity
    - X: cyclic shift |0>→|1>, |1>→|2>, |2>→|0>
    - Z: phase diag(1, ω, ω²) with ω = e^{2πi/3}
    """
    d = 3
    I = np.eye(d, dtype=complex)

    # Cyclic shift
    X = np.zeros((d, d), dtype=complex)
    for i in range(d):
        X[(i + 1) % d, i] = 1.0 + 0.0j

    omega = np.exp(2j * np.pi / d)
    Z = np.diag([omega ** 0, omega ** 1, omega ** 2])

    return {"I": I, "X": X, "Z": Z}


# ---- Minimal demo / smoke test -----------------------------------------


def demo():
    """
    Tiny self-test:
    - Build a single qutrit
    - Apply X then Z
    - Measure in Z basis
    """
    print("=== QDS Qutrit Core v0 demo ===")
    np.random.seed(42)

    sys = QuditSystem(d=3, n=1, mode="statevector")
    gates = qutrit_gates()

    # Start: |0>
    print("Initial state (amplitudes):", sys.state)

    # |0> -> X -> |1>
    sys.apply_gate(gates["X"], site=0)
    print("After X:", np.round(sys.state, 4))

    # |1> -> Z -> ω|1>
    sys.apply_gate(gates["Z"], site=0)
    print("After Z:", np.round(sys.state, 4))

    outcome, idx = sys.measure_z()
    print("Measurement outcome:", outcome, "(index:", idx, ")")
    print("Final state (post-measure):", sys.state)


if __name__ == "__main__":
    demo()
