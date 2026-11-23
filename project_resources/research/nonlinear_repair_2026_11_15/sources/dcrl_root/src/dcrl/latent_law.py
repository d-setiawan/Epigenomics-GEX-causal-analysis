"""Sparse latent probabilities and direct sampling without state enumeration."""
from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np


@dataclass
class SparseLatentLaw:
    """Mixture of stored state masses and a uniform law over K binary bits.

    A state's probability is masses.get(state, 0) + uniform_mass / 2**K.
    The uniform component represents smoothing without allocating its support.
    No implicit NumPy conversion is provided: dense export is explicit.
    """
    K: int
    masses: dict
    uniform_mass: float = 0.0

    def __post_init__(self):
        if not isinstance(self.K, Integral) or self.K < 1:
            raise ValueError('K must be a positive integer')
        self.K = int(self.K)
        cleaned = {}
        for state, mass in self.masses.items():
            if not isinstance(state, Integral) or state < 0 or int(state).bit_length() > self.K:
                raise ValueError('State is outside the binary state space')
            mass = float(mass)
            if not math.isfinite(mass) or mass < 0:
                raise ValueError('State masses must be finite and nonnegative')
            if mass:
                cleaned[int(state)] = mass
        self.masses = cleaned
        self.uniform_mass = float(self.uniform_mass)
        if not math.isfinite(self.uniform_mass) or self.uniform_mass < 0:
            raise ValueError('Uniform mass must be finite and nonnegative')
        total = math.fsum(self.masses.values()) + self.uniform_mass
        if not math.isclose(total, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Total probability must equal one')

    @classmethod
    def from_counts(cls, K, counts, L_final=0.0):
        """Preserve the existing PSAEM finalization law, including smoothing.

        With L_final > 0 the historical code adds L_final/U to EVERY state,
        where U is the number of unvisited states. Its aggregate contribution
        is represented here as one uniform-mixture weight. With zero total
        count, the historical fallback is a uniform law.
        """
        if not isinstance(K, Integral) or K < 1 or not math.isfinite(L_final) or L_final < 0:
            raise ValueError('Require K >= 1 and finite L_final >= 0')
        K = int(K)
        weights = {}
        for state, value in counts.items():
            if not isinstance(state, Integral) or state < 0 or int(state).bit_length() > K:
                raise ValueError('State is outside the binary state space')
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Counts must be finite and nonnegative')
            weights[int(state)] = value
        uniform_weight = 0.0
        total_states = 1 << K  # scalar integer only, never an array
        if L_final and len(weights) < total_states:
            # Algebraically L_final * 2**K / U, avoiding float overflow at large K.
            uniform_weight = L_final / (1.0 - len(weights) / total_states)
        total = math.fsum(weights.values()) + uniform_weight
        if total == 0:
            return cls(K, {}, 1.0)
        return cls(K, {s: w / total for s, w in weights.items() if w}, uniform_weight / total)

    @property
    def stored_state_count(self):
        return len(self.masses)

    def probability(self, state):
        if not isinstance(state, Integral) or state < 0 or int(state).bit_length() > self.K:
            return 0.0
        return self.masses.get(int(state), 0.0) + math.ldexp(self.uniform_mass, -self.K)

    def sample(self, sample_size):
        """O(S + M K) work and storage with S stored states and M output rows."""
        if not isinstance(sample_size, Integral) or sample_size < 0:
            raise ValueError('sample_size must be a nonnegative integer')
        sample_size = int(sample_size)
        samples = np.empty((sample_size, self.K), dtype=float)
        if sample_size == 0:
            return samples
        states = list(self.masses)
        weights = [self.masses[s] for s in states]
        if self.uniform_mass:
            weights.append(self.uniform_mass)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()
        component_counts = np.random.multinomial(sample_size, weights)
        start = 0
        for state, count in zip(states, component_counts):
            count = int(count)
            if count:
                pattern = [(state >> shift) & 1 for shift in range(self.K - 1, -1, -1)]
                samples[start:start + count] = pattern
                start += count
        uniform_count = int(component_counts[-1]) if self.uniform_mass else 0
        if uniform_count:
            samples[start:] = np.random.randint(0, 2, size=(uniform_count, self.K))
        return samples

    def to_dense(self):
        """Explicit compatibility export; costs O(2**K) time and memory."""
        dense = np.full((1 << self.K, 1), math.ldexp(self.uniform_mass, -self.K))
        for state, mass in self.masses.items():
            dense[state, 0] += mass
        return dense

    def save_npz(self, path):
        """Portable sparse export without object arrays or pickle."""
        states = list(self.masses)
        np.savez_compressed(path, K=np.array(self.K),
                            states=np.array([format(s, 'x') for s in states], dtype=str),
                            masses=np.array([self.masses[s] for s in states], dtype=float),
                            uniform_mass=np.array(self.uniform_mass))

    @classmethod
    def load_npz(cls, path):
        with np.load(path, allow_pickle=False) as data:
            states = [int(str(s), 16) for s in data['states']]
            return cls(int(data['K']), dict(zip(states, data['masses'])), float(data['uniform_mass']))
