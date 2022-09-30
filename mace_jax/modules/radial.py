import e3nn_jax as e3nn
import haiku as hk
import jax.numpy as jnp


class BesselBasis(hk.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis: int = 8):
        super().__init__()

        self.r_max = r_max
        self.num_basis = num_basis

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # [..., 1]
        return e3nn.bessel(x[..., 0], self.num_basis, self.r_max)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis}"
        )


class PolynomialCutoff(hk.Module):
    """
    Generalization of

    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    def __init__(self, r_max: float, n0: int = 5, n1: int = 2):
        super().__init__()
        self.r_max = r_max
        self.n0 = n0
        self.n1 = n1

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return e3nn.poly_envelope(self.n0, self.n1, x_max=self.r_max)(x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n0={self.n0}, n1={self.n1}, r_max={self.r_max})"
        )
