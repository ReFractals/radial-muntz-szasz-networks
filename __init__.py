"""Radial Müntz-Szász Networks (RMN).

Core models:
- RMNDirect: single-center radial specialist
- RMNAngular: radial + angular modes (2D Fourier / 3D real spherical harmonics)
- RMNMultiCenter: superposition with learnable centers
"""

from .parameterizations import make_ordered_bounded_signed
from .utils import compute_radius, safe_pow, log_primitive, spherical_harmonics_2d, spherical_harmonics_3d
from .layers import RadialMuntzEdge, RadialAngularMuntzEdge, MultiCenterRadialEdge
from .models import RMNDirect, RMNAngular, RMNMultiCenter
