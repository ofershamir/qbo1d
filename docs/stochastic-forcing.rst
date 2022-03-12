==================
Stochastic forcing
==================

Recall the definition of the forcing (see
:ref:`Model description`):

.. math ::
	:label: stochastic-forcing-drag-definition

	S(u, z) = \frac{1}{\rho} \frac{\partial}{\partial z} F(u, z),

where the wave flux :math:`F(u, z)` is parameterized as follows:

.. math ::
	:label: stochastic-forcing-flux-definition

	F(u, z) = \sum_{i} A_{i}
	\exp\left\{ - \int_{z_1}^{z} g_{i}(u, z') \, dz' \right\},

and where

.. math ::

	g_{i}(u, z) = \frac{\alpha(z) N}{k_{i}(u-c_{i})^2} .

Note, at the bottom :math:`z=z_1`, :math:`F(u, z=z1) = \sum_{i} A_{i}`.
We assume that the source level of the waves coincide with model's bottom and
denote by :math:`F_{S0} = \sum_{i} A_{i}` the net source flux.

To define the wave spectrum we specify the wavenumbers and phase speeds and let
:math:`A=A(k,c)`. By default, the code sums the fluxes over 20 waves with equal
wavenumbers :math:`k_i=2`, and (piecewise) linearly spaced phase speeds

.. math ::
	c_i = torch.hstack([torch.arange(-100, 0, 10), torch.arange(10, 110, 10)]),

excluding :math:`c=0`.

The amplitude depent on the phase speed as in Eq. (17) of Alexander and
Dunkerton (1999), centered around :math:`c=0`, i.e.:

.. math ::

	A(c) = \text{sgn}(c) B_m
	\exp\left[- \ln{2} \left(\frac{c}{c_w}\right)^2 \right].

The stochasticity enters by letting the total flux at the source level
:math:`F_{S0}` and spectral width :math:`c_{w}` be random variables
(so :math:`B_{m}` in the above equation is chosen accordingly to the drawn
:math:`F_{S0}`). Physically, one can think of convection generating
waves packets randomly. Following feedback from Joan and
Martina, more vigorous convection leads also to a broader spectrum, meaning
:math:`F_{S0}` and :math:`c_{w}` are positively correlated.
Therefore, we use a 5-parameter distribution representing their means,
varainces, and the correlation between the two.
At each time step two numbers are drawn from a bivariate normal
distribution. Next those numbers are mapped to a bivariate log-normal
distribution, from which :math:`F_{S0}` and :math:`c_{w}` are drawn.
The reason for mapping to a bivariate log-normal distribution is that we want
:math:`F_{S0}` and :math:`c_{w}` to be strictly positive numbers. Fortunately,
due to the Central Limit Theorem, the particular choice of distribution
does not matter here because after the sum in
:eq:`stochastic-forcing-flux-definition` the total flux will be normalay
distributed (actually it will be a sum of two Gaussians because there are two
"species").
