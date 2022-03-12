=================
Model description
=================

The model is a hybrid of the models ued by Holton and Lindzen (1972) and
Plumb (1977). The equation is:

.. math ::
	:label: model-equation

	\frac{\partial u}{\partial t} +
		w \frac{\partial u}{\partial z} -
		K \frac{\partial^2 u}{\partial z^2}
		= G(z,t) - S(u, z),

where :math:`u` is the unknown zonal wind, :math:`t` and :math:`z` are the time
and vertical coordinate, :math:`w` and :math:`K` are constant vertical advection
and diffusivity, :math:`G(z, t)` is the semi-annual oscilation (see Holton and
Lindzen, 1972, defaults to zero in the code), and :math:`S(u, z)` is the wave
forcing consisting of a sum of monochromatic waves, and parameterizing the
momentum deposition at critical levels as follows:

.. math ::
	:label: model-description-drag-definition

	S(u, z) = \frac{1}{\rho} \frac{\partial}{\partial z} F(u, z),

where :math:`\rho(z)` is the density profile and the wave flux :math:`F(u, z)`
is parameterized as follows:

.. math ::
	:label: model-description-flux-definition

	F(u, z) = \sum_{i} A_{i}
	\exp\left\{ - \int_{z_1}^{z} g_{i}(u, z') \, dz' \right\},

where :math:`A_{i}` are the wave amplitudes and

.. math ::

	g_{i}(u, z) = \frac{\alpha(z) N}{k_{i}(u-c_{i})^2},

where :math:`k_{i}` and :math:`c_{i}` are the wavenumbers and phase speeds,
:math:`N` is the Brunt-Vaisala frequency taken here to be constant, and
:math:`\alpha(z)` is the wave dissipation due to infrared cooling (see Holton
and Lindzen, 1972).

Note, the sign convention used in the code applies :math:`-S(u, z)` on the RHS of the
equation, hence the lack of minus sign in the definition of the drag
in :eq:`model-description-drag-definition`.
