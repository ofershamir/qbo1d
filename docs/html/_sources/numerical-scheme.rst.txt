================
Numerical scheme
================

The code solves the forced advection-diffusion equation in one space dimension.
The equation is discretized in time using is a semi-implicit scheme, combining
an implicit scheme for the advection/diffusion terms (centered in both time and
space) and a (explicit) leapfrog scheme for the source term.

Before discretization, the forced advection-diffusion equation is

.. math::
	:label: before-discretization

	\frac{\partial u}{\partial t} + w \frac{\partial u}{\partial z} = \kappa
	\frac{\partial^2 u}{\partial z^2} - S(t,z,u),

where :math:`t, z`, and :math:`u` are the time, space (vertical level), and
the unknown (zonal wind), respectively, :math:`w` and :math:`\kappa` are the
constant advection and diffusion coefficients, and :math:`S(t,z,u)` is the
source term.

After discretization, the equation can be written as

.. math::
	:label: after-discretization

	\left[ \mathbf{I} + \Delta t \left( w \mathbf{D1} -
	\kappa \mathbf{D2} \right) \right] \mathbf{u}^{\tau+1} =
	\left[ \mathbf{I} - \Delta t \left( w \mathbf{D1} -
	\kappa \mathbf{D2} \right) \right] \mathbf{u}^{\tau-1} -
	2 \Delta t \mathbf{S}^{\tau},

where :math:`\Delta t` is the time step,
:math:`\mathbf{u}^{\tau} = (u^{\tau}_{1}, ..., u^{\tau}_{N})^{T}` and
:math:`\mathbf{S}^{\tau} = (S^{\tau}_{1}, ..., S^{\tau}_{N})^{T}` are the
vectorsof discretized unknown and source term at time step :math:`\tau`,
:math:`\mathbf{I}` is the :math:`N \times N` identity, and
:math:`\mathbf{D1}` and :math:`\mathbf{D2}` are the differentiation
matrices for the first and second order derivatives defined here as

.. math::
	:nowrap:

	\begin{equation}
	D1_{ij} =
	\begin{cases}
		0 &\text{for $i=1$, $1 \leq j \leq N$}, \\
		(\delta_{i \, j+1} + \delta_{i \, j-1}) / 2 \Delta t &\text{for
		 $1 < i < N$, $1 \leq j \leq N$}, \\
		0 &\text{for $i=N$, $1 \leq j \leq N$},
	\end{cases}
	\end{equation}

.. math::
	:nowrap:

	\begin{equation}
	D2_{ij} =
	\begin{cases}
		0 &\text{for $i=1$,  $1 \leq j \leq N$}, \\
		(\delta_{i \, j+1} - 2\delta_{i \, j} +
		\delta_{i \, j-1}) / \Delta t^2 & \text{for
		 $1 < i < N$, $1 \leq j \leq N$},  \\
		0 &\text{for $i=N$, $1 \leq j \leq N$}.
	\end{cases}
	\end{equation}

Note (1) Equation :eq:`after-discretization` does not lead to singular matrices thanks
to the identity. (2) The matrix on the LHS side is constant and can be inverted
only once. (3) Zeroing-out the first and last rows of :math:`\mathbf{D1}` and
:math:`\mathbf{D2}`, implies that the tendency at the boundaries is determined
by the source term, i.e.

.. math::

	u_{\{1,N\}}^{\tau+1} =
	u_{\{1,N\}}^{\tau-1} -
	2 \Delta t S_{\{1,N\}}^{\tau}.

Consider now the source term in the analytic model

.. math ::

	S(u, z) = - \frac{1}{\rho} \sum_{i} g_{i}(u, z) A_{i}
	\exp\left\{ - \int_{z_1}^{z} g_{i}(u, z') \, dz' \right\},

where

.. math ::

	g_{i}(u, z) = \frac{\alpha(z) N}{k(u-c_{i})^2} .

At the bottom :math:`z=z_1`:

.. math ::

	S(u_1, z_1) = - \frac{\alpha(z) N}{\rho} \sum_{i}
	\frac{A_{i}}{k(u_1-c_{i})^2}.

If :math:`u_1=0`, the phase speeds :math:`c_i` are symmetric about
:math:`c=0`, and the amplitudes are antisymmetric with respect to the phase
speed, i.e. :math:`A(-c)=-A(c)`, then the bottom source is also zero. Then, if
the initial wind and wind tendency at the bottom are zero, they will remain
zero.

At the top, the source term does not vanish for all :math:`t` analytically. Yet,
numerically, the source terms is computed by applyting :math:`\mathbf{D1}` to
the flux, which zeros-out the source term at the top. Then, if
the initial wind and wind tendency at the top are zero, they will remain zero.
