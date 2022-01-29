================
Numerical scheme
================

The code solves the forced advection-diffusion equation in one space dimension
with vanishing Dirichlet boundary conditions (B.C. are hard-coded). The equation
is discretized in time using is a semi-implicit scheme, combining an implicit
scheme for the advection/diffusion terms (centered in both time and space) and a
(explicit) leapfrog scheme for the source term.

Before discretization, the forced advection-diffusion equation is

.. math::

	\frac{\partial u}{\partial t} + w \frac{\partial u}{\partial z} = \kappa
	\frac{\partial^2 u}{\partial z^2} - S(t,z,u),

where :math:`t, z`, and :math:`u` are the time, space (vertical level), and
the unknown (zonal wind), respectively, :math:`w` and :math:`\kappa` are the
constant advection and diffusion coefficients, and :math:`S(t,z,u)` is the
source term.

After discretization, the equation can be written as

.. math::

	\left[ \mathbf{I} + \Delta t \left( w \mathbf{D1} -
	\kappa \mathbf{D2} \right) \right] \mathbf{u}^{\tau+1} =
	\left[ \mathbf{I} - \Delta t \left( w \mathbf{D1} -
	\kappa \mathbf{D2} \right) \right] \mathbf{u}^{\tau-1} -
	2 \Delta t \mathbf{S}^{n},

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

The boundary conditions are implemented by: (1) Zeroing out the first and last
rows of :math:`\mathbf{D1}` and :math:`\mathbf{D2}`. (2) Zeroing out
the first and last components of the source term. (3) Starting with a compatible
initial condition vanishing at the boundaries. Note that (1) does not lead to
singular matrices thanks to the identity. Finally, the matrix on the LHS side is
constant and can be inverted only once.


