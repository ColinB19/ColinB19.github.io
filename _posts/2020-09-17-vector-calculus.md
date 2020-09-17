---
title: Vector Calculus
date: 2020-09-17 00:15:00 +/-0800
categories: [Physics and Math, Math]
tags: [vectors, calculus, learning]     # TAG names should always be lowercase
image: /assets/img/posts/vector-calculus/hurricane.jpg
math: true
---

# Introduction
--------------

Vectors are important mathematical tools in Physics. They describe physical quantities in both magnitude and direction; however, they also have other important properties which we will explore in this handout. Examples of physical quantities with vector character are: velocity, momentum, force, fields (electric, magnetic, gravitational etc.) and many others. Due to its prevalence in physical systems, vector analysis becomes one of the most fundamental mathematical tools in all of physics.

To handle vectors we must discuss coordinate systems. The most useful of which is the Cartesian coordinate system (other coordinate systems include spherical, cylindrical, parabolic, etc.). This is a system with three orthogonal axes, all of which describe independent directions. The notation for these axes are "$x$", "$y$" and "$z$" in general; however they also take other forms.

Let $\vec{v}$, $\vec{u}$ be vectors with a Cartesian representation. These vectors will then have three components:

$$ \vec{v}=(v_{x}, v_{y}, v_{z}), $$

$$ \vec{u}=(u_{x}, u_{y}, u_{z}), $$

such that $u_{i}$ and $v_{i}$ are real scalar variables (complex vectors do exist; however, we will not be treating them in this discussion). These variables are not required to be constants, as in the equations for kinematic motion:

$$ \vec{v}(t)=\vec{v}_{0}+\vec{a}t, $$

$$ \Delta\vec{s}(t)=\frac{1}{2}\vec{a}t^2+\vec{v}_{0}t, $$

where $\vec{s}$ is the position vector $\vec{s}=(x, y, z)$.

# Basic Vector Algebra
----------------------

As one can see from Eq.'s 3 and 4, vectors can be added together. This can be done in the following way:

$$ \vec{u}+\vec{v}=(u_{x}, u_{y})+(v_{x}, v_{y})=(u_{x}+v_{x}, u_{y}+v_{y}). $$

An entirely general version of this formula takes the form:

$$ \sum_{i = 1}^{N} \vec{v}_i = \left( \sum_{i = 1}^{N} v_{x,i} , \sum_{i = 1}^{N} v_{y,i}, \sum_{i = 1}^{N} v_{z,i} \right) $$

where $\vec{v}_i$ is the $i^{\text{th}}$ vector and $N$ is the number of vectors that are being summed.
One can also see from Eq.’s 3 and 4 that scalars and vectors have a defined product:

$$ \psi \left( x, y, z \right) \cdot \vec{u} = \psi \left( x, y, z \right)\cdot \left(u_x, u_y, u_z \right) = \left(\psi u_x, \psi u_y, \psi u_z \right) $$

where $\psi \left( x, y, z \right)$ is some arbitrary scalar function (a scalar function is a function, which may or may not depend on any of the three Cartesian variables, which produces a scalar output). As noted above, vectors have magnitude. This magnitude is found with Pythagoras’s theorem. Note that this theorem is only applicable in a Cartesian coordinate system. One must refer to new definitions
for magnitude when the system is more complicated.

> At this point you may be asking yourself why I have been making the restriction to a Cartesian coordinate system. Bravo! If the system is not Cartesian, there is not guarantee that these definitions will hold (however, they could). For example, in the spherical coordinate system these identities do not hold as the metric is not the identity matrix.

<!-- While in your particular class you will likely only calculate the sum of 2-D vectors, it is important to note that the sum of vectors of arbitrary dimension is just a vector whose components are made up of the summation of the components of the vectors being summed as in Eq. 6. Vectors can also be multiplied together; however, this is a more complicated process. In general, vectors can be multiplied together to create a scalar, another vector, or some higher order object which we will not discuss.

> This higher order object is called a tensor and is an abstraction of the concept of a vector. -->

$$ |\vec{v}|^2 = v_1^2 + v_2^2 + v_3^2 + ... v_M^2= \sum_{i = 1}^M v_i^2 $$

where $M$ is the dimensionality of the space (for 1-D motion $M$ = 1, for 2-D motion $M$ = 2, etc.) You are most likely saying, “Colin, all this is great, but how in the world do I multiply vectors?” For this level of vector mathematics, there are two ways to multiply vectors.

The first method of vector multiplication that we will discuss is the dot product (scalar product). This is a way of multiplying two vectors to creates a scalar. It is used in many applications in physics, for example, the amount of energy given to a system in which a force is applied over a distance is called work. In general, the work is given by:

$$ W = \int_C \vec{F} \cdot d\vec{\ell} $$

where $W$ is the work done, $\vec{F}$ is the force applied over the contour $C$. Note that if the force is constant along the contour (e.g. constant force along a straight line) the equation reduces to

$$  W=\vec{F}\cdot\vec{L}=|\vec{F}||\vec{L}|\cos\theta, $$

where $\theta$ is the angle between the two vectors, $\vec{F}$ is the force applied and $\vec{L}$ is the distance over which the force is applied. In general, the dot product (in a Cartesian coordinate system) can be written as:

$$ \vec{u}\cdot\vec{v}=u_{1}v_{1}+u_{2}v_{2}+...=|\vec{u}||\vec{v}|\cos{\theta} $$

where, again, $\vec{u}$ and $\vec{v}$ are arbitrary Cartesian vectors of dimension $N$ (again, you will likely only encounter vectors with dimension 2 and rarely 3) with components $u_i$ and $v_i$ respectively.

The second method of multiplying two vectors is the cross product. Contrary to the dot product, the cross product produces a vector and thus is often called the vector product. The cross product can be defined in a number of ways; however, we will stick to two. The first is the determinant:

$$ \vec{u} \times \vec{v} = \begin{vmatrix}
                \:\hat{i} & \:\hat{j} & \:\hat{j}\:\\
                \:u_x & \:u_y & \:u_z\:\\
                \:v_x & \:v_y & \:v_z\:\\
            \end{vmatrix}
 = \hat{i} (u_y v_z - u_z v_y) - \hat{j} (u_x v_z - u_z v_x) + \hat{k} (u_x v_y - u_y v_x),
$$

where $\hat{i}, \hat{j}, \hat{k}$ are the Cartesian basis vectors (discussed [later](#basis-vectors)). While this is the more mathematically detailed way to calculate the cross product, it is often useful to calculate the magnitude of the cross product then use your intuition to discern the direction (using the right-hand rule):

$$  |\vec{u}\times\vec{v}|=|\vec{u}||\vec{v}|\sin\theta, $$

where $\theta$ is the angle between the two vectors. The right hand rule states that the direction of a vector produced by a cross product can be determined using your right hand. First, hold out your right hand with your fingers out and your thumb up. Next, direct your palm in the direction of the first vector in the cross product. Next, curl your fingers in the direction of the second vector in the cross product. After doing these steps your thumb should be pointed in the direction of the resultant vector.

## Some Basic Trigonometry

Often, you will not be given a vector in terms of it's $x$ and $y$ components. In introductory physics courses it is common to be given a vector in terms of its magnitude and its orientation with respect to some reference axis (e.g. the velocity is $30$ mph at direction $30&deg;$ clockwise from the $x$-axis). If given this information, how does one calculate the components of the vector in the coordinate system? Resolving a vector into its components ends up being a trigonometry problem. As shown in Figure 2, a vector can be represented by two vectors aligned with the coordinate axes.

These three vectors form a right triangle on which trigonometry can be done. While there are many trigonometric identities that can be applied to the vectors, these are the most useful:

$$ u^2=u_{x}^2+u_{y}^2 $$

$$ \frac{u_y}{u}=\sin{\theta} $$

$$  \frac{u_x}{u}=\cos{\theta} $$

$$  \frac{u_y}{u_x}=\tan{\theta}. $$

>All quantities expressed without vector character are magnitudes. Also note that most of this analysis only applies in flat space. I might post about General Relativity in the future; this is where we will address non-flat coordinate systems.

Thus, given the components of a vector one could calculate both its magnitude and direction and vice versa. It is important to note that "cosine" and "sine" are not associated with $x$ and $y$ respectively. The fact that they are defined this way in our coordinate system is an arbitrary choice of construction. One could have just as easily defined the angle from the $y$-axis as the angle that we are concerned with.

# Basic Vector Calculus
-----------------------
In this section, we will discuss calculus concepts such as integration, the gradient, the curl, and the divergence. Integration of a vector quantity alone is a fairly simple operation. One just integrates the components individually:
$$
\begin{align*}
    \int \vec{v}(t) dt = \int (v_1 (t) \hat{i} + v_2 (t) \hat{j}& + v_3 (t) \hat{k})dt\\
    &= \hat{i}\int v_1 (t) dt + \hat{j}\int v_2 (t) dt + \hat{k}\int v_3 (t) dt,
\end{align*}
$$

where $\vec{v}$ is explicitly a function of some parameter t. Of course this can be generalized to N dimensional vectors as we have done previously in this handout. It is also useful to differentiate vectors. Normal differentiation of vectors is quite simple:

$$
  \frac{d}{dx} \vec{v} = \left(\frac{d}{dx} v_x, \frac{d}{dx} v_y,\frac{d}{dx} v_z \right)
$$

There are also other ways to differentiate vectors, we will focus on two of them. Before we do that, let us first define the differential vector operator (or the gradient operator). The gradient operator (in Cartesian coordinates) is defined as:

# Basis Vectors
---------------

# The Divergence and Stokes' Theorems
-------------------------------------

## Divergence Theorem

## Stokes' Theorem
