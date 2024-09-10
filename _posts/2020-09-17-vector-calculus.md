---
title: Vector Calculus
date: 2020-09-17 00:15:00 +/-0800
categories: [Physics and Math, Math]
tags: [vectors, calculus, learning]     # TAG names should always be lowercase
image: /assets/img/posts/vector-calculus/hurricane.jpg
excerpt: A brief introduction (or review if you're familiar already) to vector calculus. We go over basic vector operations, how calculus can be done with vectors, and some useful vector calculus theorems.
math: true
layout: post
image_alt: a
---

# Introduction
--------------

On this blog I will not only be posting about data science but about physics as well. Thus, I will often use some mathematical formalism in those posts that I will not want to explain. These posts about mathematics serve as tools to explain some of the physics and data science I will be talking about in the future. This post has been adapted from a handout that I wrote for undergraduate students in physics during my time as a grad student.

Vectors are important mathematical tools in Physics. They describe physical quantities in both **magnitude** and **direction**; however, they also have other important properties which we will explore in this handout. Examples of physical quantities with vector character are: velocity, momentum, force, fields (electric, magnetic, gravitational etc.) and many others. Due to its prevalence in physical systems, vector analysis becomes one of the most fundamental mathematical tools in all of physics.

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

As one can see from the equations of kinematic motion, vectors can be added together. This can be done in the following way:

$$ \vec{u}+\vec{v}=(u_{x}, u_{y})+(v_{x}, v_{y})=(u_{x}+v_{x}, u_{y}+v_{y}). $$

An entirely general version of this formula takes the form:

$$ \sum_{i = 1}^{N} \vec{v}_i = \left( \sum_{i = 1}^{N} v_{x,i} , \sum_{i = 1}^{N} v_{y,i}, \sum_{i = 1}^{N} v_{z,i} \right) $$

where $\vec{v}_i$ is the $i^{\text{th}}$ vector and $N$ is the number of vectors that are being summed.
One can also see from the equations of kinematic motion that scalars and vectors have a defined product:

$$ \psi \left( x, y, z \right) \cdot \vec{u} = \psi \left( x, y, z \right)\cdot \left(u_x, u_y, u_z \right) = \left(\psi u_x, \psi u_y, \psi u_z \right) $$

where $\psi \left( x, y, z \right)$ is some arbitrary scalar function (a scalar function is a function, which may or may not depend on any of the three Cartesian variables, which produces a scalar output). As noted above, vectors have magnitude. This magnitude is found with Pythagoras’s theorem. Note that this theorem is only applicable in a Cartesian coordinate system. One must refer to new definitions
for magnitude when the system is more complicated.

> At this point you may be asking yourself why I have been making the restriction to a Cartesian coordinate system. Bravo! If the system is not Cartesian, there is not guarantee that these definitions will hold (however, they could). For example, in the spherical coordinate system these identities do not hold as the metric is not the identity matrix.

<!-- While in your particular class you will likely only calculate the sum of 2-D vectors, it is important to note that the sum of vectors of arbitrary dimension is just a vector whose components are made up of the summation of the components of the vectors being summed as in Eq. 6. Vectors can also be multiplied together; however, this is a more complicated process. In general, vectors can be multiplied together to create a scalar, another vector, or some higher order object which we will not discuss.

> This higher order object is called a tensor and is an abstraction of the concept of a vector. -->

$$ |\vec{v}|^2 = v_1^2 + v_2^2 + v_3^2 + ... v_M^2= \sum_{i = 1}^M v_i^2 $$

where $M$ is the dimensionality of the space (for 1-D motion $M$ = 1, for 2-D motion $M$ = 2, etc.) You are most likely saying, “Colin, all this is great, but how in the world do I multiply vectors?” For this level of vector mathematics, there are two ways to multiply vectors.

The first method of vector multiplication that we will discuss is the **dot product** (scalar product). This is a way of multiplying two vectors to creates a scalar. It is used in many applications in physics, for example, the amount of energy given to a system in which a force is applied over a distance is called *work*. In general, the work is given by:

$$ W = \int_C \vec{F} \cdot d\vec{\ell} $$

where $W$ is the work done, $\vec{F}$ is the force applied over the contour $C$. Note that if the force is constant along the contour (e.g. constant force along a straight line) the equation reduces to

$$  W=\vec{F}\cdot\vec{L}=|\vec{F}||\vec{L}|\cos\theta, $$

where $\theta$ is the angle between the two vectors, $\vec{F}$ is the force applied and $\vec{L}$ is the distance over which the force is applied. In general, the dot product (in a Cartesian coordinate system) can be written as:

$$ \vec{u}\cdot\vec{v}=u_{1}v_{1}+u_{2}v_{2}+...=|\vec{u}||\vec{v}|\cos{\theta} $$

where, again, $\vec{u}$ and $\vec{v}$ are arbitrary Cartesian vectors of dimension $N$ (again, you will likely only encounter vectors with dimension 2 and rarely 3) with components $u_i$ and $v_i$ respectively.

The second method of multiplying two vectors is the **cross product**. Contrary to the dot product, the cross product produces a vector and thus is often called the vector product. An example of a vector product in physics can be found in the *Lorentz force*, which is the amount of force on a charged particle due to a combination of electric and magnetic fields

$$
  \vec{F} = q \vec{E} + q \vec{v} \times \vec{B},
$$

where $q, \vec{E}, \vec{v}, \vec{B}$ are the charge of the particle, the external electric field, velocity of the particle, and external magnetic field respectively. The cross product can be defined in a number of ways; however, we will stick to two. The first is the determinant:

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
In this section, we will discuss calculus concepts such as integration, the **gradient**, the **curl**, and the **divergence**. Integration of a vector quantity alone is a fairly simple operation. One just integrates the components individually:

$$
\begin{align*}
    \int \vec{v}(t) dt = \int (v_1 (t) \hat{i} + v_2 (t) \hat{j}& + v_3 (t) \hat{k})dt\\
    &= \hat{i}\int v_1 (t) dt + \hat{j}\int v_2 (t) dt + \hat{k}\int v_3 (t) dt,
\end{align*}
$$

where $\vec{v}$ is explicitly a function of some parameter $t$. Of course this can be generalized to N dimensional vectors as we have done previously in this handout.
> Since $\hat{i} = (1, 0, 0),\;\hat{j} = (0, 1, 0),\;\hat{k} = (0, 0, 1)$ , one can represent a vector $\vec{v} = (v_1 , v_2 , v_3)$ as a sum of the components of the vector $\vec{v}$ multiplied by their respective basis vector (this is treated [later](#basis-vectors)).

It is also useful to differentiate vectors. Normal differentiation of vectors is quite simple:

$$
  \frac{d}{dx} \vec{v} = \left(\frac{d}{dx} v_x, \frac{d}{dx} v_y,\frac{d}{dx} v_z \right)
$$

There are also other ways to differentiate vectors, we will focus on two of them. Before we do that, let us first define the differential vector operator (or the gradient operator). The gradient operator (in Cartesian coordinates) is defined as:

$$
  \vec{\nabla} = \hat{i} \frac{\partial}{\partial x} + \hat{j} \frac{\partial}{\partial y} + \hat{k} \frac{\partial}{\partial z}
$$

where $\frac{\partial}{\partial ...}$ indicates a **partial derivative** with respect to that specific coordinate (partial derivatives are the same as normal derivatives; however, all variables that are not the differentiating variable are held constant). This operator can be applied to scalar fields in the following way:

$$
  \vec{\nabla} \psi \left( x, y, z \right) = \hat{i} \frac{\partial}{ \partial x} \psi \left( x, y, z \right) + \hat{j} \frac{\partial}{\partial y} \psi \left( x, y, z \right) + \hat{k} \frac{\partial}{\partial z} \psi \left( x, y, z \right)
$$

where $\psi \left( x, y, z \right)$ is some scalar function with dependence on the three Cartesian coordinates. This produces an interesting result! The gradient of a scalar function is a vector! The gradient of a scalar field points in the direction of the greatest increase of the scalar field.


As discussed before, there are several types of vector products. Since the gradient operator is a vector, both the cross product and the scalar product are defined. The scalar product of the gradient operator and a vector is called the divergence.

$$
  \vec{\nabla} \vec{u} (x, y, z) = \frac{\partial}{\partial x} u_x (x, y, z) + \frac{\partial}{\partial y} u_y (x, y, z) +  \frac{\partial}{\partial z} u_z (x, y, z)
$$

where, again, $\vec{u} (x, y, z) $ is an arbitrary vector depending on the three Cartesian coordinates.

> Note that any component of $\vec{u} (x, y, z) $ can depend on any of the coordinates, for example: $\vec{u} (x, y, z)  = (z, x^2, \sin{(y)})$

The divergence of a vector is a measurement of the “flow” in or out of a point in the vector field. A good example of this is a bathtub with drain. The divergence of the vector field describing the fluids motion is a description of how much fluid is flowing into the drain. The cross product of the gradient operator with a vector is called the curl of that vector.

$$ \vec{\nabla} \times \vec{u} = \begin{vmatrix}
                \:\hat{i} & \:\hat{j} & \:\hat{j}\:\\
                \:\frac{\partial}{\partial x}  & \:\frac{\partial}{\partial y} & \:\frac{\partial}{\partial z}\:\\
                \:u_x & \:u_y & \:u_z\:\\
            \end{vmatrix}.
$$

The curl of a vector field is a measurement of how much that field is rotating about a point. As in the bathtub example, if the fluid is not only flowing into the drain but rotating around it, then the vector field describing the fluids motion will have both non-zero divergence and non-zero curl. The product differentiation rule is the same for the gradient operator:

$$
  \vec{\nabla} (\psi \gamma) = \gamma \vec{\nabla} \psi + \psi \vec{\nabla} \gamma,
$$

where $\psi$ and $\gamma$ are arbitrary scalar functions. The chain rule is a more complicated operation so we will not go into it here. Some more advanced vector calculus theorems will be handled in the following sections.

# Basis Vectors
---------------
Several times in this handout the vectors $\hat{i}, \hat{j}, \hat{k}$ (also known as $ \{ \hat{x}, \hat{y}, \hat{z}\}$ and $ \{ \hat{e}_1, \hat{e}_2, \hat{e}_3\}$) were referenced as the basis vectors for the 3-D Cartesian coordinate system. This means that these three vectors, combined in specific ways, can create ANY other vector in 3-D space. Thus, the three vectors **span** the space. The way to do this is to create any arbitrary vector in the space by summing scalar multiples of all three basis vectors:

$$
  \vec{u} = a \hat{i} + b \hat{j} + c \hat{k},
$$

where $a$, $b$, and $c$ are scalars, and $\vec{u}$ is any arbitrary vector in the Cartesian
coordinate system.

> Remember $\hat{i} = (1, 0, 0),\;\hat{j} = (0, 1, 0),\;\hat{k} = (0, 0, 1)$.

You’ll notice that the Cartesian basis vectors are orthonormal. This means that each vector has unit length, and that the dot product of a Cartesian basis vector with a different Cartesian basis vector is zero (or the vectors are normal to each other):

$$
  \hat{e}_\ell \cdot \hat{e}_k = \delta_{\ell k} = \left \{
  \begin{array}{lll}
    0  \, & \text{if} \,  &  \ell \neq k\\
    1  \, & \text{if} \,  &  \ell = k
  \end{array}
  \right. \,
$$

where $\ell $and $k$ take the values 1, 2, and 3 to filter through all three basis vectors.
Basis vectors exist in all coordinate systems. The spherical and cylindrical
coordinate systems are the most common; however, there exist many.

> For you fans of linear algebra or differential geometry this is not ALWAYS true. However, there always exist vectors that span a coordinate system. There also always exists a metric on any smooth manifold. Also, if you’re concerned about these things then this post is probably not a very useful tool.

# The Divergence and Stokes' Theorems
-------------------------------------

## Divergence Theorem

The Divergence theorem is a theorem about the flux of a vector field. Let $\vec{\Gamma}$ be
some vector field that exists in some volume $V$ bounded by the surface $S$, then
the divergence theorem states

$$
  \int_V \vec{\nabla} \cdot \vec{\Gamma} d^3 x = \oint_S \vec{\Gamma} \cdot \hat{n} d^2 x,
$$

where $d^3 x$ and $d^2 x$ are differential volume and surface elements respectively,
and $\hat{n}$ is the unit outward pointing normal to the surface. This theorem states
that to total divergence of a vector field in some volume is equal to the *flux* of
that vector field through the surface that bounds the volume.

## Stokes' Theorem

Stokes’ theorem is another useful vector calculus theorem used in physics often. Let $\vec{\Gamma}$ be some vector field that exists on some surface $S$ bounded by the curve $C$, then Stokes’ theorem states

$$
\int_S \vec{\Gamma} \cdot \hat{n} d^2 x = \oint_C \vec{\Gamma} \cdot d\vec{\ell},
$$

where $d\vec{\ell}$ is a differential element of the contour $C$ oriented in the counter- clockwise direction. Stokes’ theorem states that the total curl of a vector field on some surface is equal to the total value of the vector field parallel to the curve bounding that surface.
