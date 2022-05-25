### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 6a3016c8-aa11-11ec-3658-0ff8413852e2
begin
	using LinearAlgebra
	using MultivariateStats
	using Markdown
	using Plots
	using PlutoUI
	using Random
	#using DelimitedFiles
end

# ╔═╡ eaf77309-f4de-44e0-887c-c80bb8836782
md"""
# Lab N: Dimesional Reduction $br Intro to Kernel PCA
### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
### Kadri Nizam 
"""
#& [Eric Ford](https://www.personal.psu.edu/ebf11)

# ╔═╡ 24e5c6b1-4444-4f5e-bcd3-5f47b62de3ac
md"""
# Kernel Principal Component Analysis
In the [!previous notebook](./pca_intro.jl) we discussed dimensionality reduction using principal component analysis. This notebook will build off from that notebook, so it is advisable to start there if you have not! 

We found that a limitation of PCA is that it is a **linear** method. Presented with an example data where the only way to separate two subsets of the data involves a non-linear function, apply the standard PCA method won't change the fact that the two subsets can't be separated with a line (or plane), even though the two subsets are obviously grouped data.  To see this, let's look at an example from a previous notebook (adapted from [Scikit-Learn](https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/datasets/_samples_generator.py#L649)'s `sample-generator.py`).
"""

# ╔═╡ 2dbe6d11-f25d-4dd7-9ce2-502ce71401b0
md"""
We fit fit a PCA model.
"""

# ╔═╡ 798ea5d8-7c4b-43b3-b227-5cb0e5056c61
md"""
**Question 1:**  What do you think the data will look like once projected on to the prinipcal component vectors?

Check when you're ready to test your prediction: $(@bind ready1 CheckBox())
"""

# ╔═╡ 28dcbd88-ed12-459c-b570-2d5be189059c
if ready1
md"""
As expected, linear PCA on this dataset fails to transform the data in a way such that the two classes could be separated by a plane.
"""
end

# ╔═╡ 8da2fa0c-1813-4908-b8fd-f17390e2971a
md"""
Fortunately, the PCA approach is quite versatile and can be modified to address this limitation. 

### Mapping to a higher dimension
A standard approach to linearize non-linear data is to transform the current space to a more favourable one. You most likely have encountered this before in the context of linear regression; an exponential function can easily be fitted with a first order polynomial after performing a log-transform on the $y$ values. 

In our particular example, a nice approach is to transform the space via the mapping ``\phi(\mathbf{x}) = \phi([\mathbf{x}_0,\,\mathbf{x}_1]^\intercal) \mapsto [\mathbf{x}_0^2,\,\mathbf{x}_1^2]^\intercal``. In this new space -- known in the industry as the _feature space_ - our data becomes linearly separable. 

See the animation below to further build your intuition.
"""

# ╔═╡ a9b16cda-6012-4d23-a0ce-99149e8c4d82
html"""<iframe id="kmsembed-1_ereyc7j4" width="600" height="400" src="https://psu.mediaspace.kaltura.com/embed/secure/iframe/entryId/1_ereyc7j4/uiConfId/41416911/st/0" class="kmsembed" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" referrerPolicy="no-referrer-when-downgrade" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="domain-transform"></iframe>"""

# ╔═╡ 5c49691c-5634-4bce-b018-d16360b3c33d
md"""
The mapping does not necessarily have to be to a space with the same dimension. In fact, you will find that it is far more common to map to a higher dimension. For instance, we can define a mapping using a [_radial basis function_](https://en.wikipedia.org/wiki/Radial_basis_function) (RBF), say, a Gaussian.
```math
  \phi(\mathbf{x}) = \phi((\mathbf{x}_0,\, \mathbf{x}_1)) ↦ \left(\mathbf{x}_0,\, \mathbf{x}_1,\, \textrm{exp}\left(-\gamma ||\,\mathbf{x}\,||^2\right)\right)
```
In general, with a nonlinear dataset of $N$ features, we can define a nonlinear map that is more suitable for performing analysis on.
```math
  \phi: \mathbb{R}^N \rightarrow \mathbb{R}^M, \quad M \geq N
```
Using the above defined RBF on the circles:
"""

# ╔═╡ dd77b6ea-eee2-458d-8457-1f0b602364ce
html"""<iframe id="kmsembed-1_2hzi7m0t" width="600" height="400" src="https://psu.mediaspace.kaltura.com/embed/secure/iframe/entryId/1_2hzi7m0t/uiConfId/41416911/st/0" class="kmsembed" allowfullscreen webkitallowfullscreen mozAllowFullScreen allow="autoplay *; fullscreen *; encrypted-media *" referrerPolicy="no-referrer-when-downgrade" sandbox="allow-forms allow-same-origin allow-scripts allow-top-navigation allow-pointer-lock allow-popups allow-modals allow-orientation-lock allow-popups-to-escape-sandbox allow-presentation allow-top-navigation-by-user-activation" frameborder="0" title="domain-transform-3d"></iframe>"""

# ╔═╡ 9818ed8e-6415-4248-b768-8931526c2097
md"""
Cool! With the mapping into higher dimension, separating the data linearly is trivial. An issue with this approach is that we had to make a decision on where to center the mapping in order to have clean separable data. Had the mapping been centered at a different location, we would get a different result. Try it for yourself!
"""

# ╔═╡ 1adbc1db-0d3f-41c6-82ba-2cd99c9debb1
@bind values PlutoUI.combine() do Child
	md"""
	x: $(Child(Slider(-1:0.1:1)))
	y: $(Child(Slider(-1:0.1:1)))
	"""
end

# ╔═╡ 47e7e6a2-ccd6-43c1-acc1-70deab503221
md"""
**Question 2:**  Can you find a location for the center of the Gaussian that results in the red and blue points being readily separable in feature space with a simple plane?  How does this center relate to the initial data set?
"""

# ╔═╡ 826082ad-06c9-462d-99b2-e7e93461e399
md"""
Notice that the clusters can end up being not linearly separable when the center of the Gaussian mapping changes.

An approach to overcoming this with real world data is via trial-and-error. Perform mappings centered at random points in our dataset; on each we perform PCA until we find a mapping that cleanly separates the data. While this would work, such approach could (read: would) be computationally infeasible/expensive with large datasets or complicated mappings. It is also not automation friendly. 

Good thing we have another trick up our mathematical sleeves. 🍿
"""

# ╔═╡ abe9756d-9de8-4228-9239-026d806b9a4d
md"""
### Inner Products and Kernels

As we have seen in the PCA lab, the heart of the algorithm lies on the fact that we can project one vector onto another via the dot product. The dot product defined on $\mathbb{R}^N$, however, is a specialized case of a more general mathematical idea -- the [inner product](https://en.wikipedia.org/wiki/Inner_product_space). There are many ways one can define what it means for one vector to be "similar" to/"projected" onto another vector in different abstract spaces.

Why are we going on this mathematical digression?
Because this allows us to bypass the need to map our data into a higher dimensional space at all! 
We instead modify the PCA algorithm to use the inner product **from the space of ``\phi``** in our current space. 

More concretely, PCA diagonalizes the covariance matrix ``C``
```math
  C = \dfrac{1}{N}\sum_{i=1}^N \mathbf{x}_i\mathbf{x}_i^\intercal
```
where ``⟨\mathbf{x}_i,\,\mathbf{x}_j⟩`` denotes the inner product between vectors ``\mathbf{x}_i`` and ``\mathbf{x}_j``. After mapping with $\phi$, the optimization problem looks identical with the only change being:
```math
    C' = \dfrac{1}{N}\sum_{i=1}^N \phi(\mathbf{x}_i)\phi(\mathbf{x}_i)^\intercal = \dfrac{1}{N}\sum_{i=1}^N K(\mathbf{x}_i,\, \mathbf{x}_j)
```
Evidently, the only thing we need in order to classify non-linear data is **the inner product equipped on ``\phi``** and not ``\phi`` itself. This is known as the _kernel trick_ and $K(\mathbf{x}_i,\, \mathbf{x}_j) = ⟨\phi(\mathbf{x}_i),\,\phi(\mathbf{x}_j)\rangle$ is known as the [_kernel function_](https://en.wikipedia.org/wiki/Positive-definite_kernel).
"""

# ╔═╡ 220ed3c0-0986-4f1a-89dd-1b97f389a482
md"""
We will define a _radial basis function_ kernel. 
"""

# ╔═╡ 86062d38-0fa1-4548-b0f5-78361f8e89d2
md"""
PCA yielded a subspace where the classes are separated well.  Data transformed into such a subspace can then be used as input to a linear classification model (e.g., Support Vector Machines or a naive Bayes classifier).
"""

# ╔═╡ 0bfa7bb5-c003-4b85-b120-529a3ab58f58
md"""
### Next Steps
Learning about Kernel PCA will help you develop intuition that will be useful for SVMs.

Or if you are already familiar with [_Support Vector Machines_](https://en.wikipedia.org/wiki/Support-vector_machine) (SVMs), then you may be able to draw parallels between Kernel PCA and SVMs in the discussion from this lab.
"""

# ╔═╡ fab26e8e-f3a2-4bea-996c-8495e0f6ea5d
md"## Setup & Helper functions"

# ╔═╡ 1d11ce68-e38d-400f-9473-247d518b4f98
function make_circles(; n_samples::Integer=100, shuffle=true, factor=0.2)

    n_samples_out = floor(Int64,n_samples // 2)
    n_samples_in = n_samples - n_samples_out

	X = zeros(2,n_samples)
    y = zeros(Int64, n_samples)

	# outer circle
    X[1, 1:n_samples_out] = cos.(LinRange(0, 2π, n_samples_out))
    X[2, 1:n_samples_out] = sin.(LinRange(0, 2π, n_samples_out))
    y[1:n_samples_out] .= 0

    # inner circle
    X[1, n_samples_out+1:end] = factor*cos.(LinRange(0, 2π, n_samples_in))
    X[2, n_samples_out+1:end] = factor*sin.(LinRange(0, 2π, n_samples_in))
    y[n_samples_out+1:end] .= 1
	
	if shuffle
       ind = randperm(n_samples)
       return X[:, ind], y[ind]
    else
       return X, y
    end
end

# ╔═╡ 9f6536e2-edf4-49e1-b086-062061d2c4a9
begin
	X, labels = make_circles(shuffle=false)
	plot(X[1, :], X[2, :], seriestype=:scatter, legend=false, marker_z=labels, markersize=5, aspect_ratio=:equal, color=:RdBu_4)
end

# ╔═╡ 62b3d6de-b3fe-4999-8815-e4f715bef692
l = fit(PCA, X)

# ╔═╡ c71898b7-b060-4466-9a59-09d349c3f85f
if ready1
	local Xₜ
	Xₜ = transform(l, X)
	plot(Xₜ[1, :], Xₜ[2, :], seriestype=:scatter, legend=false, marker_z=labels, color=:RdBu_4, markersize=5, aspect_ratio=:equal, xlabel="PC1", ylabel="PC2")
end

# ╔═╡ 141af6d8-25aa-4f01-92e7-f85760c86af0
begin
	local ϕ
	
	ϕ = (X, γ) -> exp.(-γ*norm.(eachcol(X .- values)).^2)[:]
	plot(X[1, :], X[2, :], ϕ(X, 2), marker_z=labels, seriestype=:scatter, color=:RdBu_4, legend=false, markersize=5, aspect_ratio=:equal, title="Gaussian centered at: ($(values[1]), $(values[2]))")
	xlims!(-1, 1)
	ylims!(-1, 1)
	zlims!(0, 1)
end

# ╔═╡ f037dfb8-9688-44b8-b552-fca825176f21
kl = fit(KernelPCA, X; kernel=(x,y)->exp(-2*norm(x - y)^2))

# ╔═╡ a040c788-9d20-4a3e-bdd7-45b03e764c3d
begin
	local Xₜ
	Xₜ = transform(kl, X)
	plot(Xₜ[1, :], X[2, :], seriestype=:scatter, legend=false, marker_z=labels, color=:RdBu_4, markersize=5, xlabel="PC1", ylabel="PC2")
end

# ╔═╡ 19d152a8-d592-499f-a3d9-168f8dae4ea1
function make_moon(; n_samples = 100, shuffle = true)
    n_samples_out = n_samples ÷ 2
    n_samples_in = n_samples - n_samples_out

    X = zeros(2,n_samples)
    y = zeros(n_samples)

    # outer circle
    X[1,1:n_samples_out] = cos.(LinRange(0, π, n_samples_out))
    X[2,1:n_samples_out] = sin.(LinRange(0, π, n_samples_out))
    y[1:n_samples_out] .= 0

    # inner circle
    X[1,n_samples_out+1:end] = 1 .- cos.(LinRange(0, π, n_samples_in))
    X[2,n_samples_out+1:end] = 1 .- sin.(LinRange(0, π, n_samples_in)) .- 0.5
    y[n_samples_out+1:end] .= 1

    if shuffle
       ind = randperm(n_samples)
       return X[:,ind],y[ind]
    else
       return X,y
    end
end

# ╔═╡ Cell order:
# ╟─eaf77309-f4de-44e0-887c-c80bb8836782
# ╟─24e5c6b1-4444-4f5e-bcd3-5f47b62de3ac
# ╠═9f6536e2-edf4-49e1-b086-062061d2c4a9
# ╟─2dbe6d11-f25d-4dd7-9ce2-502ce71401b0
# ╠═62b3d6de-b3fe-4999-8815-e4f715bef692
# ╟─798ea5d8-7c4b-43b3-b227-5cb0e5056c61
# ╟─c71898b7-b060-4466-9a59-09d349c3f85f
# ╟─28dcbd88-ed12-459c-b570-2d5be189059c
# ╟─8da2fa0c-1813-4908-b8fd-f17390e2971a
# ╟─a9b16cda-6012-4d23-a0ce-99149e8c4d82
# ╟─5c49691c-5634-4bce-b018-d16360b3c33d
# ╟─dd77b6ea-eee2-458d-8457-1f0b602364ce
# ╟─9818ed8e-6415-4248-b768-8931526c2097
# ╟─1adbc1db-0d3f-41c6-82ba-2cd99c9debb1
# ╟─141af6d8-25aa-4f01-92e7-f85760c86af0
# ╟─47e7e6a2-ccd6-43c1-acc1-70deab503221
# ╟─826082ad-06c9-462d-99b2-e7e93461e399
# ╟─abe9756d-9de8-4228-9239-026d806b9a4d
# ╟─220ed3c0-0986-4f1a-89dd-1b97f389a482
# ╠═f037dfb8-9688-44b8-b552-fca825176f21
# ╟─a040c788-9d20-4a3e-bdd7-45b03e764c3d
# ╟─86062d38-0fa1-4548-b0f5-78361f8e89d2
# ╟─0bfa7bb5-c003-4b85-b120-529a3ab58f58
# ╟─fab26e8e-f3a2-4bea-996c-8495e0f6ea5d
# ╟─6a3016c8-aa11-11ec-3658-0ff8413852e2
# ╟─1d11ce68-e38d-400f-9473-247d518b4f98
# ╟─19d152a8-d592-499f-a3d9-168f8dae4ea1
