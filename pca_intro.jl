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

# ╔═╡ 69cd367a-959e-11ec-1d11-dbb242dc1861
begin
	using LinearAlgebra
	using Markdown
	using MultivariateStats
	using Plots, LaTeXStrings
	using PlutoUI
	using Random
	using Statistics
	
end;

# ╔═╡ d8f18c3f-1b9e-4bde-88f4-c407703fba27
md"""
In previous lessons, we explored machine learning approaches to regression and classification.  Now, we'll steer our attention to using machine learning algorithms for _dimensional reduction_. Dimensional reduction allows us to tranform high dimensional data into lower dimensional manifolds subject to a some constraints. 

There are several algorithms for dimensional reduction that impose different constraints.  In this notebook, we'll start with one of the most common, [**Principal Component Analysis (PCA)**](https://en.wikipedia.org/wiki/Principal_component_analysis).  
In future lessons, we'll explore [Kernel PCA](https://en.wikipedia.org/wiki/Kernel_principal_component_analysis), [autoencoders](https://en.wikipedia.org/wiki/Autoencoder) and T-Stochastic Neighbour Embedding [T-SNE, "tee-snee"](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding).  
"""
# TODO? Add Uniform Manifold Approximation (UMAP). # maybe?

# ╔═╡ d7168a56-8133-4c6f-9b16-c69014991669
md"""
## Principal Component Analysis (PCA)
Principal component analysis is a computationally inexpensive and flexible unsupervised method for reducing dimensionality in a dataset. Consider a dataset with a linear relationship between the variables $x_1$ and $x_2$. 
"""

# ╔═╡ 910e68dc-791a-4b91-b557-32d963f20311
md"As usual, is good to start by visualizing our dataset."

# ╔═╡ ee8976d2-ccbd-43bc-854a-c2772892b2c4
md"""
In the [Linear Regression](#) lab, we sought to find a way to train a model to make _prediction_ trying to match the true values provided in a training dataset. The goal of principal component analysis is different: to learn about the _relationship_ between $x_1$ and $x_2$ from the data itself. 
"""

# ╔═╡ 768d7d6f-300c-4da5-83cd-aa4e2854f836
md"""
### Applying PCA
A great way to develop intuition for what PCA does is run it on some simulated datasets and see what it does. For now, we'll utilize a Julia package,  [MultivariateStats.jl](https://github.com/JuliaStats/MultivariateStats.jl), so as to avoid being districated by implementing the PCA algorithm.   The package provides the `fit` function that takes two arguments, an algorithm and a dataset.  We'll pass `PCA` to specify that's the algorithm we want it to use, and rearrange our data into a matrix with variables in the rows and observations in the columns.

Since our dataset has $N$ datapoints, each with two variables $x_1$ and $x_2$, we will combine them into a single ``N\times 2`` matrix and take its transpose to produce a ``2 \times N`` matrix called $X$.
"""

# ╔═╡ 840e003f-df3d-4fde-94b5-c819ebec9be9
md"""
Now we can call the `fit` function with `PCA` and `X` as the argument. Further documentation on syntax can be found [here](https://multivariatestatsjl.readthedocs.io/en/stable/pca.html).
"""

# ╔═╡ 79246181-26d4-436a-aad3-0709c965833a
md"""
After fitting the data, all of the information is tucked away inside the `model` variable. There are several functions that you can call on `model`.  Some of the pertinent ones are: 
- `projection`: get the principal components
- `principalvars`: get principal variances
- `tprincipalvar`: get total variance
- `tvar`: get the total observational variance

Let's examine those really quickly starting with `projection`.
"""

# ╔═╡ cd1dd2ea-33b8-452d-a9db-ae92a6ce4073
md"""
We find a square matrix with dimension ``N = 2``. This is expected since our input data is 2-dimensional. The columns of the matrix are made up of vectors that are known as the *principal components*. These vectors point in the directions that are orthonormal.  The first principal componet points in the direction of maximal variance in our dataset.  The second principal componet points in the direction of second largest variance (in this case it's also the least variance, since we have only 2 dimensions).
"""

# ╔═╡ 20cef775-13e0-419d-958e-66b0c185ba79
md"""
!!! tip "Terminology: Orthonormal Vectors"
    Consider two vectors ``\mathbf{u},\,\mathbf{v} \in \mathbb{R}^N``. _Orthonormal_ describes the property of the vectors:
     - Each vector in the set is unit length i.e. normalized:
       
       $\mathbf{u} \cdot \mathbf{u} = 1 = \mathbf{v}\cdot\mathbf{v}$
     - Each vector in the set is orthogonal to each other. This also implies that the inverse of a vector is its transpose

       $\mathbf{u} \cdot \mathbf{v} = 0$

       $\mathbf{u}^{-1} \mathbf{u} = \mathbf{u}^\intercal \mathbf{u} = \mathbb{1}$
    where we have used ``\mathbb{1}`` to mean the identity matrix with dimension ``N``.
"""

# ╔═╡ 685e84f4-862b-4ab2-a8e1-94a97d0bef86
md"## Visualizing the Principal Components"

# ╔═╡ 5fcedebd-9df4-4e38-8623-2e344a9a93a7
md"Let plot the principal component vectors on top of our dataset."


# ╔═╡ 4c9c52e4-28cf-4518-9db7-11a06cd120d4
md"""
Hold on a second. The principal components don't look orthogonal at all! 
What could explain this?  
Sometimes the vector might not look orthogonal geometrically due to how it's ploted, even if they are.  Try checking the boxes below to see how each affects the plot above.
"""

# ╔═╡ 17171a41-ea52-4c57-a514-2e1986ea2b0d
md"### Checking orthogonality numerically"

# ╔═╡ 28b7f757-5865-4d32-b65c-8942dc83aed2
md"""
We can check for orthogonality numerically by left multiplying the matrix with its transpose.  If they orthogonal, then we expect to see an identity matrix (or something very close due to floating point arithmetic).  This test is particularly useful once you start working in higher dimensions.
"""

# ╔═╡ b156b404-b446-4663-886c-caff23ce19b7
md"## Visualizing Transformed Data"

# ╔═╡ c8a17a03-963e-4894-bf5a-73cf8bf4df57
md"""
We now have the principal components that point in the direction of maximal variance in our dataset. The power of PCA lies in the fact that we can transform our dataset such that these directions become our new axes. This is done by projecting all the points in our dataset onto the principal components.
"""

# ╔═╡ e75a69aa-8910-404c-8343-c8989bf5ca53
md"""
Explore how the principal component vectors and scores change as you varry the true slope and the scatter about the linear relationship using the boxes below.  
"""

# ╔═╡ 2b8f50d1-a28d-4c35-807d-5bf8cd5be4d3
md"""
True slope:  $(@bind true_slope NumberField(-3:0.2:3, default=1))
True scatter:  $(@bind true_sigma NumberField(0.01:0.01:1, default=0.2))
Number of datapoints:  $(@bind N NumberField(10:10:1000, default=100))
"""


# ╔═╡ d86b2d9b-805c-4740-8cdb-95d220f30170
begin	# generate simulated dataset
	Random.seed!(120)  # For reproducibility
	# N, true_slope and true_sigma are set by input widgets below
	x1 = rand(N)
	x1 = x1 .- mean(x1)
	
	x2 = true_slope.*x1 .+ true_sigma.*randn(N)
	x2 = x2 .- mean(x2)
end;

# ╔═╡ 1516cdc8-89fb-4398-96f0-aa6ef6114a02
begin
	xlims = extrema(x1)
	ylims = extrema(x2)
	sharedlims = (min(first(xlims),first(ylims)), max(last(xlims), last(ylims)))
	plot(x1, x2, seriestype=:scatter, legend=false, xlims=sharedlims, ylims=sharedlims)
	xlabel!(L"x_1")
	ylabel!(L"x_2")
end

# ╔═╡ 07014ad5-eb58-447f-b902-07e1d81dff2f
begin
	X = [x1 x2]'   # ' specifies matrix transpose
	size(X)
end

# ╔═╡ 6b490754-dd14-4cca-b77d-16d7cb0ab858
 model = fit(PCA, X)

# ╔═╡ 0255bb8f-b6cf-4def-984a-3a6ff18dfb36
principal_components = projection(model)

# ╔═╡ bcf5ef4a-e3db-4d61-a3ea-066913e7fd02
principal_components' * principal_components

# ╔═╡ 2d3099a5-7d19-4e42-b934-ef75f632fd97
X_transformed = transform(model, X);

# ╔═╡ 4a56f7d6-ce5a-4206-8de9-4ef4ab94c0c7
md"""
**Question:** What happens if you set the true scatter to be smaller?  very small?

!!! hint "Hint"
    The `fit` function only returns enough principal component to explain a specified fraction of the total variance (by default 99%).  When the true scatter is sufficiently small, the number of principal component needed to explain nearly all of the variation decreases.

Before proceeding, reset the true slope to 1 and true scatter to 0.2.  
"""


# ╔═╡ b0531872-cd40-4075-b2e9-7fbbad7b8509
md"""
**Question:** What will happen if you reset the true scatter to 0.2, but make the slope larger?  very small?

Before proceeding, reset the true slope to 1 and true scatter to 0.2.  
"""


# ╔═╡ 7f89fa86-2b6e-4e6a-a03f-40f0019c951b
md"## Principal variances"

# ╔═╡ 01e9800a-c1ff-44fd-99a1-0667e0a55317
md"""
We can also examine the _principal variance_ for each of the principal components. The principal components point in the directions of maximal variance, and the principal variance tells us how much variation is observed in that direction.
"""

# ╔═╡ ffe18a3f-4c97-496b-9b92-d6e59ab1bffd
principalvars(model)

# ╔═╡ 1ef8c7d6-6e08-42a5-ac38-e2de50b5337d
md"""
It make sense that the first component has the most variation, since it is pointing along the linear curve.  More useful is to rewrite these values as a fraction of the total variance.
"""

# ╔═╡ b2bff415-d191-4457-9aa9-8b6fc9650d88
principalvars(model)./tprincipalvar(model)

# ╔═╡ a9824f50-e47b-453d-8213-5857315fe8d0
per_var = (length(principalvars(model))==2) ? round.(principalvars(model)./tprincipalvar(model).*100, digits=1) : (100,0);

# ╔═╡ 91d8448d-46fd-40da-b6f3-5a24d8bc6611
md"""
We can now see that the first principal component accounts for about $(per_var[1])% of variations in our dataset while the second principal component only accounts for about $(per_var[2])%. This means, if we reconstructed our data using only the first component, then the resulting dataset would have $(per_var[1])% of the variance of the original dataset. 

**Question:**  How do expect the principal variances will change as you decrease the true scatter or increase the slope? 

"""

# ╔═╡ e8a5b5e3-9645-4f78-9bca-897493bc82f0
md"""
### Changing the criteria for picking number of principal components
In fact, the `fit` function takes an option argument `pratio` that allows us to specify that we want it to return the smallest number of principal components that account for a given fraction of the total variance in your dataset. 

Imagine if you had a dataset with 1000 variables per observation (e.g., a spectra).  It's likely that you can account for 99% of variations in your dataset with a number of principal components much less than the number of observations (e.g., number of pixels in each spectra).  By reducing the dimension of your dataset, it becomes practical to work with datasets containing more objects and/or to apply more computationally expensive algorithms to an existing dataset!
"""

# ╔═╡ 3b980f12-cb13-4ffa-88fc-b91ba6ee850d
md"""
We can visualize this reconstruction by setting a smaller value of `pratio`:
$(@bind pratio_for_reconstruction NumberField(0.1:0.01:1.0, default=0.9))
"""

# ╔═╡ 3971d349-c254-4f16-84be-c4fccec9290c
reduced_model =  fit(PCA, X; pratio = pratio_for_reconstruction);

# ╔═╡ cddb22b3-2d7b-4ade-94d5-8232e0a383a3
let
	X_transformed = transform(reduced_model, X)
	if size(X_transformed,1) == 2
		plot(X_transformed[1,:], X_transformed[2,:], seriestype=:scatter, legend=false, ylims=(-1,1))
	ylabel!("Score PC 2")
	else
		plot(X_transformed[1,:], [0], seriestype=:scatter, legend=false, ylims=(-1,1))
	end
	xlabel!("Score PC 1")
end

# ╔═╡ 890f3052-7848-4b02-97f6-a9948c6d9bdf
begin
	scatter(x1, x2, markeralpha=0.5,label="Original data", legend=:topleft)
	xx = reconstruct(reduced_model, transform(reduced_model, X))
	scatter!(xx[1, :], xx[2, :], markeralpha=0.5, label="Reconstructed data")
	xlabel!(L"x_1")
	ylabel!(L"x_2")
	title!("Reconstructing Data")
end

# ╔═╡ f1ad53ca-363f-4007-86cb-55959c37356b
md"""
**Question:**  How does the the reconstructed data compare to the original data, if you set pratio to be near 1?

**Question:**  How does the the reconstructed data compare to the original data, if you reduce pratio?

**Question:** How would your answer to the previous question change if the data being analyzed had more than 2 dimensions?
"""

# ╔═╡ af42e120-aaa4-427c-99b3-dd7218bd19a4
md"""
## Next steps
- Can you think of how dimensional reduction algorithm like PCA might be applicable to your research interests?
- What complications might arise with vanilla PCA?  
- If you're interested in further building your intuition for PCA, experiment with the excellent applet [setosa.io](https://setosa.io/ev/principal-component-analysis/).
"""

# ╔═╡ e9f8d3f7-ae46-4793-8eac-fd8ad711e4f7
md"""
## Setup & Helper code
"""

# ╔═╡ 7dc4f9f7-ee82-4af1-81cf-41201fdbdfd1
TableOfContents()

# ╔═╡ b8b39dc2-d44f-45f5-9094-ac8299d9ccec
function plot_arrow!(plt, v, offset = [0, 0])
	quiver!(plt,[offset[1]], [offset[2]], quiver = ([v[1]], [v[2]]), linewidth=3)
end

# ╔═╡ 30a05385-d378-4299-88bc-402967d67187
nbsp = html"&nbsp";

# ╔═╡ 18218e32-3dab-49ba-b1d0-fd2192367d3b
md"""
Use common axis limits: $(@bind use_common_axis_limits CheckBox(default=false)) 
$nbsp $nbsp $nbsp
Use a square aspect ratio $(@bind use_square_aspect_ratio CheckBox(default=false)) 
"""


# ╔═╡ 55829368-871a-4c4c-9fb4-0582832363a4
function plot_sim_data_and_principal_componets() 
	plt = plot(X[1, :], X[2, :], seriestype=:scatter, legend=false, markeralpha=0.3, 
		xlims=use_common_axis_limits ? sharedlims : xlims, ylims=use_common_axis_limits ? sharedlims : ylims, 
		size=use_square_aspect_ratio ? (800,800) : (800, 600))
	xlabel!(plt,L"x_1")
	ylabel!(plt,L"x_2")
	PC = model.proj
	s = sqrt.(principalvars(model))
	μ = mean(X , dims = 2)
	
	plot_arrow!(plt, s[1] * PC[:, 1])
	if length(s) >=2 
		plot_arrow!(plt, s[2] * PC[:, 2])
	end
	plt
end

# ╔═╡ 089430af-8118-4cc9-bf3c-0b1652c116fa
plot_sim_data_and_principal_componets()

# ╔═╡ 86be09b7-600f-4049-b18d-9a7aa0fc554e
let
	plt1 = plot_sim_data_and_principal_componets()
	plot!(plt1,size=(800,400))
	plt2 = plot(xlabel="Score PC 1", ylabel="Score PC 2", ylims=(-2*true_sigma,2*true_sigma), size=(800,400))
	if size(X_transformed,1) == 2
		plot!(plt2,X_transformed[1, :], X_transformed[2, :], seriestype=:scatter, legend=false)
	else
		plot!(plt2,X_transformed[1, :], zeros(N), seriestype=:scatter, legend=false)
	end
	plot(plt1,plt2, layout=@layout [a b])
end

# ╔═╡ d8964ca6-4e26-4aee-ad94-1412428158f1
br = html"<br />";

# ╔═╡ a53948d8-9deb-4802-9ab9-151e3f938070
md"""
# Lab 6: Dimesional Reduction: $br Intro to PCA
#### [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
#### Kadri Nizam & [Eric Ford](https://www.personal.psu.edu/ebf11)
"""

# ╔═╡ Cell order:
# ╟─a53948d8-9deb-4802-9ab9-151e3f938070
# ╟─d8f18c3f-1b9e-4bde-88f4-c407703fba27
# ╟─d7168a56-8133-4c6f-9b16-c69014991669
# ╟─d86b2d9b-805c-4740-8cdb-95d220f30170
# ╟─910e68dc-791a-4b91-b557-32d963f20311
# ╟─1516cdc8-89fb-4398-96f0-aa6ef6114a02
# ╟─ee8976d2-ccbd-43bc-854a-c2772892b2c4
# ╟─768d7d6f-300c-4da5-83cd-aa4e2854f836
# ╠═07014ad5-eb58-447f-b902-07e1d81dff2f
# ╟─840e003f-df3d-4fde-94b5-c819ebec9be9
# ╠═6b490754-dd14-4cca-b77d-16d7cb0ab858
# ╟─79246181-26d4-436a-aad3-0709c965833a
# ╠═0255bb8f-b6cf-4def-984a-3a6ff18dfb36
# ╟─cd1dd2ea-33b8-452d-a9db-ae92a6ce4073
# ╟─20cef775-13e0-419d-958e-66b0c185ba79
# ╟─685e84f4-862b-4ab2-a8e1-94a97d0bef86
# ╟─5fcedebd-9df4-4e38-8623-2e344a9a93a7
# ╟─089430af-8118-4cc9-bf3c-0b1652c116fa
# ╟─4c9c52e4-28cf-4518-9db7-11a06cd120d4
# ╟─18218e32-3dab-49ba-b1d0-fd2192367d3b
# ╟─17171a41-ea52-4c57-a514-2e1986ea2b0d
# ╟─28b7f757-5865-4d32-b65c-8942dc83aed2
# ╠═bcf5ef4a-e3db-4d61-a3ea-066913e7fd02
# ╟─b156b404-b446-4663-886c-caff23ce19b7
# ╟─c8a17a03-963e-4894-bf5a-73cf8bf4df57
# ╠═2d3099a5-7d19-4e42-b934-ef75f632fd97
# ╟─86be09b7-600f-4049-b18d-9a7aa0fc554e
# ╟─e75a69aa-8910-404c-8343-c8989bf5ca53
# ╟─2b8f50d1-a28d-4c35-807d-5bf8cd5be4d3
# ╟─4a56f7d6-ce5a-4206-8de9-4ef4ab94c0c7
# ╟─b0531872-cd40-4075-b2e9-7fbbad7b8509
# ╟─7f89fa86-2b6e-4e6a-a03f-40f0019c951b
# ╟─01e9800a-c1ff-44fd-99a1-0667e0a55317
# ╠═ffe18a3f-4c97-496b-9b92-d6e59ab1bffd
# ╟─1ef8c7d6-6e08-42a5-ac38-e2de50b5337d
# ╠═b2bff415-d191-4457-9aa9-8b6fc9650d88
# ╟─a9824f50-e47b-453d-8213-5857315fe8d0
# ╟─91d8448d-46fd-40da-b6f3-5a24d8bc6611
# ╟─e8a5b5e3-9645-4f78-9bca-897493bc82f0
# ╟─3b980f12-cb13-4ffa-88fc-b91ba6ee850d
# ╠═3971d349-c254-4f16-84be-c4fccec9290c
# ╟─cddb22b3-2d7b-4ade-94d5-8232e0a383a3
# ╟─890f3052-7848-4b02-97f6-a9948c6d9bdf
# ╟─f1ad53ca-363f-4007-86cb-55959c37356b
# ╟─af42e120-aaa4-427c-99b3-dd7218bd19a4
# ╟─e9f8d3f7-ae46-4793-8eac-fd8ad711e4f7
# ╠═69cd367a-959e-11ec-1d11-dbb242dc1861
# ╠═7dc4f9f7-ee82-4af1-81cf-41201fdbdfd1
# ╟─55829368-871a-4c4c-9fb4-0582832363a4
# ╟─b8b39dc2-d44f-45f5-9094-ac8299d9ccec
# ╟─30a05385-d378-4299-88bc-402967d67187
# ╟─d8964ca6-4e26-4aee-ad94-1412428158f1
