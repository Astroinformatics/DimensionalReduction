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
	using PairPlots
	using PlutoUI
	using Random
	using Statistics
	
	using CSV
	import DataFrames:DataFrame, Not, names, select

	using MLDataUtils
	using LIBSVM
end;

# ╔═╡ d8f18c3f-1b9e-4bde-88f4-c407703fba27
md"""
### Overivew

In this lesson, we'll explore what happens when we use PCA to transforms the high-${z}$ quaesar dataset from the previous regression labs.  
"""

# ╔═╡ d2424607-7e22-49a9-ae6f-212364dffa44
md"### Read in data"

# ╔═╡ e917deeb-5f97-48d9-af9e-e84c5fe9e753
begin
	df = CSV.read("./data.csv", DataFrame, limit=4000, select=[:ug, :gr, :ri, :iz, :zs1, :s1s2, :label],  ntasks=1)
	col_names = names(df)
	labels_all = df[:, :label]
	df_no_label = select(df, Not(:label), copycols=false)
	data = Matrix(df_no_label)'
end;

# ╔═╡ 2dd009f9-c2a3-447e-a10f-e1ba3f40bc51
md"We can make a **corner plot**, consisting of a scatter plot of each pair of colors.  Due to the high number of points, we've added contours.  Along the diagonal, histograms show the marginal distribution of each color."

# ╔═╡ 9cd631a0-9720-4d7f-8007-2d08d7423191
corner(df_no_label)

# ╔═╡ 4aab52ce-b12b-428c-82f2-576c6d5b22a9
md"""
Below, you can pick a pair of colors to view up close and color code by the lable.  Red points are high-${z}$ quaesars and blue points are other objects.
"""

# ╔═╡ 72e1d9bb-d466-433e-93f2-ec8b24e95fc3
md"We'll split the data into two equal subsets for training and testing models."

# ╔═╡ 709dcab8-bc1c-4eb7-b262-532273fa97c4
begin
	df_train, df_test = splitobs(df, at=0.5)
	labels_train = df_train[:, :label]
	data_train = Matrix(select(df_train, Not(:label), copycols=false))'
	labels_test = df_test[:, :label]
	data_test = Matrix(select(df_test, Not(:label), copycols=false))'
end;

# ╔═╡ 3644b575-8dec-401b-8461-d31da26e2477
md"### Standard PCA"

# ╔═╡ 0d70ead1-103c-4b0a-afb9-c6c059b757da
begin
	model_vanilla = fit(PCA, data_train)
	Xₜ_train_vanilla = MultivariateStats.transform(model_vanilla, data_train)
	Xₜ_test_vanilla = MultivariateStats.transform(model_vanilla, data_test)
	Xₜ_vanilla = hcat(Xₜ_train_vanilla,Xₜ_test_vanilla)
end;

# ╔═╡ 8a169904-9dd7-4981-a371-3315f93105b6
corner(DataFrame([view(Xₜ_vanilla, i, :) for i in 1:size(Xₜ_vanilla, 1)], 
	             ["PCS " * string(i) for i in 1:size(Xₜ_vanilla,1)])  )

# ╔═╡ ec982b00-217e-45f8-9f49-078ace6798e0
let
	cfve_vanilla = cumsum(principalvars(model_vanilla)./tvar(model_vanilla))
	plt = plot(cfve_vanilla, lc=1, label=:none)
	scatter!(plt,cfve_vanilla, mc=1, label="Standard PCA", legend=:bottomright)
	xlabel!("Number Principal Compoents")
	ylabel!("Cumlative Fraction of Variance Explained")
	ylims!(0,1)
end

# ╔═╡ 3fa364ef-e0d0-4abb-a1cf-6f2cba2194a9
md"### Kernel PCA"

# ╔═╡ 0d1b5789-7e32-4c5c-99c5-a8f4037c9873
md"γ (Exponent for radial basis function): $(@bind kpca_gamma confirm(NumberField(0.05:0.05:2, default=0.5)))"

# ╔═╡ 1cdab078-297d-498a-8d31-aa3682d57151
md"## Fit SVM classifers"

# ╔═╡ 7ff72df3-eb8d-401a-8d22-70c2180c3562
md"First, we'll fit an SVM classifier using the raw data and use it to predict the labels for our dataset. 
(Note that by default, the 'SVC' function uses a radial basis kernel.  Since we want to compare the performance of the SVM using different inputs, we'll specify that the SVM should use a linear kernel.)"

# ╔═╡ 98bcdb6b-cb0f-487d-8d45-d1d401e38acb
begin 
	svm_raw = fit!(SVC(kernel=Kernel.Linear), data_train', labels_train)  
	ŷ_svm_raw = LIBSVM.predict(svm_raw, data_test')   # Predict the labels for data
end;

# ╔═╡ 3743a33f-f9da-4e59-9960-8c2220abfdfc
md"To evaluate the classifier, we'll evaluate the model based on calculating the fraction of actual high-${z}$ quaesars missed (false negative rate) and the fraction of other objects mistakenly labeled as high-${z}$ quaesars (false positive rate)."

# ╔═╡ f172c447-91ac-4e43-96e1-75d9689c1c0a
begin
	mask_test = labels_test.==1               # Find actual quaesars
	(; false_negative_fraction = sum(ŷ_svm_raw[mask_test] .!= labels_test[mask_test])/sum(mask_test),
	   false_positive_fraction = sum(ŷ_svm_raw[.!mask_test] .!= labels_test[.!mask_test])/sum(.!mask_test) )
end

# ╔═╡ 8d39ca66-a14e-4852-ba70-eda1d2eb2879
md"""Next, we'll fit an SVM classifier to the data after being transformed by PCA.

**Question:** Do you expect the performance of this classifier will differ from that above?  
If so, do you expect it to be better or worse?  If not, why not?
"""

# ╔═╡ c6a841db-ece0-4e09-826d-be3347e4e53d
md"Ready to see what happens? $(@bind ready_svm_pca CheckBox())"

# ╔═╡ 41f8d92a-d184-4387-9027-5bd44786792d
if ready_svm_pca
	svm_pca = fit!(SVC(kernel=Kernel.Linear), Xₜ_train_vanilla', labels_train)
	ŷ_svm_pca = LIBSVM.predict(svm_pca, Xₜ_test_vanilla')
	(; false_negative_fraction = sum(ŷ_svm_pca[mask_test] .!= labels_test[mask_test])/sum(mask_test),
	   false_positive_fraction =  sum(ŷ_svm_pca[.!mask_test] .!= labels_test[.!mask_test])/sum(.!mask_test) )
end

# ╔═╡ a4c41470-bc8a-4141-a934-af90b9b29c45
md"""Now, we'll fit an SVM classifier using the data transformed into a new manifold using kernel PCA.  

**Question:** Do you expect the performance of this classifier will differ from those above?  If so, do you expect it to be better or worse?  If not, why not?

Ready to see what happens? $(@bind ready_svm_kpca CheckBox())
"""

# ╔═╡ e9f8d3f7-ae46-4793-8eac-fd8ad711e4f7
md"""
## Setup & Helper code
"""

# ╔═╡ 0118ef60-7422-493f-8fee-ca90a594c967
begin
	kernel_linear = (x,y)->x'y
	kpca_c = 1.0
	kpca_d = 2
	kernel_polynomial = (x,y)->(x'y+kpca_c)^kpca_d
	kernel_radial = (x,y)->exp(-kpca_gamma.*norm(x-y)^2)
	#kernel_to_use = kernel_radial
end

# ╔═╡ 14f8cc90-2610-47d0-bc4e-702a40fd20d7
begin
	model_kernel = fit(KernelPCA, data_train; kernel=kernel_radial,  inverse=true) #, maxoutdim=3)
	Xₜ_train_kernel = MultivariateStats.transform(model_kernel, data_train)
	Xₜ_test_kernel = MultivariateStats.transform(model_kernel, data_test)
	Xₜ_kernel = hcat(Xₜ_train_kernel, Xₜ_test_kernel)
end;

# ╔═╡ ba228583-b2af-47a0-bbc5-01235ea52fa5
corner(DataFrame([view(Xₜ_kernel, i, :) for i in 1:size(Xₜ_kernel, 1)], 
	             ["KPCS " * string(i) for i in 1:size(Xₜ_kernel,1)])  )

# ╔═╡ 75ef65ba-8bce-4eac-b6b9-cbf91da8d4f1
let
	cfve_vanilla = cumsum(principalvars(model_vanilla)./tvar(model_vanilla))
	plt = plot(cfve_vanilla, lc=1, label=:none)
	scatter!(plt,cfve_vanilla, mc=1, label="Standard PCA", legend=:bottomright)
	cfve_kernel = cumsum(model_kernel.:λ)/sum(model_kernel.:λ)
	plot!(plt,cfve_kernel, lc=2, label=:none)
	scatter!(plt,cfve_kernel, mc=2, label="Kernel PCA", legend=:bottomright)
	xlabel!("Number Principal Compoents")
	ylabel!("Fraction of variance explained")  # TODO:  Check if this is the right label for kernel PCA?
	ylims!(0,1)
end

# ╔═╡ 1af16c3f-739c-4d2e-95c4-d962491d09c6
if ready_svm_kpca
	svm_kpca = fit!(SVC(kernel=Kernel.Linear), Xₜ_train_kernel', labels_train)
	ŷ_svm_kpca = LIBSVM.predict(svm_kpca, Xₜ_test_kernel')
	(; false_negative_fraction = sum(ŷ_svm_kpca[mask_test] .!= labels_test[mask_test])/sum(mask_test),
	   false_positive_fraction =  sum(ŷ_svm_kpca[.!mask_test] .!= labels_test[.!mask_test])/sum(.!mask_test) )
end

# ╔═╡ 30a05385-d378-4299-88bc-402967d67187
nbsp = html"&nbsp";

# ╔═╡ 5d2bd890-8edd-430f-bc63-6776181af1ae
@bind values_color_cols PlutoUI.combine() do Child
	md"""
	Color for x:  $(Child(@bind plt_col_x Select(Pair.(1:length(col_names)-1,col_names[1:end-1]), default=1)))
	$nbsp $nbsp $nbsp
	Color for y: $(Child(@bind plt_col_y Select(Pair.(1:length(col_names)-1,col_names[1:end-1]), default=2)))
	"""
end

# ╔═╡ 1bfa8472-523f-4fc9-a1de-b9ecf96da1bb
let
	plot(data[plt_col_x, :], data[plt_col_y, :]; c=2 .- labels_all, ms=2.5, ma=0.5, seriestype=:scatter, legend=false, xlabel=col_names[plt_col_x], ylabel=col_names[plt_col_y])
end

# ╔═╡ 17c66f11-fcaf-43f5-a1d9-fe447c2b8274
@bind values_pcs_vanilla PlutoUI.combine() do Child
	md"""
	PC Score for x:  $(Child(@bind plt_pc_x Select(1:size(Xₜ_vanilla,1), default=1)))
	$nbsp $nbsp $nbsp
	PC Score for y: $(Child(@bind plt_pc_y Select(1:size(Xₜ_vanilla,1), default=2)))
	"""
end

# ╔═╡ d4b002d7-97d4-43de-b14a-5de4ab23f315
let
	plot(Xₜ_vanilla[plt_pc_x, :], Xₜ_vanilla[plt_pc_y, :]; c=2 .-labels_all, ms=2.5, ma=0.5, seriestype=:scatter, legend=false, xlabel="Standard PCA Score $plt_pc_x", ylabel="Standard PCA Score $plt_pc_y")
end

# ╔═╡ 741f93ea-4023-439f-a35a-77c172e84b92
@bind values_pcs_kernel PlutoUI.combine() do Child
	md"""
	Kernal PCA Score for x:  $(Child(@bind plt_kpc_x Select(1:size(Xₜ_kernel,1), default=1)))
	$nbsp $nbsp $nbsp
	Kernel PCA Score for y: $(Child(@bind plt_kpc_y Select(1:size(Xₜ_kernel,1), default=2)))
	"""
end

# ╔═╡ b64a6fa7-4781-439d-8edf-e7f01ae7f014
begin
	plot(Xₜ_kernel[plt_kpc_x, :], Xₜ_kernel[plt_kpc_y, :]; ms=2.5, ma=0.5, c=2 .- labels_all, seriestype=:scatter, legend=false, xlabel="Kernel PCA Score $plt_kpc_x", ylabel="Kernel PCA Score $plt_kpc_y")
end

# ╔═╡ f879a7de-5344-4329-9fe3-6f67e35fab58
br = html"<br>";

# ╔═╡ a53948d8-9deb-4802-9ab9-151e3f938070
md"""
# Lab 8: Dimesional Reduction $br Applying PCA to Astronomical Data
## [Penn State Astroinformatics Summer School 2022](https://sites.psu.edu/astrostatistics/astroinfo-su22-program/)
## Kadri Nizam & [Eric Ford](https://www.personal.psu.edu/ebf11)
"""

# ╔═╡ 8bb4c747-2637-4cdd-b5af-336be9c23112
md"## Old code"

# ╔═╡ d6da64d3-da7b-43a1-a60e-aabaf55b4169
begin 
	#model_vanilla_n = map(n->fit(PCA, data; maxoutdim=n), 1:size(data,1)-1)
	model_vanilla_n = vcat(model_vanilla_n, model_vanilla)
end

# ╔═╡ fa296e92-2ea1-48bc-a94f-6c015b227883
begin 
	#model_kernel_n = map(n->fit(KernelPCA, data; kernel=kernel_radial,  inverse=true, maxoutdim=n), 1:size(data,1)-1)
	model_kernel_n = vcat(model_kernel_n, model_kernel)
end

# ╔═╡ 6c5aadb0-8b71-4df6-95d6-47a62e29163d
[sum((reconstruct(model_kernel_n[n],predict(model_kernel_n[n])).-data).^2) for n in 1:6]

# ╔═╡ fa5985d5-2e8f-42e4-a577-8e0e70836e95
[sum((reconstruct(model_vanilla_n[n],predict(model_vanilla_n[n],data)).-data).^2) for n in 1:6]

# ╔═╡ 84c83c10-491f-475f-9022-3ad538adda94
begin
	frac_var_explained_kernel_n = zeros(length(model_kernel_n))
	for n in 1:length(model_kernel_n)
		m = model_kernel_n[n] 
		Yte = MultivariateStats.transform(m, data)
		Xr = reconstruct(m, Yte)
		frac_var_explained_kernel_n[n] = sum((data.-Xr).^2)./sum(data.^2)
	end
	frac_var_explained_kernel_n
end

# ╔═╡ 3adb6787-1cd7-433d-9689-af464cc9cc9c
begin
	frac_var_explained_pca_n = zeros(length(model_vanilla))
	for n in 1:length(model_kernel_n)
		m = model_kernel_n[n] 
		Yte = MultivariateStats.transform(m, data)
		Xr = reconstruct(m, Yte)
		frac_var_explained_kernel_n[n] = sum((data.-Xr).^2)./sum(data.^2)
	end
	frac_var_explained_kernel_n
end

# ╔═╡ Cell order:
# ╟─a53948d8-9deb-4802-9ab9-151e3f938070
# ╟─d8f18c3f-1b9e-4bde-88f4-c407703fba27
# ╟─d2424607-7e22-49a9-ae6f-212364dffa44
# ╠═e917deeb-5f97-48d9-af9e-e84c5fe9e753
# ╟─2dd009f9-c2a3-447e-a10f-e1ba3f40bc51
# ╟─9cd631a0-9720-4d7f-8007-2d08d7423191
# ╟─4aab52ce-b12b-428c-82f2-576c6d5b22a9
# ╟─5d2bd890-8edd-430f-bc63-6776181af1ae
# ╟─1bfa8472-523f-4fc9-a1de-b9ecf96da1bb
# ╟─72e1d9bb-d466-433e-93f2-ec8b24e95fc3
# ╠═709dcab8-bc1c-4eb7-b262-532273fa97c4
# ╟─3644b575-8dec-401b-8461-d31da26e2477
# ╠═0d70ead1-103c-4b0a-afb9-c6c059b757da
# ╟─8a169904-9dd7-4981-a371-3315f93105b6
# ╟─17c66f11-fcaf-43f5-a1d9-fe447c2b8274
# ╟─d4b002d7-97d4-43de-b14a-5de4ab23f315
# ╟─ec982b00-217e-45f8-9f49-078ace6798e0
# ╟─3fa364ef-e0d0-4abb-a1cf-6f2cba2194a9
# ╠═14f8cc90-2610-47d0-bc4e-702a40fd20d7
# ╟─ba228583-b2af-47a0-bbc5-01235ea52fa5
# ╟─0d1b5789-7e32-4c5c-99c5-a8f4037c9873
# ╟─741f93ea-4023-439f-a35a-77c172e84b92
# ╟─b64a6fa7-4781-439d-8edf-e7f01ae7f014
# ╠═75ef65ba-8bce-4eac-b6b9-cbf91da8d4f1
# ╟─1cdab078-297d-498a-8d31-aa3682d57151
# ╟─7ff72df3-eb8d-401a-8d22-70c2180c3562
# ╠═98bcdb6b-cb0f-487d-8d45-d1d401e38acb
# ╟─3743a33f-f9da-4e59-9960-8c2220abfdfc
# ╟─f172c447-91ac-4e43-96e1-75d9689c1c0a
# ╟─8d39ca66-a14e-4852-ba70-eda1d2eb2879
# ╟─c6a841db-ece0-4e09-826d-be3347e4e53d
# ╟─41f8d92a-d184-4387-9027-5bd44786792d
# ╟─a4c41470-bc8a-4141-a934-af90b9b29c45
# ╟─1af16c3f-739c-4d2e-95c4-d962491d09c6
# ╟─e9f8d3f7-ae46-4793-8eac-fd8ad711e4f7
# ╟─69cd367a-959e-11ec-1d11-dbb242dc1861
# ╟─0118ef60-7422-493f-8fee-ca90a594c967
# ╠═30a05385-d378-4299-88bc-402967d67187
# ╠═f879a7de-5344-4329-9fe3-6f67e35fab58
# ╟─8bb4c747-2637-4cdd-b5af-336be9c23112
# ╠═d6da64d3-da7b-43a1-a60e-aabaf55b4169
# ╠═fa296e92-2ea1-48bc-a94f-6c015b227883
# ╠═6c5aadb0-8b71-4df6-95d6-47a62e29163d
# ╠═fa5985d5-2e8f-42e4-a577-8e0e70836e95
# ╠═3adb6787-1cd7-433d-9689-af464cc9cc9c
# ╠═84c83c10-491f-475f-9022-3ad538adda94
