# Dimensional Reduction Labs

### led by [Prof. Ashley Villar](http://ashleyvillar.com/menu/about/)
### Astroinformatics Summer School 2022 
### Organized by [Penn State Center for Astrostatistics](https://sites.psu.edu/astrostatistics/)
---

This repository contains several computational notebooks: 
- pca_intro.jl (Pluto notebook):  Provides an introduction to Principal Components Analysis in two dimensions.
- kernel_pca_intro.jl (Pluto notebook):  Demonstrates how using a kernel allows PCA to separate clusters of points when standard PCA can not.
- pca_apply.jl (Pluto notebook):  Demonstrates combining PCA with SVM to classify high-redshift quasars.
- application_to_galaxy_images.ipynb (Jupyter notebook): Demonstrates combining PCA, radial basis function kernel, and SVM to classify galaxy images. 

Most students will want to proceed in the order above.  Students already confident in their understanding of PCA and Kernel PCA are welcome to jump to the applicaitons.

Files ending in .jl are Pluto notebooks written in Julia and files ending in .ipynb are Jupyter notebooks written in Python.
Labs do not assume familiarity with either language.  While it can be useful to "read" selected portions of the code, the lab tutorials aim to emphasize understanding how algorithms work, while minimizing need to pay attention to a language's syntax.

## Running Labs
Instructions will be provided for students to run labs on AWS severs during the summer school.  Below are instruction for running them outside of the summer school.

### Running Pluto notebooks on your local computer
Summer School participants will be provided instructions for accessing a Pluto server.  Others may install Julia and Pluto on their local computer with the following steps:
1.  Download and install current version of Julia from [julialang.org](https://julialang.org/downloads/).
2.  Run julia
3.  From the Julia REPL (command line), type
```julia
julia> using Pkg
julia> Pkg.add("Pluto")
```
(Steps 1 & 3 only need to be done once per computer.)

4.  Start Pluto
```julia
julia> using Pluto
julia> Pluto.run()
```
5.  Open the Pluto notebook for your lab

### Running Jupter/Python notebooks 
Summer School participants will be provided instructions for accessing JupyterLab server.  
Others may install Python 3 and Jupyter (or JupyterLab) on their local computer or use [Google Colab](https://colab.research.google.com/) to open the Jupyter notebooks.

## Contributing
We welcome people filing issues and/or pull requests to improve these labs for future summer schools.

---
## Additional Links
- [GitHub respository](https://github.com/Astroinformatics/SummerSchool2022) for all of Astroinformatics Summer school
- Astroinformatics Summer school [Website & registration](https://sites.psu.edu/astrostatistics/astroinfo-su22/)
