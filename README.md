# README
This repository contains code for the persistent extension method and analogous bars method from the paper <a href="https://arxiv.org/abs/2201.05190">"Persistent Extension and Analogous Bars: Data-Induced Relations Between Persistence Barcodes" (Yoon, Ghrist, and Giusti, 2022)</a>. If you use these methods, please cite as the following.

@ARTICLE{yoonghristgiusti2022,  
      title={Persistent Extension and Analogous Bars: Data-Induced Relations Between Persistence Barcodes},   
      author={Hee Rhang Yoon and Robert Ghrist and Chad Giusti},  
      journal= {ArXiv e-prints},  
      archivePrefix={arXiv},  
      year={2022},  
      eprint={2201.05190},  
      primaryClass={math.AT}
}   

## Code
* We implement the <b>persistent extension method</b> and the <b>analogous bars method</b>. 
* While the math works in all dimensions, our code implements the methods in dimension 1. 
* The subset of code that computes the Witness (Dowker) persistence diagrams can be found at [Dowker_persistence](https://github.com/irishryoon/Dowker_persistence).

### Persistent extension method
* The extension method compares two distinct filtrations built on a common point cloud. There are three different types of comparisons implemented in this repository: VR to VR, VR to W, and W to VR.
* For each comparison, we implemented the <b>bar-to-bars persistent extension method</b> (Algorithm 3 in paper) and the <b>cycle-to-bars persistent extension method</b> (Algorithm 5 in paper).
* All algorithms are implemented via component-wise extension (Section 4.4 of paper) with F2 coefficients.
* There are six different functions that implement the extension method.
	* Implementation of <b> bar-to-bars extension method:</b>
		* `run_extension_VR_to_VR_bar`: Given a bar in a Vietoris-Rips barcode, function finds the cycle & bar extensions in the other Vietoris-Rips barcodes. See notebook `examples/EXAMPLE_EXTENSION_VR_VR.ipynb` for example. 
		* `run_extension_VR_to_W_bar`: Given a bar in the Vietoris-Rips barcode, function finds the cycle & bar extensions in the Witness filtration.
		* `run_extension_W_to_VR_bar`: Given a bar in the Witness barcode, function finds the cycle & bar extension in the Vietoris-Rips filtration. See notebook `examples/EXAMPLE_EXTENSION_W_VR.ipynb` for example. 
	* Implementation of the <b>cycle-to-bars extension method: </b>
		* `run_extension_VR_to_VR`: Given a cycle in the Vietoris-Rips filtration, function finds the cycle & bar extensions in the other Vietoris-Rips filtration.
		* `run_extension_VR_to_W`: Given a cycle in the Vietoris-Rips filtration, function finds the cycle & bar extensions in the Witness filtration.
		* `run_extension_W_to_VR`: Given a cycle in the Witness filtration, function finds the cycle & bar extensions in the Vietoris-Rips filtration.
	
* To view documentation on a function, run `?function_name` on a Jupyter notebook.

### Analogous bars method

#### Similarity-centric analogous bars method
* Call function `run_similarity_analogous()`
* See `examples/EXAMPLE_SIMILARITY_CENTRIC_ANALOGOUS_BARS.ipynb` for exmaple notebook implementation.
 
#### Feature-centric analogous bars method
* There is no single function that implements the feature-centric analogous bars method. Instead, the user will have to run the functions `extension_VR_to_W_bar` and `extension_W_to_VR` with appropriate inputs. We chose such implementation to avoid long computation times. 
* See `examples/EXAMPLE_FEATURE_CENTRIC_ANALOGOUS_BARS.ipynb` for an example implementation.

## Quick start
#### Persistent Extension method
Let's apply the persistent extension method to compare two Vietoris-Rips barcodes. The following block of code samples a random point cloud, computes two distinct distances, computes persistence, and plots the 1-dimensional barcodes. 

```
# load libraries
include("extension_method.jl")
using .ext
using Distances
using Eirene
using Plots

# generate random point cloud
P = rand(3, 50)

# compute two different distance matrices
D_Z = pairwise(Euclidean(), P, dims = 2)
D_Y = pairwise(Cityblock(), P, dims = 2)

# run Eirene 
C_Z = eirene(D_Z, record = "all")
C_Y = eirene(D_Y, record = "all")

# plot barcodes
barcode_Z = barcode(C_Z, dim = 1)
barcode_Y = barcode(C_Y, dim = 1)
p1 = plot_barcode(barcode_Z, title = "barcode(VR(Z))")
p2 = plot_barcode(barcode_Y, title = "barcode(VR(Y))")
plot(p1, p2, layout = grid(2,1))
```
From exmaining the resulting barcode, let's say that we are interested in understanding how bar 1 of `barcode(VR(Z))` is represented in `barcode(VR(Y))`. We can then run the following code to run extension and analyze the results

```
# select a bar of interest and run extension 
Z_bar = 1
extension = run_extension_VR_to_VR_bar(C_Z = C_Z, D_Z = D_Z, C_Y = C_Y, D_Y = D_Y, Z_bar = Z_bar, dim = 1)

# explore baseline and offset bar extensions
return_extension_results_at_parameter(extension)

# find all cycle extensions and bar extensions under fixed interval decomposition of barcode(VR(Y))
CE, BE_fixed = find_CE_BE(extension)

# find alternative bar extensions for a specific bar extension
param = extension["epsilon_0"]
bar_ext = BE_fixed[param][0]
alt_ext = find_alternative_bar_extension(extension, param, bar_extension = bar_ext);
```

#### Analogous bars method
Let's apply the analogous bars method to find bars in two barcodes that represent similar features. The following generates two point clouds and computes the VR barcodes in dimension 1.

```
# load libraries
include("extension_method.jl")
using .ext
using Distances
using Eirene
using Plots

# generate random point cloud
P = rand(3, 50)
Q = rand(3, 50)

# compute distance matrices
D_P = pairwise(Euclidean(), P, dims = 2)
D_Q = pairwise(Euclidean(), Q, dims = 2)
D_P_Q = pairwise(Euclidean(), P, Q, dims = 2)
D_Q_P = pairwise(Euclidean(), Q, P, dims = 2)

# Compute Vietoris-Rips persistence on two regions
dim = 1
VR_P = eirene(D_P, record = "all", maxdim = dim)
VR_Q = eirene(D_Q, record = "all", maxdim = dim)

# compute Witness persistence
W_P = compute_Witness_persistence(D_P_Q, maxdim = dim)
W_Q = compute_Witness_persistence(D_Q_P, maxdim = dim)

# plot all four barcodes
barcode_VR_P = barcode(VR_P, dim = 1)
barcode_W_P = barcode(W_P["eirene_output"], dim = 1)
barcode_W_Q = barcode(W_Q["eirene_output"], dim = 1)
barcode_VR_Q = barcode(VR_Q, dim = 1)

p1 = plot_barcode(barcode_VR_P, title = "Barcode(VR(P))")
p2 = plot_barcode(barcode_W_P, title = "Barcode(W(P,Q))")
p3 = plot_barcode(barcode_W_Q, title = "Barcode(W(Q,P))")
p4 = plot_barcode(barcode_VR_Q, title = "Barcode(VR(Q))")
plot(p1, p2, p3, p4, layout = grid(4,1), size = (500, 700))
```
From the resulting barcode, user must select a bar of interest in `barcode(W(P,Q))`. Once a bar is selected, we can run the similarity-centric analogous bars method and plot the baseline results as follows.

```
# select bar
W_P_bar = 1

# run extension
extension_P, extension_Q = run_similarity_analogous(VR_P = VR_P,
                                                    D_P = D_P,
                                                    VR_Q = VR_Q,
                                                    D_Q = D_Q,
                                                    W_PQ = W_P,
                                                    W_PQ_bar = W_P_bar,
                                                    dim = 1)

plot_analogous_bars(extension_P, extension_Q)
```


## Examples
* See the "EXAMPLES" directory to get started:
	* `EXAMPLE_EXTENSION_VR_VR.ipynb` : Example use of extension method to compare two Vietoris-Rips barcodes 
	* `EXAMPLE_EXTENSION_W_VR.ipynb`: Example use of the extension method to compare a Witness barcode and a Vietoris-Rips barcode	.
	* `EXAMPLE_FEATURE_CENTRIC_ANALOGOUS_BARS.ipynb`: Example use of the feature-centric analogous bars method.
	* `EXAMPLE_SIMILARITY_CENTRIC_ANALOGOUS_BARS.ipynb`: Example use of the similarity-centric analogous bars method.
* For additional examples, see the `EXAMPLES/data` directory.

## Understanding outputs
* The main functions output dictionaries that summarize the component-wise extension results. We thus provide various tools for extracting the cycle and bar extensions from these dictionaries. If you wish to understand the dictionary output of the main functions, please read `code_details/code_details.pdf`.

### Understanding the cycle extensions & bar extensions (under a fixed interval decomposition)
* Note that cycle extensions do not depend on the interval decomposition of the target filtration. The bar extension, however, may change depending on the target filtration. For our computations, we use the default interval decomposition that is used by <a href="https://github.com/Eetion/Eirene.jl">Eirene</a>.
* <b>Understanding the parameters </b>:  We recommend using the function `plot_pY` to plot all parameters `p_Y` at which an extension occurs. 
* <b> Understanding the baseline & offset bar extensions </b>: Using the function `return_extension_results_at_parameter()`, users can explore baseline and offset bar extensions at specific parameters. Given a parameter and offset bar extensions, the function returns a visualization of the resulting (baseline + offset) bar extension. 
* <b>Finding all cycle extensions and bar extensions</b>
	* Function `find_CE_BE_at_param()` will find all cycle extensions and bar extensions at a specific parameter
	* Function `find_CE_BE()` will find all cycle extensions and bar extensions at every parameter. 


### Understanding the alternative bar extensions
* Given a cycle extension, its corresponding bar extension depends on the choice of the interval decomposition of the target filtration. There are three different ways to explore alternative bar extensions. 
	* `find_alt_BE()`: Finds all bar extensions for every possible cycle extension under all possible interval decompositions. We recommend using this method when analyzing dataset with small barcodes (small number of bars in both the auxiliary filtration barcode and the target barcode) 
	* `find_alt_BE_at_param()`: Finds all bar extensions for every possible cycle extension under all possible interval decompositions <b>at a specific parameter</b>. Recommended for medium-sized barcodes.
	* `find_alternative_bar_extension()`: Given a specific bar extension at some parameter, the function finds all alternative bar extensions under different interval decompositions. This method is recommended for datasets with large barcodes. Note that even this method may result in a memory error for large barcodes. 
 

## Notes for contributions
* If you wish to contribute to this project, reading `code_details/code_details.pdf` may help orient you. The document also contains a list of action items for future versions.
* For questions, please contact Iris Yoon (<irishryoon@gmail.com>) or Chad Giusti (<cgiusti@udel.edu>).
