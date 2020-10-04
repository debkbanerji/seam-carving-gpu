# GPU Accelerated Seam Carving Variations

*Note: Currently a work in progress, so some or all implementations may be incomplete - contact me if you're interested in using these. May or may not be currently maintained based on other priorities.*


TODO: describe novel approach to parallelization, with dependency array stuff, and how it can be applied to different dynamic programming problems whose dependenccy graphs contain multiple topological orderings


## TODO
Improvements and additional features I need to add once I have the time/motivation

### 1. Implement increasing of sizes using seam insertion
This is fairly straightforward algorithm wise, but is tedious to code relative to how interesting it is, and there are workarounds that achieve a similar affect (scaling up the image normally and then shrinking). Even so, this is a nice to have and would add to the completeness of any of the variations presented here.

### 2. Finish forward energy implementation
This improves the quality of the result without much of a performance penalty, so it's a no brainer, but some of the energy map update code needs to be rewritten in order for this to work correctly. Work on this is partially done, in that a separate subfolder for this implementation has been created.

### 3. Further memory optimizations
I've used CudaMallocManaged to allocate memory because it's easier to code, but being more explicit with how memory is allocated within GPU RAM may speed the code up.

### 4. More speed tests
More comprehensive speed test implementations, on small and large images, comparing the new parallel approach to the standard parallel approach as well as the single threaded approach graphed as a function of input size.

### 4. Add in clustering algorithm variation (single K)
This is a more novel improvement - it's a change in the seam carving algorithm itself that has the potential to greatly improve the algorithm's ability to reason about preserving the global structure of the image, something which seam carving has historically struggled with. This hasn't been done before, so I actually need to implement it to see how well it works.

The idea is that in addition to minimizing the energy values based on differences in values of pixels, we also try to maintain the sizes of different regions of the image using image segmentation with K means clustering. One nice thing about this variation is that the k means clustering preprocessing step can also take advantage of GPU acceleration (TODO: find a library that already implements a parallelized version of this using CUDA).

We use the k means clustering to label each pixel as belonging to a certain region, and then create a vector of length K (the number of regions) of the number of pixels belonging to each region. Our second objective, in addition to minimizing the old energy function is to maintain this region balance as well as we can when removing seams. In order to do this, we create an additional cumulative energy table variant that keeps track of the 'best' vector of pixels per region that could be created when removing a seam that ends at each respective pixel. The 'best' vector here can be calculated through dynamic programming from the previous values. In order to define 'best', we look at this vector in relation to the reference (original) vector and attempt to minimize the cross product after normalization, in order to make them point in the same direction as much as possible. Our new 'energy function' for each pixel is now a linear combination of the old energy function (after accumulation) and the cross product of the two normalized vectors. What makes this different than other energy function definitions is that it is not additive in the same way - the dynamic programming table is different and we do not use cumulative energy.

This would be painful to code due to the additional dimension added to the dynamic programming table, and the memory allocation code that would have to be rewritten to accommodate this. The table shape has to be changed since the energy function used is complex in that the energies are no longer additive. Whenever I do this, I'll likely directly implement the multiple K variation, and test it with k = 1 to get the results for this variation. Though the multiple K variation would be even more painful to code, this way I would not have to implement this twice.

### 5. Add in clustering algorithm variation (multiple K)

In the previous approach, K is a parameter that defines the number of regions, and the 'broadness' with which the algorithm looks at the image. Instead of using a single K, we can also choose multiple values of K. This means we run the clustering algorithm at the beginning, say N times, with different values of K. Each cell in our dynamic programming table now contains a length N vector of vectors, and we're minimizing a linear combination of the cross products of those vectors with the reference. The weights which we assign each cross product determine how much we want to preserve the regions at each scale.

### 6. Add in semantic image segmentation based approach

It's easy to see how instead of using K means clustering to determine the regions, we could use semantic image segmentation, likely using an existing convolutional neural network architecture.
