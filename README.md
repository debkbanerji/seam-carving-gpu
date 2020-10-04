# GPU Accelerated Seam Carving Variations

*Note: Currently a work in progress, so some or all implementations may be incomplete - contact me if you're interested in using these. May or may not be currently maintained based on other priorities.*


TODO: describe novel approach to parallelization


## TODO
### 1. Implement increasing of sizes using seam insertion
This is fairly straightforward algorithm wise, but is tedious to code relative to how interesting it is, and there are workarounds that achieve a similar affect (scaling up the image normally and then shrinking). Even so, this is a nice to have and would add to the completeness of any of the variations presented here.
### 2. Finish forward energy implementation
This improves the quality of the result without much of a performance penalty, so it's a no brainer, but some of the energy map update code needs to be rewritten in order for this to work correctly. Work on this is partially done, in that a separate subfolder for this implementation has been created.
### 3. Further memory optimizations

### 4. Add in clustering algorithm variation (single K)
This is a more novel improvement - it's a change in the seam carving algorithm itself that has the potential to greatly improve the algorithm's ability to reason about preserving the global structure of the image, something which seam carving has historically struggled with. This hasn't been done before, so I actually need to implement it to see how well it works.

TODO: describe

This would be painful to code due to the additional dimension added to the dynamic programming table, and the memory allocation code that would have to be rewritten to accommodate this. The table shape has to be changed since the energy function used is complex in that the energies are no longer additive. Whenever I do this, I'll likely directly implement the multiple K variation, and test it with k = 1 to get the results for this variation. Though the multiple K variation would be even more painful to code, this way I would not have to implement this twice.

### 4. Add in clustering algorithm variation (multiple K)

