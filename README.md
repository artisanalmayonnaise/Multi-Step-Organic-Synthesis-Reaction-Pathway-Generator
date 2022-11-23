# Organic-Synthesis-Reaction-Pathway-Generator

This package can be used to generate the chemical organic reactions for a multi-step pathway that are required to transform a user-specified initial organic product 
into a user-specified final organic product. These chemical reactions are based off of those listed in the textbook "Organic Chemistry" by John McMurry. There are 
currently code for reactions involving alkenes, alkynes, and aromatics, with plans to expand this database in the near-future to include other functional groups. 

This package is unique in that it allows the user to specify initial and final products, which can be of great utility for undergraduate chemistry undergraduates studying
organic chemistry, who frequently encounter problems where they have to transform a defined initial product into a desired final product. It can also be of use to
laboratory chemists, who may be limited in their abilities to conveniently and cheaply acquire certain starting products and may have to rely on other starting products.
This package can also be easily-modified by researchers who may be more interested in studying how chemicals of interest may be transformed. 

The search through the reaction space can be accomplished extremely quickly, even for pathways that may involve ~5 reactions and are calculated on an ordinary laptop (ex.
a 5-step reaction pathway can be correctly calculated in less than 5 seconds on an 8 GB RAM, Intel i5 CPU laptop). The reaction search is accomplished using a 
self-pruning depth-first tree-search algorithm.
