SUGAR Geometry-Based Data Generation
-------------------------------------------------------

SUGAR is a tool for generating high dimensional data that follows a low dimensional manifold. SUGAR (Synthesis Using Geometrically Aligned Random-walks) uses a diffusion process to learn a manifold geometry from the data. Then, it generates new points evenly along the manifold by pulling randomly generated points into its intrinsic structure using a diffusion kernel. SUGAR equalizes the density along the manifold by selectively generating points in sparse areas of the manifold.

[Ofir Lindenbaum, Jay S. Stanley III, Guy Wolf, Smita Krishnaswamy **Geometry-Based Data Generation**. 2018. *Arxiv*](https://arxiv.org/abs/1802.04927)


SUGAR has been implemented in Python3 and Matlab. Future support for the package is at https://github.com/stanleyjs/sugar
