import numpy as np
from scipy.sparse import bsr_array

def curvature(P, precomputed_powered_P, aperture = 20, smoothing = True):
	""" Diffusion Curvature
	Estimates curvature by measuring the amount of mass remaining within an initial neighborhood after t steps of diffusion. Akin to measuring the laziness of a random walk after t steps.
	
	Parameters
	----------
	P : n x n 
		The diffusion matrix of the graph, assumed to be in scipy sparse bsr_array format
	diffusion_powers : int, optional
		Number of steps of diffusion to take before measuring the laziness, by default 8
	aperture : int, optional
		The size of the initial neighborhood, from which the percentage of mass remaining in this neighborhood is calculated, by default 20
	smoothing : int, optional
		Amount of smoothing to apply. Currently works by multiplying the raw laziness values with the diffusion operator, as a kind of iterated weighted averaging; by default 1
	precomputed_powered_P : ndarray, optional
		Optionally pass a precomputed powered diffusion operator, to speed up computation, by default None
	avg_transition_probability: bool, default True
		Use the definition of diffusion curvature in which the summed transition probabilities are divided by the total number of points in the aperture neighborhood.
		As a result, gives not the summed "return probability within the neighborhood" but the average return probability to each point in the aperture neighborhood.
		This formulation of diffusion curvature was used in a proof given in our NeurIPS 2022 paper.
	
	Returns
	-------
	length n array
		The laziness curvature values for each point
	"""
	# Set diffusion probability thresholds by sampling 100 points, taking their k-highest diffusion prob, and returning the mean
	sample_number = 100
	row_partitions = np.empty(sample_number)
	for idx, i in enumerate(np.random.randint(P.shape[0], size=(sample_number))):
		row_partitions[idx] = np.partition(P.getrow(i).data,-aperture)[-aperture]
	P_threshold = np.mean(row_partitions)
	# Calculate thresholded values, as a mask of 0s and 1s
	P_thresholded = (P >= P_threshold).astype(int)
	# Compute powers of P
	if precomputed_powered_P is not None:
		P_powered = precomputed_powered_P
	else:
		raise ValueError("Fast Diffusion Curvature currently requires a precomputed powered t")
	# take the diffusion probs of the neighborhood
	near_neighbors_only = P_powered * P_thresholded
	laziness_aggregate = np.sum(near_neighbors_only,axis=1)
	# local_density = # TODO: Divining this is too expensive for huge numbers of elements, and assuming uniform density (or equivalent normalizations with the kernel), it should be roughly equal everywhere.
	# # divide by the number of neighbors diffused to
	# # TODO: In case of isolated points, replace local density of 0 with 1. The laziness will evaluate to zero.
	# local_density[local_density==0]=1
	# laziness_aggregate = laziness_aggregate / local_density
	laziness = laziness_aggregate
	# TODO: Implement fast smoothing; for now we assume a single power of P is plausible
	if smoothing:
		# currently applies only a single application of smoothing
		average_laziness = P @ laziness[:,None]
		average_laziness = average_laziness.squeeze()
		laziness = average_laziness
	return laziness
	
