import numpy as np

def dmr_loading_level_single_J(eigvals_D,z_D,replica,wngc_dB=-3,wngc_tol_dB=1,max_iterations=30,epsilon_tolerance=1e-3):
    """Find a diagonal load that satisfies one beam's WNG constraint.

    This is the scalar counterpart of :func:`dmr_wng_batch_J`. The bracketed
    search implements the direction-dependent norm constraint described by
    Cox et al. (2005), equations (20)--(23). WNG is evaluated in the retained
    eigenbasis using the definition in Hulbert and Wage (2022), equation (7).
    Full citations are in ``docs/references.md``.
    """

    # For a unity-gain beam, WNG = 1 / (w^H w). The unshaded CBF
    # value ||v||^2 is the maximum available WNG for this aperture.
    unshaded_wng = (replica.conj().T@replica).real
    minimum_wng = unshaded_wng*10**((wngc_dB-wngc_tol_dB)/10)

    # scale epsilon relative to the covariance
    cov_scale = np.sum(eigvals_D)/len(replica)

    # Start close to the unloaded solution, then expand the upper bound until
    # it satisfies the WNG target.
    epsilon_low = 1e-10*cov_scale
    wng_low = dmr_compute_wng_single_J(eigvals_D,z_D,unshaded_wng,epsilon_low)

    if wng_low>=minimum_wng:
        return epsilon_low, wng_low

    epsilon_high = epsilon_low*4
    while dmr_compute_wng_single_J(eigvals_D,z_D,unshaded_wng,epsilon_high)<minimum_wng:
        epsilon_high *= 4
        if epsilon_high>1e8*cov_scale:
            raise RuntimeError("Could not reach requested WNG target")
    
    # Bisect geometrically because useful loading values span many decades.
    for _ in range(max_iterations):
        # split the sqrt so two very small float32s don't go to 0 before the sqrt
        epsilon_mid = np.sqrt(epsilon_low)*np.sqrt(epsilon_high)
        wng_mid = dmr_compute_wng_single_J(eigvals_D,z_D,unshaded_wng,epsilon_mid)

        if wng_mid < minimum_wng:
            epsilon_low = epsilon_mid
        else:
            epsilon_high = epsilon_mid

        if epsilon_high/epsilon_low - 1.0 < epsilon_tolerance:
            break
    
    epsilon = epsilon_high

    return epsilon, dmr_compute_wng_single_J(eigvals_D,z_D,unshaded_wng,epsilon)

def dmr_compute_wng_single_J(eigvals_D,z_D,unshaded_wng,epsilon):
    """Compute one beam's WNG without forming its full weight vector.

    The loaded covariance is
    ``R_loaded = U_D diag(eigvals_D) U_D^H + epsilon I``. The reduction is
    obtained by expanding ``1 / (w^H w)`` in the retained eigenbasis; compare
    Cox et al. (2005), equations (22)--(23), with the WNG definition in
    Hulbert and Wage (2022), equation (7).
    """

    # z_D is the steer vector coordinates in the retained subspace
    q_D = np.abs(z_D) ** 2

    # accounting for the energy in the discarded space
    q_null = max(unshaded_wng - np.sum(q_D),0)

    # inverse_D = 1/(eigvals_D+epsilon)
    #
    # s1 = np.sum(q_D * inverse_D) + q_null/epsilon
    # s2 = np.sum(q_D * inverse_D**2) + q_null /epsilon**2
    
    # Factoring out epsilon avoids division by a very small load. The common
    # factor cancels between the squared response and weight norm.
    scaled_inverse_D = epsilon/(eigvals_D+epsilon)
    
    s1 = np.sum(q_D * scaled_inverse_D) + q_null
    s2 = np.sum(q_D * scaled_inverse_D**2) + q_null

    return s1**2/s2

def dmr_wng_batch_J(eigvals_D,z_DJ,replica_NJ,wngc_dB=-3,wngc_tol_dB=1,max_iterations=30,epsilon_tolerance=1e-3):
    """Find independent WNG-constrained diagonal loads for a beam batch.

    All beams share one retained eigensystem but receive separate loading
    values because the norm constraint depends on steering direction. This is
    the batched rank-one specialization of Cox et al. (2005), section II-E.
    """

    # Unshaded CBF WNG sets the per-beam reference; wngc_tol_dB relaxes the
    # requested relative target when deciding whether a load is feasible.
    unshaded_wng_J = np.sum(np.abs(replica_NJ)**2,axis=0)
    minimum_wng_J = unshaded_wng_J*10**((wngc_dB-wngc_tol_dB)/10)

    # scale epsilon relative to the covariance
    cov_scale = np.sum(eigvals_D)/replica_NJ.shape[0]

    # Split steering-vector energy into retained and discarded subspaces once.
    q_DJ = np.abs(z_DJ)**2
    q_null_J = np.maximum(unshaded_wng_J - np.sum(q_DJ,axis=0),0)

    # Keep loading bounds in float32, matching the streaming covariance.
    epsilon_low_J = np.full(replica_NJ.shape[1],cov_scale*1e-8,dtype=np.float32)
    epsilon_high_J = np.full(replica_NJ.shape[1],cov_scale,dtype=np.float32)

    # Try to bracket the desired solution for every beam.
    # The upper-bound search is limited to eight increases.
    # If the resulting upper bound meets the WNG constraint, bisection
    # refines the loading while retaining a feasible upper bound.
    # Finite iterations may leave more loading than necessary.
    #
    # If the upper bound does not meet the constraint, this function
    # still proceeds and returns a result; satisfying the requested
    # WNG constraint is then not guaranteed.
    for _ in range(8):
        wng_high_J = dmr_compute_wng_batch_J(eigvals_D,q_DJ,q_null_J,epsilon_high_J)
        idx_needs_larger_J = wng_high_J<minimum_wng_J

        if not np.any(idx_needs_larger_J):
            break

        epsilon_high_J[idx_needs_larger_J] *=10

    # Bisect in log-epsilon space while retaining a feasible upper bound.
    for _ in range(max_iterations):
        # bisect, split the sqrt so two very small float32s don't go to 0 before the sqrt
        epsilon_mid_J = np.sqrt(epsilon_low_J)*np.sqrt(epsilon_high_J)

        w_mid_J = dmr_compute_wng_batch_J(eigvals_D,q_DJ,q_null_J,epsilon_mid_J)

        needs_more_loading_J = w_mid_J < minimum_wng_J

        # Loads below the WNG target become new lower bounds; feasible loads
        # become new upper bounds.
        epsilon_low_J = np.where(needs_more_loading_J,epsilon_mid_J,epsilon_low_J)
        epsilon_high_J = np.where(needs_more_loading_J,epsilon_high_J,epsilon_mid_J)

        if np.all(epsilon_high_J/epsilon_low_J - 1.0 < epsilon_tolerance):
            break

    return epsilon_high_J, dmr_compute_wng_batch_J(eigvals_D,q_DJ,q_null_J,epsilon_high_J)

def dmr_compute_wng_batch_J(eigvals_D,q_DJ,q_null_J,epsilon_J):
    """Evaluate WNG for a beam batch in the retained covariance eigenbasis.

    ``q_DJ`` and ``q_null_J`` contain steering-vector energy in the retained
    and discarded subspaces. The returned expression is algebraically equal
    to ``1 / (w_j.conj().T @ w_j)`` after unity-response normalization.
    """

    # inverse_DJ = 1.0 / (eigvals_D[:,np.newaxis] + epsilon_J[np.newaxis,:])
    #
    # s1_J = np.sum(q_DJ * inverse_DJ,axis=0) + q_null_J/epsilon_J
    # s2_J = np.sum(q_DJ * inverse_DJ**2,axis=0) + q_null_J / epsilon_J**2

    # Factoring out epsilon avoids divisions by very small loads. The common
    # factor cancels between the squared response and weight norm.
    scaled_inverse_DJ = epsilon_J[None,:] / (eigvals_D[:,np.newaxis] + epsilon_J[np.newaxis,:])

    s1_J = np.sum(q_DJ * scaled_inverse_DJ,axis=0) + q_null_J
    s2_J = np.sum(q_DJ * scaled_inverse_DJ**2,axis=0) + q_null_J

    return s1_J**2 / s2_J
