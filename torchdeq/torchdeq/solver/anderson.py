"""
References:
    https://github.com/locuslab/deq 
    https://github.com/pv/scipy-work/tree/master/scipy/optimize
"""
import torch
import gc

from .utils import init_solver_info, batch_flatten, update_state, solver_stat_from_info


__all__ = ["anderson_solver"]


def anderson_solver(
    func,
    x0,
    max_iter, # 50
    tol, # 1e-3
    stop_mode, # 'abs'
    indexing=None, # None
    m=6, # 6
    lam=1e-4, # 1e-4
    tau=1.0, # 1.0
    return_final=False, # False
    **kwargs
):
    """
    Implements the Anderson acceleration for fixed-point iteration.

    Anderson acceleration is a method that can accelerate the convergence of fixed-point iterations. It improves
    the rate of convergence by generating a sequence that converges to the fixed point faster than the original
    sequence.

    Args:
        func (callable): The function for which we seek a fixed point.
        x0 (torch.Tensor): Initial estimate for the fixed point.
        max_iter (int, optional): Maximum number of iterations. Default: 50.
        tol (float, optional): Tolerance for stopping criteria. Default: 1e-3.
        stop_mode (str, optional): Stopping criterion. Can be 'abs' for absolute or 'rel' for relative. Default: 'abs'.
        indexing (None or list, optional): Indices for which to store and return solutions. If None, solutions are not stored. Default: None.
        m (int, optional): Maximum number of stored residuals in Anderson mixing. Default: 6.
        lam (float, optional): Regularization parameter in Anderson mixing. Default: 1e-4.
        tau (float, optional): Damping factor. It is used to control the step size in the direction of the solution. Default: 1.0.
        return_final (bool, optional): If True, returns the final solution instead of the one with smallest residual. Default: False.
        kwargs (dict, optional): Extra arguments are ignored.

    Returns:
        tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]: a tuple containing the following.
            - torch.Tensor: Fixed point solution.
            - list[torch.Tensor]: List of the solutions at the specified iteration indices.
            - dict[str, torch.Tensor]:
                A dict containing solver statistics in a batch.
                Please see :class:`torchdeq.solver.stat.SolverStat` for more details.

    Examples:
        >>> f = lambda z: 0.5 * (z + 2 / z)                 # Function for which we seek a fixed point
        >>> z0 = torch.tensor(1.0)                          # Initial estimate
        >>> z_star, _, _ = anderson_solver(f, z0)           # Run Anderson Acceleration
        >>> print((z_star - f(z_star)).norm(p=1))           # Print the numerical error
    """
    # x0 = x0.detach().clone().requires_grad_(False) # Todo@temp

    # Wrap the input function to ensure the same shape
    # funcsh = lambda _x: func(_x.view_as(x0)).reshape_as(_x)

    # Flatten the input tensor into (B, *)
    x0_flat = batch_flatten(x0)
    bsz, dim = x0_flat.shape

    alternative_mode = "rel" if stop_mode == "abs" else "abs"

    # Initialize tensors to store past values (X) and their corresponding function images (F)
    # X stores past iterates (the solution guesses)
    # F stores their corresponding values under the fixed-point function
    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device, requires_grad=False)
    F = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device, requires_grad=False)

    # Initialize the first two values for X and F
    # X[:, 0] holds the initial guess x0_flat
    X[:, 0] = x0_flat
    # F[:, 0] stores the result of applying the fixed-point function to x0_flat
    # F[:, 0] = funcsh(x0_flat)
    F[:, 0] = func(x0_flat.view_as(x0)).reshape_as(x0_flat)
    # X[:, 1] is updated to be the result of applying the function at x0 (essentially F[:, 0])
    X[:, 1] = F[:, 0]
    # F[:, 1] stores the result of applying the function to F[:, 0]
    # F[:, 1] = funcsh(F[:, 0])
    F[:, 1] = func(F[:, 0].view_as(x0)).reshape_as(F[:, 0])

    # Initialize tensors for the Anderson mixing process
    # Initialize tensor H, used to solve the least-squares problem in Anderson mixing
    # H is a matrix used to store inner products of differences between past iterates 
    H = torch.zeros(
        bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device, requires_grad=False
    )
    # Initialize the first row and column of H to 1
    # These represent boundary conditions for the Anderson mixing algorithm
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    # Initialize the vector y for the least-squares system (contains boundary condition elements)
    y = torch.zeros(
        bsz, m + 1, 1, dtype=x0.dtype, device=x0.device, requires_grad=False
    )
    # First element of y is 1, corresponding to the first row of H being all 1s
    y[:, 0] = 1

    # Initialize dictionaries to track solver information, the lowest error value, 
    # and the best solution so far
    trace_dict, lowest_dict, lowest_step_dict = init_solver_info(bsz, x0.device)
    lowest_xest = x0 # Keep track of the best estimate for the solution

    # List to store indices of previous iterates (not yet used in the loop)
    indexing_list = []

    # Begin the Anderson mixing iterations, starting from the third iteration
    for k in range(2, max_iter):
        # Apply the Anderson mixing process to compute a new estimate
        # Determine how many past iterates to consider (limited by memory m)
        n = min(k, m)

        # Compute G, the differences between F (function values) and X (iterates) for the last n steps
        # G stores the residual differences, G = F - X
        G = F[:, :n] - X[:, :n]

        # Update the H matrix with the inner product of G with itself, 
        # incorporating a regularization term 'lam'
        H[:, 1 : n + 1, 1 : n + 1] = (
            # Compute Gram matrix of G
            torch.bmm(G, G.transpose(1, 2))
            # Add regularization (lambda) to avoid singular matrices
            + lam
            * torch.eye(n, dtype=x0.dtype, device=x0.device, requires_grad=False)[None]
        )

        # Solve the least-squares system to find the optimal mixing coefficients alpha
        # H alpha = y, where alpha is the coefficient vector to minimize residuals
        alpha = torch.linalg.solve(H[:, : n + 1, : n + 1], y[:, : n + 1])[
            :, 1 : n + 1, 0
        ]

        # Update the next iterate X using the weighted combination of 
        # past F and X with coefficients alpha
        # tau is a mixing parameter controlling the weight 
        # between the past function values and iterates
        X[:, k % m] = (
            # Weighted sum of past function values F
            tau * (alpha[:, None] @ F[:, :n])[:, 0]
            # Weighted sum of past iterates X
            + (1 - tau) * (alpha[:, None] @ X[:, :n])[:, 0]
        )

        # Recompute the function at the new iterate X[:, k % m] and store it in F
        # F[:, k % m] = funcsh(X[:, k % m])
        F[:, k % m] = func(X[:, k % m].view_as(x0)).reshape_as(X[:, k % m])

        # Compute the difference between F and X, i.e., the residual gx
        gx = F[:, k % m] - X[:, k % m]
        # [B, dim] -> [B]

        # Compute the absolute difference (L2 norm of gx) for convergence monitoring
        abs_diff = gx.norm(dim=1) 
        # # Avoid division by zero
        rel_diff = abs_diff / (F[:, k % m].norm(dim=1) + 1e-9)

        # Update the state based on the new estimate
        lowest_xest = update_state(
            lowest_xest=lowest_xest,
            x_est=F[:, k % m].view_as(x0),
            nstep=k + 1,
            stop_mode=stop_mode,
            abs_diff=abs_diff,
            rel_diff=rel_diff,
            trace_dict=trace_dict,
            lowest_dict=lowest_dict,
            lowest_step_dict=lowest_step_dict,
            return_final=return_final,
        )

        # Store the solution at the specified index
        if indexing and (k + 1) in indexing:
            indexing_list.append(lowest_xest)
        # print('grad lowest_xest', lowest_xest.grad) # None
        # print('grad lowest_xest', lowest_xest.requires_grad) # False

        # If the difference is smaller than the given tolerance, terminate the loop early
        if not return_final and trace_dict[stop_mode][-1].max() < tol:
            # TODO:verbose
            for _ in range(max_iter - 1 - k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(
                    lowest_dict[alternative_mode]
                )
            break

    # If no solution was stored during the iteration process, store the final estimate
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    # Clear the memory
    X = None
    F = None
    # del X, F, G, H, y
    # del funcsh, gx, abs_diff, rel_diff
    # gc.collect()
    # torch.cuda.empty_cache()

    info = solver_stat_from_info(stop_mode, lowest_dict, trace_dict, lowest_step_dict)
    # [v.grad for k,v in info.items()]
    return lowest_xest, indexing_list, info
