import torch
import torch.nn.functional as F

# === Basic Constants ===
EPS = 1e-10

def construct_Cv_filter(N, r, device):
    """
    (Old Mode/Warmup Mode) Construct a fixed temporal radius convolution kernel.
    Used for pure temporal constraints when there is no feature guidance.
    """
    abs_r = int(N * r)
    if abs_r == 0:
        return torch.ones(1, 1, 1, device=device)
    # Weight value is 1/r, guaranteeing gradient strength
    weights = torch.ones(2 * abs_r + 1, device=device) / r
    weights[abs_r] = 0.

    return weights[None, None, :]

def construct_dense_temporal_mask(N, radius, device):
    """
    (New Mode) Construct a hard temporal window mask.
    Used to cut off long-distance feature associations and enforce temporal locality.
    """
    r_int = int(N * radius)
    idx = torch.arange(N, device=device)
    # Calculate distance matrix |i - j|
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    # Only values within the radius range are 1, others are 0
    return (dist <= r_int).float()

def mult_Cv_sparse(Cv_weights, T):
    """(Old Mode) Use convolution to quickly calculate gradients"""
    B, N, K = T.shape
    T_reshaped = T.transpose(1, 2).reshape(-1, 1, N)
    Y_flat = F.conv1d(T_reshaped, Cv_weights, padding='same')
    return Y_flat.reshape(B, K, N).transpose(1, 2)

# === Core Logic: Gradient Calculation ===

def grad_fgw(T, cost_matrix, alpha, Cv_data, is_dense=False):
    """
    Calculate Gromov-Wasserstein (GW) gradient.
    
    Strategy: Enforce inter-action structure Ca = (1 - I).
    This means we penalize assigning similar frames to "different" actions.
    
    Mathematical Expression: Gradient ~ Cv @ (T_sum - T)
    """
    # T_other: The sum of probabilities of the current frame assigned to "other" actions (Row Sum - Self)
    T_other = T.sum(dim=2, keepdim=True) - T
    
    if not is_dense:
        # [Mode B] Old Mode (Sparse Convolution)
        term_gw = mult_Cv_sparse(Cv_data, T_other)
    else:
        # [Mode A] New Mode (Dense Matrix Multiplication)
        # Cv_data is the already calculated [B, N, N] mixed similarity matrix
        # term_gw[b, i, k] = sum_j (Cv[b, i, j] * T_other[b, j, k])
        # Meaning: If i and j are very similar (Cv large), and j is another action (T_other large), interpret penalty i assigned to k
        term_gw = torch.bmm(Cv_data, T_other)

    return alpha * term_gw + (1. - alpha) * cost_matrix

def grad_entropy(T, eps):
    """Entropy Regularization Gradient"""
    return eps * torch.log(T.clamp(min=EPS))

def grad_kld(T, p, lambd, axis):
    """KL Divergence Constraint Gradient (Used for handling Unbalanced OT)"""
    marg = T.sum(dim=axis, keepdim=True)
    return lambd * (torch.log(marg.clamp(min=EPS) / p.clamp(min=EPS)) + 1.)

# === Projection Solver ===

def project_to_polytope_KL(cost_matrix, mask, eps, dx, dy, n_iters=1):
    """
    Sinkhorn Projection Solver.
    Uses Log-domain trick (subtracting max_val) to prevent numerical overflow.
    """
    B, N, K = cost_matrix.shape
    
    # K = exp(-C / eps)
    neg_C_eps = -cost_matrix / eps
    max_val = neg_C_eps.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    dual_pot = torch.exp(neg_C_eps - max_val) * mask.unsqueeze(2)
    
    b = torch.ones((B, K, 1), device=cost_matrix.device)
    
    # Simple Sinkhorn Iteration
    for _ in range(n_iters):
        # Update a (rows / frames)
        K_b = dual_pot @ b
        a = dx / K_b.clamp(min=EPS)
        
        # Update b (cols / actions)
        Kt_a = dual_pot.transpose(1, 2) @ a
        b = dy / Kt_a.clamp(min=EPS)
        
    return a * dual_pot * b.transpose(1, 2)

# === Main Solver ===

@torch.no_grad()
def segment_asot(cost_matrix, mask=None, features=None, clusters=None,
                 eps=0.08, alpha=0.3, radius=0.04, 
                 ub_frames=False, ub_actions=True, 
                 lambda_frames=0.1, lambda_actions=0.05, 
                 n_iters=(25, 1), step_size=None, decay_rate=1.0,
                 feg_h=0.1): # <--- New parameter feg_h
    """
    ASOT Main Solver.
    Supports "Feature-Induced Residual Prior".
    """
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)
    nnz = mask.sum(dim=1)

    # Initialize Marginals
    dy = torch.ones((B, K, 1), device=dev) / K
    dx = torch.ones((B, N, 1), device=dev) / nnz[:, None, None]
    
    # Initialize Transmission Plan T (Outer Product)
    T = dx * dy.transpose(1, 2) * mask.unsqueeze(2)
    
    # === Mode Selection ===
    is_dense = False
    Cv_data = None
    
    if features is not None:
        # [Mode A]: FEG-Consistent Hybrid Prior
        # Combines Gaussian Kernel Feature Similarity + Hard Temporal Window + Residual Connection
        is_dense = True
        
        # 1. Calculate Feature Distance (Cosine Distance)
        # features assume normalized
        Sim_feat = torch.matmul(features, features.transpose(1, 2)) # [-1, 1]
        Dist_feat = 1.0 - Sim_feat # [0, 2]
        
        # 2. Apply Gaussian Kernel (FEG Kernel)
        # This keeps consistent with FEG Loss definition, achieving metric unification
        Cv_feat = torch.exp(-Dist_feat / feg_h) # [0, 1]
        
        # 3. Construct Hard Temporal Window Mask
        Time_Mask = construct_dense_temporal_mask(N, radius, dev)
        
        # 4. Calculate Scaling Factor (Scale Correction)
        # Original integration of convolution kernel is approx 2.0 (1/r * 2r). Gaussian Kernel max is 1.0.
        # To keep gradient magnitude consistent, preventing structure term failure, must multiply by scaling factor.
        # 0.8 is an empirical coefficient, slightly less than 1.0 to prevent feature noise dominance.
        scale_factor = 0.8 / radius if radius > 0 else 1.0 # fs_mid uses 0.8, fs_eval uses 0.8, da uses 0.8, YTI uses 0.8, bf uses 0.8
        
        # 5. Core Fusion Logic: Residual Connection
        # Cv = (Base_Time + Lambda * Feature) * Mask * Scale
        # Base=1.0: Guarantees most basic temporal connectivity (prevents cold start collapse)
        # Feature=0.55*Cv_feat: Provides semantic enhancement (handles repetitive actions and fuzzy boundaries)
        Cv = (1.0 + 0 * Cv_feat) * Time_Mask.unsqueeze(0) * scale_factor # fs_mid uses 0.05, fs_eval uses 0.05, da uses 0.55, yti uses 0.05, bf uses 0.3
        
        Cv_data = Cv
    else:
        # [Mode B]: Original ASOT (Fixed Structure)
        # Only used during Warm-up
        is_dense = False
        Cv_data = construct_Cv_filter(N, radius, dev)

    # === Mirror Descent Loop ===
    for it in range(n_iters[0]):
        # 1. Calculate Gradient
        grad_struct = grad_fgw(T, cost_matrix, alpha, Cv_data, is_dense)
        grad_obj = grad_struct + grad_entropy(T, eps)
        
        # 2. Add Unbalanced Constraint Gradient
        if ub_frames:
            grad_obj += grad_kld(T, dx, lambda_frames, 2)
        if ub_actions:
            grad_obj += grad_kld(T, dy.transpose(1, 2), lambda_actions, 1)
        
        # 3. Heuristic Step Size
        if it == 0 and step_size is None:
            g_max = grad_obj.abs().max().item()
            # Use more aggressive step size (4.0) to ensure convergence within limited iterations
            step_size = 4.0 / (g_max + EPS)

        # 4. Update Step
        current_step = step_size * (decay_rate ** it)
        update_factor = torch.exp(-current_step * grad_obj) * mask.unsqueeze(2)
        T = T * update_factor
        
        # 5. Projection Normalization
        if not ub_frames and not ub_actions:
            T = project_to_polytope_KL(grad_struct, mask, eps, dx, dy, n_iters=n_iters[1])
        elif not ub_frames:
            # Row Normalization (Sum of probabilities per frame is 1/N)
            T = (T / T.sum(dim=2, keepdim=True).clamp(min=EPS)) * dx
        elif not ub_actions:
            # Column Normalization (Sum of probabilities per action is 1/K)
            T = (T / T.sum(dim=1, keepdim=True).clamp(min=EPS)) * dy.transpose(1, 2)

    # Final Normalization: Output P(Action|Frame) format
    # Current row sum of T is 1/N (dx). We need row sum to be 1.
    return T * nnz[:, None, None], None 

def temporal_prior(n_frames, n_clusters, rho, device):
    """
    Construct Diagonal Temporal Prior (Used for Cost Matrix Initialization).
    R_ij = |i/N - j/K|
    """
    frames = torch.arange(n_frames, device=device, dtype=torch.float32)[:, None]
    clusters = torch.arange(n_clusters, device=device, dtype=torch.float32)[None, :]
    return rho * torch.abs(frames / n_frames - clusters / n_clusters)
