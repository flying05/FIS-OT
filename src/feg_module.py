import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class FEGTripletLoss(nn.Module):
    """
    FEG (Feature Enhanced Graph) Triplet Loss Function
    
    Core concept: By mixing temporal and semantic similarity distributions, mine positive samples 
    that are temporally adjacent or semantically similar, and negative samples at fuzzy boundaries 
    (Semi-hard negatives). Use KL divergence to pull positive samples closer and push negative samples further away.
    """
    def __init__(self, margin=1.0, num_classes=10, max_samples=1000, h_param=0.05):
        super(FEGTripletLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.max_samples = max_samples
        self.h_param = h_param  # Semantic Gaussian kernel bandwidth parameter (h)
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    def _compute_semantic_similarity(self, features):
        """
        Compute semantic similarity matrix (f_s)
        f_s = exp(-(1 - cos_sim) / h)
        """
        # features are normalized, matrix multiplication is Cosine Similarity
        # Range: [-1, 1]
        sim_cos = torch.mm(features, features.t())
        
        # Convert to distance: 0 (same) ~ 2 (opposite)
        # clamp to avoid tiny negative numbers due to numerical errors
        dist = 1.0 - torch.clamp(sim_cos, -1.0, 1.0)
        
        # Apply Gaussian kernel
        G = torch.exp(-dist / self.h_param)
        
        # Row normalization to probability distribution
        row_sum = torch.sum(G, dim=1, keepdim=True) + 1e-16
        sim_semantic = G / row_sum
        
        return sim_semantic

    def _compute_temporal_similarity(self, T, device):
        """
        Compute temporal similarity matrix (f_t)
        w(d) = -1 + 2 * exp(-d / beta)
        """
        # Dynamically calculate beta: make similarity decay to half at window boundary
        # Assume average action length is about T / num_classes
        action_len = max(1.0, float(T) / self.num_classes)
        beta = -action_len / (2 * math.log(0.5))
        
        # Time index
        idx = torch.arange(T, device=device, dtype=torch.float32)
        # Distance matrix |i - j|
        dist_matrix = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
        
        # Compute weight w(d)
        temporal_raw = -1 + 2 * torch.exp(-dist_matrix / beta)
        
        # ReLU truncates negative values, keeping only positively correlated parts (Local Neighborhood)
        sim_temporal = F.relu(temporal_raw)
        
        # Row normalization
        row_sum = torch.sum(sim_temporal, dim=1, keepdim=True) + 1e-16
        sim_temporal = sim_temporal / row_sum
            
        return sim_temporal

    def forward(self, features, alpha_logit, mask=None):
        """
        Args:
            features: [B, T, D]
            alpha_logit: Learnable parameter for mixing weight
        """
        B, T, D = features.shape
        device = features.device
        
        total_loss = 0.0
        num_valid_batches = 0
        
        # Mixing weight sigmoid(logit) -> (0, 1)
        alpha = torch.sigmoid(alpha_logit)
        
        for b in range(B):
            # --- 1. Preprocessing and Sampling ---
            if mask is not None:
                valid_idx = torch.where(mask[b])[0]
                if len(valid_idx) < 2: continue
                feat_b = features[b][valid_idx]
                curr_T = len(valid_idx)
            else:
                feat_b = features[b]
                curr_T = T
            
            # Downsampling strategy (if sequence is too long, GPU memory insufficient)
            if curr_T > self.max_samples:
                # Uniform downsampling preserves temporal structure better than random sampling
                indices = torch.linspace(0, curr_T-1, self.max_samples).long().to(device)
                feat_b = feat_b[indices]
                curr_T = self.max_samples

            # --- 2. Construct Mixed Distribution ---
            sim_s = self._compute_semantic_similarity(feat_b)      # [T, T]
            sim_t = self._compute_temporal_similarity(curr_T, device) # [T, T]
            
            # f_ts = alpha * f_t + (1-alpha) * f_s
            mts = alpha * sim_t + (1 - alpha) * sim_s
            
            # --- 3. Triplet Mining (Hard Sample Mining) ---
            # To select samples by index, we need detached numpy array
            mts_np = mts.detach().cpu().numpy()
            
            batch_loss = 0.0
            valid_anchors = 0
            
            # Randomly sample parts of Anchors for training (Stochastic Pooling idea, reduce computation)
            # e.g., calculate only 20% of Anchors
            anchor_indices = np.random.choice(curr_T, size=max(1, curr_T // 5), replace=False)
            
            for i in anchor_indices:
                sim_row = mts_np[i]
                
                # --- Positive: Top 5% ---
                # Exclude self (i)
                sorted_indices = np.argsort(sim_row)[::-1] # Descending
                sorted_indices = sorted_indices[sorted_indices != i]
                
                if len(sorted_indices) == 0: continue
                
                # Take Top 5%
                k_top = max(1, int(len(sorted_indices) * 0.05))
                pos_candidates = sorted_indices[:k_top]
                pos_idx = np.random.choice(pos_candidates)
                
                # --- Negative: Mean < Sim < Mean + Std (Semi-hard) ---
                # FEG paper strategy: select between mean and mean + std
                mean_sim = np.mean(sim_row)
                std_sim = np.std(sim_row)
                upper_bound = mean_sim + std_sim
                lower_bound = mean_sim
                
                neg_candidates = np.where((sim_row > lower_bound) & (sim_row < upper_bound))[0]
                neg_candidates = neg_candidates[neg_candidates != i]
                
                if len(neg_candidates) > 0:
                    neg_idx = np.random.choice(neg_candidates)
                else:
                    # Fallback 1: If no semi-hard, try to find hard negatives (Sim > Mean+Std)
                    hard_neg = np.where(sim_row > upper_bound)[0]
                    hard_neg = hard_neg[hard_neg != i]
                    if len(hard_neg) > 0:
                        neg_idx = np.random.choice(hard_neg)
                    else:
                        # Fallback 2: Randomly select one below mean (Easy Negative)
                        easy_neg = np.where(sim_row < mean_sim)[0]
                        if len(easy_neg) > 0:
                            neg_idx = np.random.choice(easy_neg)
                        else:
                            continue # Give up if really can't find one

                # --- 4. KL Divergence Loss ---
                # KL(P || Q) = sum P * log(P/Q)
                # Here input (Q) should be log_softmax, target (P) is probs
                # We want Anchor(P) and Positive(Q) to be close -> KL(P||Q) small
                
                # Note: nn.KLDivLoss(input, target) -> input=log_probs, target=probs
                dist_pos = self.kl_div_loss(torch.log(mts[pos_idx] + 1e-16), mts[i])
                dist_neg = self.kl_div_loss(torch.log(mts[neg_idx] + 1e-16), mts[i])
                
                # Triplet Margin Loss
                loss_val = F.relu(dist_pos - dist_neg + self.margin)
                
                batch_loss += loss_val
                valid_anchors += 1
            
            if valid_anchors > 0:
                total_loss += batch_loss / valid_anchors
                num_valid_batches += 1
                
        if num_valid_batches > 0:
            return total_loss / num_valid_batches
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)

class FEGModule(nn.Module):
    """
    Plug-and-play FEG module
    Includes feature extractor and built-in Loss calculation
    """
    def __init__(self, input_dim, output_dim=64, num_layers=1, hidden_dim=512, 
                 feg_h_param=0.05, num_classes=10):
        super(FEGModule, self).__init__()
        
        # 1. Encoder Network Structure
        layers = []
        curr_dim = input_dim
        
        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim)) # BatchNorm helps convergence
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
            
        # Output layer (No ReLU, because Cosine Sim is calculated later, negative values also have meaning)
        layers.append(nn.Linear(curr_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        
        # 2. FEG Parameters
        self.alpha_logit = nn.Parameter(torch.tensor(0.0)) # Init alpha=0.5
        
        # 3. Loss Module
        self.feg_loss_fn = FEGTripletLoss(
            margin=1.0, 
            num_classes=num_classes, 
            h_param=feg_h_param
        )
        
    def forward(self, x, mask=None, return_loss=True):
        """
        Args:
            x: [B, T, Input_Dim]
            return_loss: Whether to compute Loss (Only valid during Training)
        Returns:
            features: [B, T, Output_Dim] (Normalized)
            loss: scalar tensor
        """
        B, T, D = x.shape
        
        # Merge batch & time for Linear layer
        x_flat = x.view(-1, D)
        z_flat = self.encoder(x_flat)
        features = z_flat.view(B, T, -1)
        
        # L2 Normalize (Critical for Cosine Similarity)
        features = F.normalize(features, p=2, dim=-1)
        
        loss = torch.tensor(0.0, device=x.device)
        
        # Compute Loss only in training mode and when needed
        if self.training and return_loss:
            loss = self.feg_loss_fn(features, self.alpha_logit, mask)
            
        return features, loss
