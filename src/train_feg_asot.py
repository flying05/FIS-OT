import argparse
from copy import deepcopy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

# Import internal modules
from video_dataset import VideoDataset
import asot
from utils import *
from metrics import ClusteringMetrics, indep_eval_metrics

# Import FEG module (must ensure feg_module.py is in the same directory)
from feg_module import FEGModule

num_eps = 1e-11

class FEG_ASOT(pl.LightningModule):
    """
    Video Action Segmentation Model combining FEG (Feature Enhanced Graph) and ASOT (Optimal Transport)
    
    Encoder: FEG Module (Shallow MLP + Triplet Loss)
    Decoder/Pseudo-labeling: ASOT (Optimal Transport with Temporal Regularization)
    """
    def __init__(self, 
                 # General model parameters
                 lr=1e-4, weight_decay=1e-4, 
                 layer_sizes=[64, 128, 40], # [Input_Dim, Hidden_Dim, Output_Dim]
                 n_clusters=20, 
                 learn_clusters=True,
                 
                 # ASOT (Optimal Transport) parameters
                 alpha_train=0.3, alpha_eval=0.3, 
                 n_ot_train=[50, 1], n_ot_eval=[50, 1], 
                 step_size=None, train_eps=0.06, eval_eps=0.01, 
                 ub_frames=False, ub_actions=True,
                 lambda_frames_train=0.05, lambda_actions_train=0.05, 
                 lambda_frames_eval=0.05, lambda_actions_eval=0.01, 
                 temp=0.1, radius_gw=0.04, 
                 n_frames=256, rho=0.1, 
                 
                 # FEG (Feature Enhanced Graph) specific parameters
                 feg_weight=0.3,          # Weight to balance CE Loss and FEG Loss
                 feg_warmup_epochs=5,     # Warmup epochs for FEG Loss
                 feg_h_param=0.05,        # Semantic Gaussian kernel parameter
                 
                 # Other system parameters
                 exclude_cls=None, visualize=False):
        super().__init__()
        
        # Save all hyperparameters to self.hparams
        self.save_hyperparameters()
        
        # Explicitly save member variables for easy access
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_clusters = n_clusters
        self.learn_clusters = learn_clusters
        self.layer_sizes = layer_sizes
        self.exclude_cls = exclude_cls
        self.visualize = visualize

        # ASOT parameters
        self.alpha_train = alpha_train
        self.alpha_eval = alpha_eval
        self.n_ot_train = n_ot_train
        self.n_ot_eval = n_ot_eval
        self.step_size = step_size
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.radius_gw = radius_gw
        self.ub_frames = ub_frames
        self.ub_actions = ub_actions
        self.lambda_frames_train = lambda_frames_train
        self.lambda_actions_train = lambda_actions_train
        self.lambda_frames_eval = lambda_frames_eval
        self.lambda_actions_eval = lambda_actions_eval

        self.temp = temp
        self.n_frames = n_frames
        self.rho = rho

        # FEG parameters
        self.feg_weight = feg_weight
        self.feg_warmup_epochs = feg_warmup_epochs

        # === Initialize FEG Module ===
        # input_dim=layer_sizes[0] (original feature dimension), output_dim=layer_sizes[-1] (embedding dimension)
        # hidden_dim takes the middle dimension of layer_sizes
        hidden_dim = layer_sizes[1] if len(layer_sizes) > 2 else layer_sizes[0]
        
        self.feg_module = FEGModule(
            input_dim=layer_sizes[0],
            hidden_dim=hidden_dim,
            output_dim=layer_sizes[-1],
            num_layers=max(1, len(layer_sizes) - 2), # Automatically calculate intermediate layers
            feg_h_param=feg_h_param,
            num_classes=n_clusters # Used for calculating temporal window size
        )

        # Initialize cluster centers (Action Prototypes)
        d = layer_sizes[-1]
        self.clusters = nn.parameter.Parameter(
            data=F.normalize(torch.randn(self.n_clusters, d), dim=-1), 
            requires_grad=learn_clusters
        )

        # Initialize evaluation metrics
        self.mof = ClusteringMetrics(metric='mof')
        self.f1 = ClusteringMetrics(metric='f1')
        self.miou = ClusteringMetrics(metric='miou')
        
        # Cache for storing test results
        self.test_cache = []

    def get_feg_weight_schedule(self):
        """Dynamically adjust FEG Loss weight (Warmup strategy)"""
        if self.current_epoch < self.feg_warmup_epochs:
            # Linearly increase weight
            return self.feg_weight * (self.current_epoch + 1) / self.feg_warmup_epochs
        else:
            # Keep constant or decay exponentially after warmup (exponential decay chosen here to focus more on clustering loss)
            decay_factor = 0.95 ** (self.current_epoch - self.feg_warmup_epochs)
            return self.feg_weight * decay_factor

    def training_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        
        # Normalize cluster centers (Project to Sphere)
        with torch.no_grad():
            self.clusters.data = F.normalize(self.clusters.data, dim=-1)
        
        B, T, _ = features_raw.shape
        
        # === 1. Feature Extraction and Enhancement (FEG Step) ===
        # Get normalized features Z and FEG Triplet Loss
        features, feg_loss = self.feg_module(features_raw, mask, return_loss=True)
        
        # === 2. ASOT Pseudo-label Generation (OT Step) ===
        # Calculate similarity probability P(z|c) between features and cluster centers
        codes = torch.exp(features @ self.clusters.T[None, ...] / self.temp)
        codes = codes / codes.sum(dim=-1, keepdim=True)
        
        with torch.no_grad():
            # Build temporal prior matrix
            temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
            
            # Build base cost matrix (1 - Cosine Similarity)
            cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
            cost_matrix += temp_prior
            
            # === [Core Modification] Warmup Strategy ===
            # If in warmup period (feg_warmup_epochs), force pass None.
            # This makes asot.py fallback to old mode (fixed temporal constraint), ensuring stability of initial pseudo-labels.
            # Only enable "Feature-Induced OT" after warmup ends.
            if self.current_epoch < self.feg_warmup_epochs:
                use_features = None
                use_clusters = None
            else:
                use_features = features
                use_clusters = self.clusters

            # Solve Optimal Transport (OT)
            opt_codes, _ = asot.segment_asot(
                cost_matrix, mask,
                features=use_features,      # <--- Dynamically passed (may be None)
                clusters=use_clusters,      # <--- Dynamically passed (may be None)
                feg_h=self.hparams.feg_h_param,# <--- Pass FEG h parameter
                eps=self.train_eps, alpha=self.alpha_train, 
                radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions, 
                lambda_frames=self.lambda_frames_train, lambda_actions=self.lambda_actions_train, 
                n_iters=self.n_ot_train, step_size=self.step_size
            )

        # === 3. Cross Entropy Loss (Self-Training Step) ===
        loss_ce = -((opt_codes * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        
        # === 4. Total Loss Fusion ===
        current_feg_weight = self.get_feg_weight_schedule()
        total_loss = loss_ce + current_feg_weight * feg_loss
        
        # === 5. Logging ===
        self.log('train_loss', total_loss)
        self.log('train_loss_ce', loss_ce)
        self.log('train_loss_feg', feg_loss)
        self.log('feg_weight', current_feg_weight)
        self.log('feg_alpha', torch.sigmoid(self.feg_module.alpha_logit))
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        features_raw, mask, gt, fname, n_subactions = batch
        B, T, _ = features_raw.shape
        D = self.layer_sizes[-1]
        
        # === Feature Extraction (No Loss Calculation) ===
        features, _ = self.feg_module(features_raw, mask, return_loss=False)
        # features are already L2 Normalized

        # === ASOT Inference (Eval Mode) ===
        temp_prior = asot.temporal_prior(T, self.n_clusters, self.rho, features.device)
        cost_matrix = 1. - features @ self.clusters.T.unsqueeze(0)
        cost_matrix += temp_prior
        
        # Segmentation using Eval parameters
        segmentation, _ = asot.segment_asot(
            cost_matrix, mask,
            features=features,          # <--- New
            clusters=self.clusters,     # <--- New
            feg_h=self.hparams.feg_h_param, # <--- Pass FEG h parameter
            eps=self.eval_eps, alpha=self.alpha_eval, 
            radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions, 
            lambda_frames=self.lambda_frames_eval, lambda_actions=self.lambda_actions_eval, 
            n_iters=self.n_ot_eval, step_size=self.step_size
        )
        
        segments = segmentation.argmax(dim=2)
        
        # === Update Evaluation Metrics ===
        self.mof.update(segments, gt, mask)
        self.f1.update(segments, gt, mask)
        self.miou.update(segments, gt, mask)
        
        # Calculate metrics for single video
        metrics = indep_eval_metrics(
            segments, gt, mask, ['mof', 'f1', 'miou'], 
            exclude_cls=self.exclude_cls
        )
        self.log('val_mof_per', metrics['mof'])
        self.log('val_f1_per', metrics['f1'])
        self.log('val_miou_per', metrics['miou'])

        # === Calculate Validation Loss (For Monitoring Only) ===
        codes = torch.exp(features @ self.clusters.T / self.temp)
        codes /= codes.sum(dim=-1, keepdim=True)
        
        # Generate pseudo-labels using Train parameters for Loss calculation (Consistency)
        pseudo_labels, _ = asot.segment_asot(
            cost_matrix, mask, eps=self.train_eps, alpha=self.alpha_train, 
            radius=self.radius_gw, ub_frames=self.ub_frames, ub_actions=self.ub_actions, 
            lambda_frames=self.lambda_frames_train, lambda_actions=self.lambda_actions_train, 
            n_iters=self.n_ot_train, step_size=self.step_size
        )
        
        loss_val_ce = -((pseudo_labels * torch.log(codes + num_eps)) * mask[..., None]).sum(dim=2).mean()
        self.log('val_loss', loss_val_ce)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def get_args():
        parser = argparse.ArgumentParser()
        
        # Dataset setup
        parser.add_argument("-p", "--path", type=str, required=True, help="Path to data directory")
        parser.add_argument("-d", "--dataset", type=str, required=True, choices=["Breakfast", "FS", "FSeval", "desktop_assembly"])
        parser.add_argument("-ac", "--activity", type=str, default="all", help="Specific activity class (e.g. 'coffee')")
        
        # Cluster setup
        parser.add_argument("-c", "--clusters", type=int, default=20, help="Number of clusters (sub-actions)")
        
        # Model hyperparameters
        parser.add_argument("-ne", "--num-epochs", type=int, default=50)
        parser.add_argument("-bs", "--batch-size", type=int, default=1)
        parser.add_argument("-lr", "--learning-rate", type=float, default=1e-4) # Fixed: argparse uses - for flags, python calls replace with _
        parser.add_argument("-wd", "--weight-decay", type=float, default=1e-4)
        parser.add_argument("-vf", "--val-freq", type=int, default=5)
        parser.add_argument("-ls", "--layer-sizes", type=int, nargs="+", default=[64, 128, 40], help="Layer sizes for MLP")
        
        # FEG Parameters
        parser.add_argument("--feg-weight", type=float, default=0.3, help="Weight for FEG Tripel Loss")
        parser.add_argument("--feg-warmup", type=int, default=5, dest="feg_warmup_epochs", help="Warmup epochs for FEG")
        parser.add_argument("--feg-h", type=float, default=0.1, dest="feg_h_param", help="Gaussian kernel bandwidth (h) for semantic similarity")
        
        # ASOT parameters
        parser.add_argument("-g", "--gpu", type=int, default=0)
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--group", type=str, default="default_group")
        parser.add_argument("-wandb", "--wandb", action="store_true")
        parser.add_argument("-v", "--verbose", action="store_true")

        # ASOT core parameters
        parser.add_argument("-ua", "--ub-actions", action="store_true", help="Unbalanced OT for actions")
        parser.add_argument("-uf", "--ub-frames", action="store_true", help="Unbalanced OT for frames")
        parser.add_argument("--rho", type=float, default=0.2, help="Temporal prior strength")
        parser.add_argument("-lat", "--lambda-actions-train", type=float, default=0.05)
        parser.add_argument("-lft", "--lambda-frames-train", type=float, default=0.05)
        parser.add_argument("-r", "--radius-gw", type=float, default=0.04)
        
        # Extra settings
        parser.add_argument("-km", "--use-kmeans-init", action="store_true", help="Use K-Means to initialize clusters")

        return parser.parse_args()

def main():
    args = FEG_ASOT.get_args()
    
    # Random Seed
    pl.seed_everything(args.seed)
    
    # Dataset Setup
    dataset = VideoDataset(
        args.path, 
        args.dataset, 
        args.activity, 
        n_frames=256 # Default sampling frames
    )
    
    # DataLoader
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda x: dataset.collate_fn(x)
    )
    
    # WandB
    if args.wandb:
        wandb.init(project="FIS-OT-FEG", group=args.group, config=args)
    
    # Create Model
    # Important: Ensure Layer Sizes match Feature Dimension
    # Need to verify input dimension from dataset
    input_dim = dataset.get_feature_dim()
    
    # If using default parameters, modify first layer to match input_dim
    layer_sizes = args.layer_sizes
    if layer_sizes[0] != input_dim:
        layer_sizes[0] = input_dim
        print(f"Adjusted input layer size to {input_dim}")

    model = FEG_ASOT(
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        layer_sizes=layer_sizes,
        n_clusters=args.clusters,
        
        ub_actions=args.ub_actions,
        ub_frames=args.ub_frames,
        lambda_actions_train=args.lambda_actions_train,
        lambda_frames_train=args.lambda_frames_train,
        radius_gw=args.radius_gw,
        rho=args.rho,
        
        feg_weight=args.feg_weight,
        feg_warmup_epochs=args.feg_warmup_epochs,
        feg_h_param=args.feg_h_param
    )
    
    # (Optional) K-Means Initialization
    if args.use_kmeans_init:
        print("Initializing clusters with K-Means...")
        # Gather all features from dataset (random sampling to save memory)
        all_feats = []
        for i in range(len(dataset)):
            feat, _, _, _, _ = dataset[i]
            # Randomly sample some frames
            if feat.shape[0] > 100:
                idx = np.random.choice(feat.shape[0], 100, replace=False)
                feat = feat[idx]
            all_feats.append(feat)
        all_feats = np.concatenate(all_feats, axis=0) # [N, D]
        
        kmeans = KMeans(n_clusters=args.clusters, n_init=10, random_state=args.seed)
        kmeans.fit(all_feats)
        center_init = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        
        with torch.no_grad():
            model.clusters.data = F.normalize(center_init, dim=-1)
            
        print("K-Means Initialization Completed.")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus=[args.gpu] if torch.cuda.is_available() else 0, # Use specific GPU
        check_val_every_n_epoch=args.val_freq,
        logger=pl.loggers.WandbLogger() if args.wandb else None,
        log_every_n_steps=5
    )
    
    trainer.fit(model, train_loader)
    
    # Final Validation
    print("Training Finished. Running Final Validation...")
    results = trainer.validate(model, train_loader)
    print(results)

if __name__ == "__main__":
    main()
