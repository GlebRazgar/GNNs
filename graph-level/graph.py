"""
Weather Prediction GNN - Node-level Precipitation Forecasting
Data Pipeline Implementation
"""

import numpy as np
import xarray as xr
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import os

# ============================================================================
# PART 1: DATA LOADING
# ============================================================================

def load_weather_data(zarr_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load weather data from zarr file and reshape for GNN processing.
    
    Args:
        zarr_path: Path to zarr dataset
        
    Returns:
        features: [T, N, F] where T=timesteps, N=nodes, F=features
        target: [T, N, 1] precipitation target
        lat: [32] latitude values
        lon: [64] longitude values
    """
    print("=" * 70)
    print("LOADING WEATHER DATA")
    print("=" * 70)
    
    # Load dataset
    ds = xr.open_zarr(zarr_path)
    print(f"✓ Loaded dataset from {zarr_path}")
    
    # Get dimensions
    n_time = len(ds.time)
    n_lat = len(ds.latitude)
    n_lon = len(ds.longitude)
    n_nodes = n_lat * n_lon
    
    print(f"\nDimensions:")
    print(f"  • Timesteps: {n_time}")
    print(f"  • Latitude: {n_lat}")
    print(f"  • Longitude: {n_lon}")
    print(f"  • Total nodes: {n_nodes}")
    
    # Extract feature arrays [time, lon, lat]
    t2m = ds['t2m'].values      # temperature
    q850 = ds['q850'].values    # specific humidity
    rh = ds['rh'].values        # relative humidity
    z_sfc = ds['z_sfc'].values  # surface geopotential (no time dim)
    
    # Extract target [time, lon, lat]
    tp6h = ds['tp6h'].values    # precipitation
    
    # Get coordinates
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    print(f"\nFeature shapes:")
    print(f"  • t2m:  {t2m.shape}")
    print(f"  • q850: {q850.shape}")
    print(f"  • rh:   {rh.shape}")
    print(f"  • z_sfc: {z_sfc.shape} (static)")
    print(f"  • tp6h: {tp6h.shape} (target)")
    
    # Reshape from [T, Lon, Lat] to [T, N, 1] where N = Lon*Lat
    # We need to flatten the spatial dimensions
    t2m_flat = t2m.reshape(n_time, -1, 1)      # [T, N, 1]
    q850_flat = q850.reshape(n_time, -1, 1)
    rh_flat = rh.reshape(n_time, -1, 1)
    
    # z_sfc is static, broadcast to all timesteps
    z_sfc_flat = np.tile(z_sfc.reshape(1, -1, 1), (n_time, 1, 1))  # [T, N, 1]
    
    # Target
    tp6h_flat = tp6h.reshape(n_time, -1, 1)    # [T, N, 1]
    
    # Stack features along last dimension: [T, N, F]
    features = np.concatenate([t2m_flat, q850_flat, rh_flat, z_sfc_flat], axis=-1)
    
    print(f"\nReshaped data:")
    print(f"  • Features: {features.shape} [timesteps, nodes, features]")
    print(f"  • Target:   {tp6h_flat.shape} [timesteps, nodes, 1]")
    
    # Check for NaN or inf
    print(f"\nData quality check:")
    print(f"  • NaN in features: {np.isnan(features).sum()}")
    print(f"  • NaN in target: {np.isnan(tp6h_flat).sum()}")
    print(f"  • Inf in features: {np.isinf(features).sum()}")
    print(f"  • Inf in target: {np.isinf(tp6h_flat).sum()}")
    
    print(f"\n✓ Data loading complete!")
    print("=" * 70)
    
    return features, tp6h_flat, lat, lon


# ============================================================================
# PART 2: GRAPH CONSTRUCTION
# ============================================================================

def create_grid_edges(n_lat: int, n_lon: int, wraparound: bool = True) -> torch.Tensor:
    """
    Create edge index for grid-based 4-neighbor connectivity.
    
    Args:
        n_lat: Number of latitude points (32)
        n_lon: Number of longitude points (64)
        wraparound: Whether longitude wraps around (360° → 0°)
        
    Returns:
        edge_index: [2, E] tensor of edges
    """
    print("\n" + "=" * 70)
    print("CREATING GRAPH STRUCTURE")
    print("=" * 70)
    
    edges = []
    
    # Helper to convert (lat_idx, lon_idx) to node_idx
    def to_node_idx(lat, lon):
        return lat * n_lon + lon
    
    # For each grid cell
    for lat in range(n_lat):
        for lon in range(n_lon):
            current_node = to_node_idx(lat, lon)
            
            # North neighbor (lat + 1)
            if lat < n_lat - 1:
                neighbor = to_node_idx(lat + 1, lon)
                edges.append([current_node, neighbor])
                edges.append([neighbor, current_node])  # Undirected
            
            # South neighbor (lat - 1)
            if lat > 0:
                neighbor = to_node_idx(lat - 1, lon)
                edges.append([current_node, neighbor])
                edges.append([neighbor, current_node])
            
            # East neighbor (lon + 1)
            if lon < n_lon - 1:
                neighbor = to_node_idx(lat, lon + 1)
                edges.append([current_node, neighbor])
                edges.append([neighbor, current_node])
            elif wraparound:
                # Wrap around: rightmost connects to leftmost
                neighbor = to_node_idx(lat, 0)
                edges.append([current_node, neighbor])
                edges.append([neighbor, current_node])
            
            # West neighbor (lon - 1)
            if lon > 0:
                neighbor = to_node_idx(lat, lon - 1)
                edges.append([current_node, neighbor])
                edges.append([neighbor, current_node])
            elif wraparound:
                # Wrap around: leftmost connects to rightmost
                neighbor = to_node_idx(lat, n_lon - 1)
                edges.append([current_node, neighbor])
                edges.append([neighbor, current_node])
    
    # Remove duplicate edges
    edges = list(set(map(tuple, edges)))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    n_nodes = n_lat * n_lon
    n_edges = edge_index.shape[1]
    avg_degree = n_edges / n_nodes
    
    print(f"\nGraph statistics:")
    print(f"  • Nodes: {n_nodes}")
    print(f"  • Edges: {n_edges}")
    print(f"  • Average degree: {avg_degree:.2f}")
    print(f"  • Wraparound: {wraparound}")
    
    print(f"\n✓ Graph structure created!")
    print("=" * 70)
    
    return edge_index


# ============================================================================
# PART 3: FEATURE NORMALIZATION
# ============================================================================

def normalize_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_target: np.ndarray,
    val_target: np.ndarray,
    test_target: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Normalize features and target using StandardScaler.
    Fit on training data only to prevent data leakage.
    
    Returns:
        Normalized train/val/test features and targets, plus scalers dict
    """
    print("\n" + "=" * 70)
    print("NORMALIZING FEATURES")
    print("=" * 70)
    
    # Get shapes
    n_train, n_nodes, n_features = train_features.shape
    
    # Reshape for sklearn: [samples, features]
    # Combine time and node dimensions
    train_flat = train_features.reshape(-1, n_features)
    val_flat = val_features.reshape(-1, n_features)
    test_flat = test_features.reshape(-1, n_features)
    
    # Fit scaler on training data only
    feature_scaler = StandardScaler()
    train_flat_norm = feature_scaler.fit_transform(train_flat)
    val_flat_norm = feature_scaler.transform(val_flat)
    test_flat_norm = feature_scaler.transform(test_flat)
    
    # Reshape back
    train_features_norm = train_flat_norm.reshape(train_features.shape)
    val_features_norm = val_flat_norm.reshape(val_features.shape)
    test_features_norm = test_flat_norm.reshape(test_features.shape)
    
    print(f"\nFeature normalization:")
    print(f"  • Mean: {feature_scaler.mean_}")
    print(f"  • Std:  {feature_scaler.scale_}")
    
    # Normalize target (precipitation)
    # Note: Precipitation is non-negative, might want special handling
    train_target_flat = train_target.reshape(-1, 1)
    val_target_flat = val_target.reshape(-1, 1)
    test_target_flat = test_target.reshape(-1, 1)
    
    target_scaler = StandardScaler()
    train_target_norm = target_scaler.fit_transform(train_target_flat).reshape(train_target.shape)
    val_target_norm = target_scaler.transform(val_target_flat).reshape(val_target.shape)
    test_target_norm = target_scaler.transform(test_target_flat).reshape(test_target.shape)
    
    print(f"\nTarget normalization:")
    print(f"  • Mean: {target_scaler.mean_[0]:.6f}")
    print(f"  • Std:  {target_scaler.scale_[0]:.6f}")
    
    # Store scalers for later use
    scalers = {
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler
    }
    
    print(f"\n✓ Normalization complete!")
    print("=" * 70)
    
    return (train_features_norm, val_features_norm, test_features_norm,
            train_target_norm, val_target_norm, test_target_norm, scalers)


# ============================================================================
# PART 4: DATA SPLITTING
# ============================================================================

def create_temporal_splits(
    features: np.ndarray,
    target: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, ...]:
    """
    Split data temporally (respecting time order).
    
    Args:
        features: [T, N, F] feature array
        target: [T, N, 1] target array
        
    Returns:
        train/val/test splits for features and targets
    """
    print("\n" + "=" * 70)
    print("CREATING TEMPORAL SPLITS")
    print("=" * 70)
    
    n_timesteps = features.shape[0]
    
    # Calculate split indices
    train_end = int(n_timesteps * train_ratio)
    val_end = train_end + int(n_timesteps * val_ratio)
    
    # Split data
    train_features = features[:train_end]
    train_target = target[:train_end]
    
    val_features = features[train_end:val_end]
    val_target = target[train_end:val_end]
    
    test_features = features[val_end:]
    test_target = target[val_end:]
    
    print(f"\nSplit sizes:")
    print(f"  • Train: {train_features.shape[0]} timesteps ({train_ratio*100:.0f}%)")
    print(f"  • Val:   {val_features.shape[0]} timesteps ({val_ratio*100:.0f}%)")
    print(f"  • Test:  {test_features.shape[0]} timesteps ({test_ratio*100:.0f}%)")
    
    print(f"\n✓ Temporal splits created!")
    print("=" * 70)
    
    return (train_features, train_target,
            val_features, val_target,
            test_features, test_target)


# ============================================================================
# PART 5: GNN MODEL ARCHITECTURE
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm


class WeatherGNN(nn.Module):
    """
    Graph Neural Network for weather prediction.
    
    Architecture:
        Input [N, F] → Encoder → GNN Layers → Decoder → Output [N, 1]
    
    Args:
        in_channels: Number of input features (4)
        hidden_dim: Hidden dimension size (default: 64)
        num_layers: Number of GNN layers (default: 2)
        dropout: Dropout probability (default: 0.2)
        conv_type: Type of GNN layer ('GCN' or 'GAT')
        use_skip: Whether to use skip connections (default: True)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = 'GCN',
        use_skip: bool = True
    ):
        super(WeatherGNN, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.conv_type = conv_type
        self.use_skip = use_skip
        
        # Feature encoder: map input features to hidden dimension
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # Create GNN layer based on type
            if conv_type == 'GCN':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == 'GAT':
                conv = GATConv(hidden_dim, hidden_dim, heads=1)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Skip connection projection (if dimensions don't match)
        if use_skip:
            self.skip_proj = nn.Linear(in_channels, hidden_dim)
        
        # Output decoder: map hidden to prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            
        Returns:
            predictions: Precipitation predictions [N, 1]
        """
        # Store original input for skip connection
        x_input = x
        
        # Encode input features
        x = self.encoder(x)
        
        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_prev = x
            
            # Message passing
            x = conv(x, edge_index)
            
            # Batch normalization
            x = bn(x)
            
            # Activation
            x = F.relu(x)
            
            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Skip connection (residual)
            if self.use_skip and i > 0:
                x = x + x_prev
        
        # Add skip connection from input
        if self.use_skip:
            x_skip = self.skip_proj(x_input)
            x = x + x_skip
        
        # Decode to prediction
        out = self.decoder(x)
        
        return out
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# PART 6: PYTORCH GEOMETRIC DATASET WRAPPER
# ============================================================================

from torch.utils.data import Dataset as TorchDataset

class WeatherGraphDataset(TorchDataset):
    """
    PyTorch Dataset wrapper for weather graph data.
    Returns PyTorch Geometric Data objects.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        edge_index: torch.Tensor
    ):
        """
        Args:
            features: [T, N, F] feature array
            targets: [T, N, 1] target array
            edge_index: [2, E] edge connectivity (shared across timesteps)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.edge_index = edge_index
        self.n_timesteps = features.shape[0]
    
    def __len__(self) -> int:
        return self.n_timesteps
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get Data object for a single timestep.
        
        Returns:
            Data(x=[N, F], y=[N, 1], edge_index=[2, E])
        """
        data = Data(
            x=self.features[idx],
            y=self.targets[idx],
            edge_index=self.edge_index
        )
        return data


# ============================================================================
# PART 7: TRAINING PIPELINE
# ============================================================================

from torch_geometric.loader import DataLoader  # Use PyG DataLoader for graph batching
from tqdm import tqdm
import time


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average training loss
        avg_rmse: Average RMSE
    """
    model.train()
    total_loss = 0
    total_rmse = 0
    n_batches = 0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # Forward pass
        pred = model(batch.x, batch.edge_index)
        loss = criterion(pred, batch.y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        rmse = torch.sqrt(F.mse_loss(pred, batch.y))
        total_rmse += rmse.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_rmse = total_rmse / n_batches
    
    return avg_loss, avg_rmse


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Validate the model.
    
    Returns:
        avg_loss: Average validation loss
        avg_rmse: Average RMSE
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch.x, batch.edge_index)
            loss = criterion(pred, batch.y)
            
            # Metrics
            total_loss += loss.item()
            rmse = torch.sqrt(F.mse_loss(pred, batch.y))
            total_rmse += rmse.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_rmse = total_rmse / n_batches
    
    return avg_loss, avg_rmse


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int = 100,
    early_stop_patience: int = 15,
    device: str = 'cpu',
    save_path: str = 'best_weather_gnn.pt'
) -> Dict:
    """
    Train the model with early stopping and progress tracking.
    
    Returns:
        history: Dictionary with training history
    """
    print("\n" + "=" * 70)
    print("TRAINING WEATHER GNN")
    print("=" * 70)
    
    # Training history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': [],
        'epoch_times': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nTraining configuration:")
    print(f"  • Device: {device}")
    print(f"  • Epochs: {num_epochs}")
    print(f"  • Batch size: {train_loader.batch_size}")
    print(f"  • Train batches: {len(train_loader)}")
    print(f"  • Val batches: {len(val_loader)}")
    print(f"  • Optimizer: {optimizer.__class__.__name__}")
    print(f"  • Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  • Early stop patience: {early_stop_patience}")
    print(f"  • Model parameters: {model.count_parameters():,}")
    
    print("\n" + "-" * 70)
    print("Starting training...")
    print("-" * 70)
    
    # Main training loop with progress bar
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase with progress bar
        model.train()
        train_loss_accum = 0
        train_rmse_accum = 0
        
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            ncols=100,
            leave=False
        )
        
        for batch in train_pbar:
            batch = batch.to(device)
            
            # Forward pass
            pred = model(batch.x, batch.edge_index)
            loss = criterion(pred, batch.y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss_accum += loss.item()
            rmse = torch.sqrt(F.mse_loss(pred, batch.y))
            train_rmse_accum += rmse.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'rmse': f'{rmse.item():.4f}'
            })
        
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_rmse = train_rmse_accum / len(train_loader)
        
        # Validation phase with progress bar
        model.eval()
        val_loss_accum = 0
        val_rmse_accum = 0
        
        val_pbar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ",
            ncols=100,
            leave=False
        )
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                
                # Forward pass
                pred = model(batch.x, batch.edge_index)
                loss = criterion(pred, batch.y)
                
                # Metrics
                val_loss_accum += loss.item()
                rmse = torch.sqrt(F.mse_loss(pred, batch.y))
                val_rmse_accum += rmse.item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'rmse': f'{rmse.item():.4f}'
                })
        
        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_rmse = val_rmse_accum / len(val_loader)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_rmse'].append(avg_train_rmse)
        history['val_loss'].append(avg_val_loss)
        history['val_rmse'].append(avg_val_rmse)
        history['epoch_times'].append(epoch_time)
        
        # Calculate time estimates
        avg_epoch_time = np.mean(history['epoch_times'])
        remaining_epochs = num_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        eta_min = int(eta // 60)
        eta_sec = int(eta % 60)
        
        # Print epoch summary
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"ETA: {eta_min}m {eta_sec}s | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"Train RMSE: {avg_train_rmse:.6f} | "
              f"Val RMSE: {avg_val_rmse:.6f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'history': history
            }, save_path)
            print(f"  → New best model saved! (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best validation loss: {best_val_loss:.6f}")
                break
    
    # Training complete
    total_time = sum(history['epoch_times'])
    print("\n" + "-" * 70)
    print("Training complete!")
    print(f"  • Total time: {total_time/60:.1f} minutes")
    print(f"  • Best val loss: {best_val_loss:.6f}")
    print(f"  • Model saved to: {save_path}")
    print("=" * 70)
    
    return history


# ============================================================================
# TESTING: Model architecture
# ============================================================================

def test_model():
    """Test model forward pass."""
    print("\n" + "=" * 70)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 70)
    
    # Create dummy data
    n_nodes = 2048
    n_features = 4
    n_edges = 8064
    
    # Dummy input
    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    
    # Create model
    model = WeatherGNN(
        in_channels=4,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        conv_type='GCN',
        use_skip=True
    )
    
    print(f"\nModel architecture:")
    print(f"  • Input features: {model.in_channels}")
    print(f"  • Hidden dimension: {model.hidden_dim}")
    print(f"  • Number of GNN layers: {model.num_layers}")
    print(f"  • Dropout: {model.dropout}")
    print(f"  • Conv type: {model.conv_type}")
    print(f"  • Skip connections: {model.use_skip}")
    print(f"  • Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
    
    print(f"\nForward pass test:")
    print(f"  • Input shape: {x.shape}")
    print(f"  • Edge index shape: {edge_index.shape}")
    print(f"  • Output shape: {out.shape}")
    print(f"  • Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    print(f"\n✓ Model architecture test passed!")
    print("=" * 70)
    
    return model


# ============================================================================
# TESTING: Full pipeline test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("WEATHER GNN FULL PIPELINE TEST")
    print("=" * 70)
    
    # ========================================================================
    # PART 1: DATA PIPELINE
    # ========================================================================
    
    # Path to data
    zarr_path = "wb2_64x32_weather_2020-2023.zarr"
    
    # Step 1: Load data
    features, target, lat, lon = load_weather_data(zarr_path)
    
    # Step 2: Create graph structure
    n_lat, n_lon = len(lat), len(lon)
    edge_index = create_grid_edges(n_lat, n_lon, wraparound=True)
    
    # Step 3: Create temporal splits
    splits = create_temporal_splits(features, target)
    train_feat, train_tgt, val_feat, val_tgt, test_feat, test_tgt = splits
    
    # Step 4: Normalize features
    normalized = normalize_features(
        train_feat, val_feat, test_feat,
        train_tgt, val_tgt, test_tgt
    )
    train_feat_norm, val_feat_norm, test_feat_norm = normalized[:3]
    train_tgt_norm, val_tgt_norm, test_tgt_norm = normalized[3:6]
    scalers = normalized[6]
    
    print("\n" + "=" * 70)
    print("DATA PIPELINE ✓")
    print("=" * 70)
    
    # ========================================================================
    # PART 2: MODEL ARCHITECTURE
    # ========================================================================
    
    # Test model with dummy data first
    model = test_model()
    
    # ========================================================================
    # PART 3: TEST WITH REAL DATA
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("TESTING MODEL WITH REAL DATA")
    print("=" * 70)
    
    # Take first timestep from training data
    x_real = torch.tensor(train_feat_norm[0], dtype=torch.float32)
    y_real = torch.tensor(train_tgt_norm[0], dtype=torch.float32)
    
    print(f"\nReal data shapes:")
    print(f"  • Features: {x_real.shape}")
    print(f"  • Target: {y_real.shape}")
    print(f"  • Edges: {edge_index.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(x_real, edge_index)
    
    print(f"\nPrediction:")
    print(f"  • Shape: {pred.shape}")
    print(f"  • Range: [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"  • Mean: {pred.mean():.4f}")
    
    # Compute simple MSE
    mse = F.mse_loss(pred, y_real)
    print(f"\nInitial MSE (untrained): {mse.item():.6f}")
    
    print(f"\n✓ Model works with real data!")
    
    # ========================================================================
    # PART 4: CREATE DATASETS
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("CREATING PYTORCH DATASETS")
    print("=" * 70)
    
    train_dataset = WeatherGraphDataset(train_feat_norm, train_tgt_norm, edge_index)
    val_dataset = WeatherGraphDataset(val_feat_norm, val_tgt_norm, edge_index)
    test_dataset = WeatherGraphDataset(test_feat_norm, test_tgt_norm, edge_index)
    
    print(f"\nDataset sizes:")
    print(f"  • Train: {len(train_dataset)} graphs")
    print(f"  • Val: {len(val_dataset)} graphs")
    print(f"  • Test: {len(test_dataset)} graphs")
    
    # Test dataset access
    sample = train_dataset[0]
    print(f"\nSample data object:")
    print(f"  • x shape: {sample.x.shape}")
    print(f"  • y shape: {sample.y.shape}")
    print(f"  • edge_index shape: {sample.edge_index.shape}")
    
    print(f"\n✓ Datasets created successfully!")
    
    # ========================================================================
    # PART 5: TRAIN THE MODEL
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("PREPARING FOR TRAINING")
    print("=" * 70)
    
    # Create DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\nDataLoaders created:")
    print(f"  • Train: {len(train_loader)} batches")
    print(f"  • Val: {len(val_loader)} batches")
    print(f"  • Test: {len(test_loader)} batches")
    
    # Create fresh model for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WeatherGNN(
        in_channels=4,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        conv_type='GCN',
        use_skip=True
    ).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"\nTraining setup:")
    print(f"  • Device: {device}")
    print(f"  • Optimizer: Adam (lr=0.001, weight_decay=1e-5)")
    print(f"  • Loss: MSE")
    print(f"  • Model params: {model.count_parameters():,}")
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=50,
        early_stop_patience=15,
        device=device,
        save_path='best_weather_gnn.pt'
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETE ✓")
    print("=" * 70)
    print(f"\n🎉 All stages completed:")
    print(f"  ✓ Data loading & preprocessing")
    print(f"  ✓ Graph construction (4-neighbor grid)")
    print(f"  ✓ Feature normalization")
    print(f"  ✓ Temporal train/val/test splits")
    print(f"  ✓ GNN model architecture ({model.count_parameters():,} params)")
    print(f"  ✓ Model training with early stopping")
    print(f"  ✓ Best model saved")
    
    print(f"\n📊 Final results:")
    print(f"  • Epochs trained: {len(history['train_loss'])}")
    print(f"  • Best val loss: {min(history['val_loss']):.6f}")
    print(f"  • Best val RMSE: {min(history['val_rmse']):.6f}")
    print(f"  • Total training time: {sum(history['epoch_times'])/60:.1f} minutes")
    print(f"  • Model saved to: best_weather_gnn.pt")
    print("\n" + "=" * 70)

