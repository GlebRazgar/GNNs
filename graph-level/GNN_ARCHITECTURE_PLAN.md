# GNN Architecture Plan: Weather Precipitation Prediction
## Node-Level Regression Task

---

## 1. PROBLEM DEFINITION

### Task Type
**Node-level regression**: Predict precipitation (tp6h) at each grid cell given weather features

### Input
- **Node features** (4 per node):
  - `t2m`: 2-meter temperature (Kelvin)
  - `q850`: Specific humidity at 850 hPa (kg/kg)
  - `rh`: Relative humidity at 850 hPa (fraction)
  - `z_sfc`: Surface geopotential/elevation (m²/s²) - static

### Output
- **Target** (1 per node): `tp6h` - 6-hour total precipitation (m)

### Graph Structure
- **Nodes**: 2,048 grid cells (64 longitude × 32 latitude)
- **Edges**: Spatial connections between neighboring grid cells
- **Samples**: 4,424 timesteps (independent graph snapshots)

---

## 2. DATA PIPELINE ARCHITECTURE

### 2.1 Data Loading & Preprocessing
```
Input: zarr dataset (wb2_64x32_weather_2020-2023.zarr)
  ↓
Extract arrays:
  - features: [T, Lon, Lat, F] where T=4424, Lon=64, Lat=32, F=4
  - target: [T, Lon, Lat, 1]
  - coordinates: lat[32], lon[64]
  ↓
Reshape to graph format:
  - features: [T, N, F] where N=2048 nodes
  - target: [T, N, 1]
  - node_positions: [N, 2] (lat/lon coordinates)
```

### 2.2 Feature Engineering
```
For each feature:
  1. Handle missing values (if any)
  2. Normalize using StandardScaler or MinMaxScaler
     - Fit on training set only
     - Transform train/val/test separately
  3. Store normalization parameters for inference
```

**Special considerations**:
- `z_sfc` is static (no time dimension) - broadcast to all timesteps
- Precipitation is non-negative - consider log transform or clip
- Temperature in Kelvin - may convert to Celsius for interpretability

### 2.3 Graph Construction

**Option A: Grid-based connectivity (4-neighbors)**
```
Each interior node connects to:
  - North neighbor (lat + 1)
  - South neighbor (lat - 1)
  - East neighbor (lon + 1, with wraparound for global data)
  - West neighbor (lon - 1, with wraparound)

Edge cases:
  - Poles: connect only to available neighbors
  - Longitude wraparound: 360° → 0° (continuous)
```

**Option B: K-nearest neighbors (KNN) in lat/lon space**
```
For each node:
  - Compute haversine distance to all other nodes
  - Connect to k=8 nearest neighbors
  - Results in more connections for learning
```

**Recommendation**: Start with Option A (grid-based), then try Option B

**Edge attributes** (optional enhancement):
```
- Distance between nodes (haversine)
- Direction vector (lat_diff, lon_diff)
- Elevation difference (for topographic effects)
```

### 2.4 Data Splitting Strategy

**Temporal split** (respect time ordering):
```
Train:      70% → timesteps [0:3097]        (2020-01-01 to 2022-02-18)
Validation: 15% → timesteps [3097:3760]     (2022-02-18 to 2022-08-15)
Test:       15% → timesteps [3760:4424]     (2022-08-15 to 2023-01-10)
```

**Why temporal split?**
- Tests model's ability to generalize to future unseen weather patterns
- Prevents data leakage from adjacent timesteps

### 2.5 PyTorch Geometric Data Structure
```python
Data object per timestep:
  - data.x: [2048, 4] node features
  - data.y: [2048, 1] target precipitation
  - data.edge_index: [2, E] edge connections (E = ~8192 for 4-neighbors)
  - data.edge_attr: [E, d] edge features (optional)
  - data.pos: [2048, 2] node positions (lat/lon)
```

---

## 3. GNN MODEL ARCHITECTURE

### 3.1 Overall Structure
```
Input Features [N, 4]
    ↓
Feature Encoder (Linear/MLP)
    ↓
GNN Layer 1 (Message Passing)
    ↓
Activation + Normalization + Dropout
    ↓
GNN Layer 2 (Message Passing)
    ↓
Activation + Normalization + Dropout
    ↓
[Optional] GNN Layer 3
    ↓
[Optional] Skip Connections
    ↓
Output Decoder (MLP)
    ↓
Prediction [N, 1]
```

### 3.2 Component Details

#### A. Feature Encoder
```
Purpose: Transform raw features to hidden representation
Architecture:
  - Linear(in=4, out=hidden_dim)
  - Optional: MLP with 2 layers for richer encoding
  
Hyperparameters:
  - hidden_dim: 64, 128, or 256 (start with 64)
```

#### B. GNN Layers (Message Passing)

**Option 1: GCNConv (Graph Convolutional Network)**
```python
Pros:
  - Simple, efficient
  - Good for regular grid structures
  - Proven for spatial data

Formula: h_i' = Σ_{j∈N(i)} (1/√(d_i·d_j)) · W · h_j
```

**Option 2: GATConv (Graph Attention Network)**
```python
Pros:
  - Learns importance of neighbors
  - Better for irregular patterns
  - More expressive

Formula: h_i' = Σ_{j∈N(i)} α_ij · W · h_j
  where α_ij = attention coefficient (learned)
```

**Option 3: GraphSAGE**
```python
Pros:
  - Sampling-based (scalable)
  - Good aggregation strategies
  
Formula: h_i' = σ(W · [h_i || AGG({h_j : j∈N(i)})])
```

**Recommendation**: Start with GCNConv, then experiment with GATConv

**Layer Configuration**:
```
Layer 1: hidden_dim → hidden_dim
Layer 2: hidden_dim → hidden_dim
Layer 3 (optional): hidden_dim → hidden_dim

Each layer followed by:
  - BatchNorm1d(hidden_dim)
  - ReLU() or LeakyReLU()
  - Dropout(p=0.1 to 0.3)
```

#### C. Skip Connections (Residual)
```
Purpose: Preserve information flow, prevent vanishing gradients

Implementation:
  x_out = GNN_layer(x_in) + x_in  # Residual
  or
  x_out = GNN_layer(x_in) + W_skip @ x_in  # Learnable skip
```

#### D. Output Decoder
```
Purpose: Map hidden representation to precipitation prediction

Architecture:
  - Option 1 (Simple): Linear(hidden_dim → 1)
  - Option 2 (MLP): 
      Linear(hidden_dim → hidden_dim//2)
      ReLU()
      Linear(hidden_dim//2 → 1)

Post-processing:
  - ReLU() or Softplus() to ensure non-negative precipitation
  - Or clip to [0, max_precip]
```

### 3.3 Complete Model Hyperparameters

**Initial Configuration**:
```python
hidden_dim = 64           # Hidden representation size
num_gnn_layers = 2        # Number of message passing layers
dropout = 0.2             # Dropout probability
conv_type = 'GCNConv'     # GNN layer type
use_skip = True           # Use residual connections
use_batch_norm = True     # Use batch normalization
activation = 'relu'       # Activation function
```

---

## 4. TRAINING PIPELINE

### 4.1 Loss Function

**Primary**: Mean Squared Error (MSE)
```python
loss = MSE(predictions, targets)
```

**Alternative**: Mean Absolute Error (MAE)
```python
loss = MAE(predictions, targets)
```

**Considerations**:
- MSE penalizes large errors more (good for outliers)
- MAE more robust to outliers
- Could use weighted MSE to emphasize high precipitation events

**Advanced**: Combined loss
```python
loss = α * MSE + β * MAE + γ * PerceptualLoss
```

### 4.2 Optimizer & Learning Rate

**Optimizer**: Adam
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,          # Initial learning rate
    weight_decay=1e-5  # L2 regularization
)
```

**Learning Rate Scheduler**:
```python
# Option 1: ReduceLROnPlateau
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)

# Option 2: CosineAnnealingLR
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=50
)
```

### 4.3 Training Loop

```
For each epoch:
  1. Training phase:
     For each batch (timestep):
       - Forward pass through GNN
       - Compute loss
       - Backward pass
       - Update weights
       - Accumulate metrics
  
  2. Validation phase:
     - No gradient computation
     - Evaluate on validation set
     - Compute validation metrics
     - Check for best model
  
  3. Logging:
     - Print epoch statistics
     - Save metrics for plotting
     - Save best model checkpoint
  
  4. Early stopping:
     - If validation loss doesn't improve for N epochs
     - Stop training
```

**Hyperparameters**:
```python
num_epochs = 100          # Maximum training epochs
batch_size = 32           # Number of timesteps per batch
early_stop_patience = 15  # Epochs without improvement
```

### 4.4 Metrics & Evaluation

**Training Metrics** (per epoch):
```python
1. MSE (Mean Squared Error)
   - Primary training objective
   
2. RMSE (Root Mean Squared Error)
   - Same units as precipitation (m)
   - More interpretable
   
3. MAE (Mean Absolute Error)
   - Average absolute error
   - Robust metric
   
4. R² Score (Coefficient of Determination)
   - Proportion of variance explained
   - Range: [0, 1], higher is better
   
5. Correlation Coefficient
   - Linear relationship strength
   - Range: [-1, 1]
```

**Advanced Metrics**:
```python
6. Precipitation-specific:
   - Accuracy for precipitation detection (>0 threshold)
   - False alarm rate
   - Hit rate for heavy precipitation events
   
7. Spatial metrics:
   - Error distribution across latitudes
   - Coastal vs inland performance
```

---

## 5. EVALUATION & VISUALIZATION

### 5.1 Quantitative Evaluation
```
1. Compute metrics on test set
2. Analyze errors by:
   - Spatial location (latitude, longitude)
   - Precipitation intensity bins
   - Season (if applicable)
```

### 5.2 Visualization Pipeline

**A. Prediction Maps**:
```
For selected timesteps:
  - Plot ground truth precipitation map
  - Plot predicted precipitation map
  - Plot absolute error map
  - Side-by-side comparison
```

**B. Training Curves**:
```
- Training loss vs validation loss over epochs
- Metric evolution (RMSE, MAE, R²)
- Learning rate schedule
```

**C. Scatter Plots**:
```
- Predicted vs actual precipitation
- Color-coded by location or intensity
- Regression line + R² score
```

**D. Error Analysis**:
```
- Histogram of errors
- Spatial distribution of high errors
- Temporal evolution of predictions
```

**E. Interactive Animations** (optional):
```
- Time-lapse of predictions vs ground truth
- Error evolution over time
```

---

## 6. IMPLEMENTATION MODULES

### File Structure
```
graph.py                    # Main implementation file
├── DataLoader
│   ├── load_zarr_data()
│   ├── create_graph_structure()
│   ├── normalize_features()
│   └── create_data_splits()
├── GraphConstruction
│   ├── build_grid_edges()
│   ├── build_knn_edges()
│   └── compute_edge_attributes()
├── WeatherGNN (nn.Module)
│   ├── __init__()
│   ├── forward()
│   └── helper methods
├── Training
│   ├── train_epoch()
│   ├── validate_epoch()
│   └── train_model()
├── Evaluation
│   ├── compute_metrics()
│   ├── evaluate_test_set()
│   └── error_analysis()
└── Visualization
    ├── plot_training_curves()
    ├── plot_predictions()
    └── create_comparison_maps()
```

### Key Classes & Functions

**1. WeatherDataset(torch.utils.data.Dataset)**
```python
Purpose: Wrap data for PyTorch DataLoader
Methods:
  - __init__(features, targets, edge_index, ...)
  - __len__(): return number of timesteps
  - __getitem__(idx): return Data object for timestep idx
```

**2. WeatherGNN(torch.nn.Module)**
```python
Purpose: Main GNN model
Attributes:
  - encoder: feature encoding layer
  - gnn_layers: list of GNN conv layers
  - batch_norms: normalization layers
  - decoder: output prediction layer
Methods:
  - __init__(in_channels, hidden_dim, ...)
  - forward(x, edge_index): predict precipitation
  - reset_parameters(): initialize weights
```

**3. Training Functions**
```python
train_one_epoch(model, train_loader, optimizer, criterion)
validate(model, val_loader, criterion)
train_model(model, train_loader, val_loader, num_epochs, ...)
```

**4. Evaluation Functions**
```python
compute_metrics(predictions, targets) → dict
evaluate_test_set(model, test_loader) → results
analyze_errors(predictions, targets, positions) → analysis
```

---

## 7. HYPERPARAMETER TUNING STRATEGY

### Initial Baseline
```python
# Start simple, then increase complexity
hidden_dim = 64
num_layers = 2
dropout = 0.2
learning_rate = 0.001
batch_size = 32
```

### Tuning Order (Sequential)
```
1. Learning rate: [0.0001, 0.0005, 0.001, 0.005]
2. Hidden dimension: [32, 64, 128, 256]
3. Number of layers: [1, 2, 3, 4]
4. Dropout: [0.0, 0.1, 0.2, 0.3]
5. Batch size: [16, 32, 64]
6. GNN type: [GCN, GAT, GraphSAGE]
```

---

## 8. EXPECTED CHALLENGES & SOLUTIONS

### Challenge 1: Class Imbalance (Precipitation)
**Problem**: Most timesteps have little/no precipitation
**Solutions**:
- Weighted loss function (emphasize rain events)
- Focal loss
- Separate models for detection vs amount

### Challenge 2: Spatial Autocorrelation
**Problem**: Nearby cells are highly correlated
**Solutions**:
- This is actually good for GNNs!
- Use appropriate edge construction
- Consider attention mechanisms

### Challenge 3: Extreme Values
**Problem**: Very high precipitation outliers
**Solutions**:
- Log transform target
- Robust normalization (quantile-based)
- Clip extreme values

### Challenge 4: Static vs Dynamic Features
**Problem**: Elevation is static, others vary with time
**Solutions**:
- Separate encoding for static features
- Concatenate static features at each timestep
- Learn importance weights

---

## 9. SUCCESS CRITERIA

### Minimum Viable Model
```
- RMSE < baseline (climatology or persistence)
- R² > 0.5
- Visually reasonable predictions
- Stable training (no divergence)
```

### Good Performance
```
- RMSE significantly better than baseline
- R² > 0.7
- Captures spatial patterns well
- Good generalization to test set
```

### Excellent Performance
```
- R² > 0.85
- Accurate precipitation detection
- Handles extreme events
- Low spatial bias
```

---

## 10. IMPLEMENTATION CHECKLIST

### Phase 1: Data Pipeline ✓ (Done)
- [x] Download and store data
- [ ] Load zarr data into numpy/torch
- [ ] Create graph structure (nodes + edges)
- [ ] Normalize features
- [ ] Create train/val/test splits
- [ ] Create PyTorch Geometric Data objects
- [ ] Test data loading

### Phase 2: Model Architecture
- [ ] Implement WeatherGNN class
- [ ] Add feature encoder
- [ ] Add GNN layers (start with GCN)
- [ ] Add output decoder
- [ ] Test forward pass
- [ ] Count parameters

### Phase 3: Training Pipeline
- [ ] Implement train_one_epoch()
- [ ] Implement validate()
- [ ] Add loss function
- [ ] Add optimizer
- [ ] Add learning rate scheduler
- [ ] Add early stopping
- [ ] Add model checkpointing

### Phase 4: Evaluation
- [ ] Implement metrics computation
- [ ] Test set evaluation
- [ ] Error analysis
- [ ] Statistical significance tests

### Phase 5: Visualization
- [ ] Training curves
- [ ] Prediction maps
- [ ] Scatter plots
- [ ] Error distributions
- [ ] Save figures

### Phase 6: Optimization
- [ ] Hyperparameter tuning
- [ ] Try different GNN architectures
- [ ] Experiment with edge construction
- [ ] Add advanced features

---

## 11. TIMELINE ESTIMATE

```
Phase 1 (Data):         ✓ Complete
Phase 2 (Model):        30-45 minutes
Phase 3 (Training):     30-45 minutes
Phase 4 (Evaluation):   20-30 minutes
Phase 5 (Viz):          20-30 minutes
Phase 6 (Optimization): Ongoing/iterative

Total initial implementation: ~2-3 hours
Full optimization: Days/weeks of experimentation
```

---

## READY TO IMPLEMENT!

This architecture provides a solid foundation for a production-quality weather prediction GNN. We'll implement this step-by-step in `graph.py`, starting with the data pipeline and progressively building up to a complete, trainable model.

