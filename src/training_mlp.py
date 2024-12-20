import os
from scripts.load_and_save import load_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists('src/data/embeddings_full_ic50.csv.zip'):
    # Load and preprocess data
    logger.info("Loading data...")
    data_df = load_data()
    path_full_df = 'src/data/embeddings_full.csv.zip'
    data = load_data(path_full_df) 
    # Add IC50 column to data, number of lines should be the same, use Ligand SMILES to match with data_df
    data['pIC50'] = np.nan
    data['Target Name'] = ""
    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        ligand_id = row['Ligand SMILES']
    
        ic50 = data_df.loc[data_df['Ligand SMILES'] == ligand_id, 'pIC50']
        target_name = data_df.loc[data_df['Ligand SMILES'] == ligand_id, 'Target Name']
        
        if len(ic50) > 0:
            data.at[i, 'pIC50'] = ic50.values[0]
        
        data.at[i, 'Target Name'] = target_name.values[0]

    # Save data
    data.to_csv('src/data/embeddings_full_ic50.csv.zip', index=False,  compression='zip')

# Load data
data = load_data('src/data/embeddings_full_ic50.csv.zip')
print(data.head())
print(data['pIC50'].isna().sum()/data.shape[0])
logger.info(f"Data shape: {data.shape}")

# Handle missing values
data = data.dropna(subset=['pIC50'])
logger.info(f"After dropping NA shape: {data.shape}")

print(data['pIC50'].isna().sum()/data.shape[0])

data = data.dropna()
logger.info(f"After dropping NA shape: {data.shape}")

y = data['pIC50'].values

# Ordinal encoding of Target Name column
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
data['Target Name'] = encoder.fit_transform(data['Target Name'].values.reshape(-1, 1))

X = data.drop(['Ligand SMILES', 'pIC50'], axis=1)
X = X.select_dtypes(include=np.number)

# Check for any remaining invalid values
assert not np.any(np.isnan(X))
assert not np.any(np.isnan(y))
logger.info(f"Feature shape: {X.shape}, Target shape: {y.shape}")

# Use RobustScaler instead of StandardScaler for better handling of outliers
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
assert not np.any(np.isinf(X_train_scaled)), "Inf values in X_train_scaled"
print(" No Inf values in X_train_scaled")
X_test_scaled = scaler.transform(X_test)

X_train_scaled = np.clip(X_train_scaled, -10, 10)
X_test_scaled = np.clip(X_test_scaled, -10, 10)


# Verify scaled data
assert not np.any(np.isnan(X_train_scaled))
assert not np.any(np.isnan(X_test_scaled))

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)


# Dataset class
class MoleculeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class IC50Predictor(pl.LightningModule):
    def __init__(self, input_dim):
        super().__init__()
        self.save_hyperparameters()
        self.train_losses = []
        self.val_losses = []
        # Initialize weights with smaller values
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.fill_(0.0)
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),  # Replace BatchNorm with LayerNorm
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 1)
        )
        
        self.model.apply(init_weights)
        self.validation_step_outputs = []
    
    def forward(self, x):
        # Add input validation
        x = torch.clamp(x, min=-10, max=10)
        if torch.isnan(x).any():
            raise ValueError("NaN values in input")
        return self.model(x)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.val_losses.append(loss.item())
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.train_losses.append(loss.item())
        return loss
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), 
                        lr=0.0001,  # Lower learning rate
                        weight_decay=1e-4)  # Increased weight decay
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss"  # Changed to train_loss
        }
        
# Create data loaders
train_dataset = MoleculeDataset(X_train_tensor, y_train_tensor)

val_dataset = MoleculeDataset(X_test_tensor, y_test_tensor)


# Train model
model = IC50Predictor(input_dim=X_train.shape[1])


# Training setup
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min',
            verbose=True,
            min_delta=0.01
        )
    ],
    gradient_clip_val=0.5,
    enable_progress_bar=True,
    accelerator='auto',
    log_every_n_steps=10,
    val_check_interval=0.25  # Validate every 25% of training epoch
)

# Smaller batch size
train_loader = DataLoader(
    train_dataset, 
    batch_size=16,  # Reduced batch size
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=16,  # Reduced batch size
    num_workers=4
)


trainer.fit(model, train_loader, val_loader)

# Evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    predictions = model(X_test_tensor)
    y_pred = predictions.cpu().numpy()


# Validate predictions
assert not np.any(np.isnan(y_pred)), "NaN values in predictions"
assert not np.any(np.isinf(y_pred)), "Inf values in predictions"

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Test MAE: {mae:.4f}')
print(f'Test MSE: {mse:.4f}')
print(f'Test R2: {r2:.4f}')


# Save model
torch.save(model.state_dict(), 'mlp_model.pt')

# Plot training history
plt.plot(model.train_losses, label='Train loss')
plt.plot(model.val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# save plot
plt.savefig('src/data/training_history.png')
