import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_scatter import scatter_add
import rdkit.Chem as Chem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
# Assuming load_data is defined elsewhere and returns a DataFrame
from scripts.load_and_save import load_data

############################################################
# Feature extraction functions
############################################################
def featurize_atom(atom):
    # Example comprehensive atom features
    atom_type = atom.GetAtomicNum()  # Atomic number
    degree = atom.GetDegree()
    formal_charge = atom.GetFormalCharge()
    hybridization = atom.GetHybridization()
    hybridization_map = {
        Chem.rdchem.HybridizationType.SP: 1,
        Chem.rdchem.HybridizationType.SP2: 2,
        Chem.rdchem.HybridizationType.SP3: 3,
        Chem.rdchem.HybridizationType.SP3D: 4,
        Chem.rdchem.HybridizationType.SP3D2: 5
    }
    hybridization_val = hybridization_map.get(hybridization, 0)
    aromatic = atom.GetIsAromatic()
    # Create a feature vector
    return [
        atom_type,
        degree,
        formal_charge,
        hybridization_val,
        1 if aromatic else 0
    ]

def featurize_bond(bond):
    bond_type = bond.GetBondType()
    # One-hot for bond types: SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=4
    bond_type_map = {
        Chem.rdchem.BondType.SINGLE: 1,
        Chem.rdchem.BondType.DOUBLE: 2,
        Chem.rdchem.BondType.TRIPLE: 3,
        Chem.rdchem.BondType.AROMATIC: 4
    }
    btype = bond_type_map.get(bond_type, 0)
    # Bond features
    return [btype]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Handle invalid SMILES
        return None

    # Atom features
    x = []
    for atom in mol.GetAtoms():
        x.append(featurize_atom(atom))
    x = torch.tensor(x, dtype=torch.float)

    # Edges
    edges = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = featurize_bond(bond)
        edges.append([i, j])
        edges.append([j, i])
        edge_features.append(bf)
        edge_features.append(bf)

    if len(edges) == 0:
        # Handle molecules with no bonds (e.g., single atom)
        edge_index = torch.zeros((2,0), dtype=torch.long)
        edge_attr = torch.zeros((0,1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

############################################################
# Model Definition (DMPNN-like)
############################################################

class DMPNN(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_heads=4):
        super(DMPNN, self).__init__(aggr='add')
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)

        # Message transformation
        self.W_msg = nn.Linear(hidden_dim, hidden_dim)

        # Attention layers
        # For edge update: edges attend to the incoming messages from target nodes.
        self.edge_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)
        
        # For node update: nodes attend to the aggregated edge states.
        self.node_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=False)

        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # Initial embeddings
        h_node = F.relu(self.node_embedding(x))
        h_edge = F.relu(self.edge_embedding(edge_attr))

        # We'll do a fixed number of message passing steps, say 3
        num_steps = 3
        h_edge_states = h_edge.clone()

        for _ in range(num_steps):
            # Compute messages
            h_prop = self.propagate(edge_index, x=h_node, edge_attr=h_edge_states)
            
            # Update edge states with attention:
            # Queries: current edge states
            # Keys/Values: incoming messages at target nodes (matched per-edge)
            target_nodes = edge_index[1]

            # Reshape for attention: (sequence_length, batch_size, embed_dim)
            # Treat each edge as a "sequence element" and batch=1
            queries = h_edge_states.unsqueeze(1)  # (E, 1, H)
            kv = h_prop[target_nodes].unsqueeze(1)  # (E, 1, H)

            # MultiheadAttention expects shape (L, N, E) [L=seq_len, N=batch_size, E=embed_dim]
            # Currently queries and kv are (E, 1, H), which matches (L=E, N=1)
            # This is acceptable. We'll get output of the same shape.
            updated_edges, _ = self.edge_attention(query=queries, key=kv, value=kv)
            h_edge_states = updated_edges.squeeze(1)  # back to (E, H)

        # After message passing at the edge-level, aggregate edge states to nodes
        node_states = scatter_add(h_edge_states, edge_index[1], dim=0, dim_size=h_node.size(0))
        counts = scatter_add(torch.ones_like(edge_index[1], dtype=node_states.dtype), edge_index[1], dim=0, dim_size=h_node.size(0))
        counts = counts.clamp(min=1)
        node_states = node_states / counts.unsqueeze(-1)

        # Update node states with attention:
        # Queries: current node states
        # Keys/Values: newly computed node_states from edges
        queries = h_node.unsqueeze(1)     # (N, 1, H)
        kv = node_states.unsqueeze(1)     # (N, 1, H)
        updated_nodes, _ = self.node_attention(query=queries, key=kv, value=kv)
        h_node = updated_nodes.squeeze(1)  # (N, H)

        # Global pooling
        h_mol = global_mean_pool(h_node, batch)
        return self.readout(h_mol).squeeze(-1)

    def message(self, x_j, edge_attr):
        # Message is a function of the neighbor node and edge states
        return self.W_msg(x_j + edge_attr)

    def update(self, aggr_out):
        # No direct update needed here, handled above
        return aggr_out
    
############################################################
# Training Function
############################################################
def train_model():
    data_df = load_data()
    data_df = data_df.dropna(subset=["pIC50"])
    smiles_list = data_df['Ligand SMILES'].values
    y_values = data_df['pIC50'].values.reshape(-1, 1)

    # Train/Val/Test Split
    X_train, X_temp, y_train, y_temp = train_test_split(smiles_list, y_values, test_size=0.1, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    # Convert SMILES to graphs
    def featurize_smiles_list(X, y):
        graphs = []
        valid_y = []
        for smi, val in zip(X, y):
            g = smiles_to_graph(smi)
            if g is not None:
                g.y = torch.tensor(val, dtype=torch.float)
                graphs.append(g)
                valid_y.append(val)
        return graphs, np.array(valid_y)

    train_graphs, y_train = featurize_smiles_list(X_train, y_train)
    val_graphs, y_val = featurize_smiles_list(X_val, y_val)
    test_graphs, y_test = featurize_smiles_list(X_test, y_test)

    # Target Normalization
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y_train)
    y_val = scaler.transform(y_val)
    y_test = scaler.transform(y_test)

    for g, val in zip(train_graphs, y_train):
        g.y = torch.tensor(val, dtype=torch.float)
    for g, val in zip(val_graphs, y_val):
        g.y = torch.tensor(val, dtype=torch.float)
    for g, val in zip(test_graphs, y_test):
        g.y = torch.tensor(val, dtype=torch.float)

    # Data Loaders
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

    # Determine feature dimensions
    # Assume node feature dimension from first sample
    node_dim = train_graphs[0].x.size(1)
    edge_dim = train_graphs[0].edge_attr.size(1) if train_graphs[0].edge_attr is not None else 0
    hidden_dim = 128

    model = DMPNN(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    max_epochs = 1
    losses = []
    for epoch in tqdm(range(max_epochs)):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader)
        for batch in pbar:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = criterion(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            losses.append(loss.item())
            pbar.set_description(f"Loss: {loss.item():.4f}")

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                loss = criterion(out, batch.y.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping
        if abs(val_loss - best_val_loss) > 0.01:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("Early stopping triggered.")
                break
    
    # Plot losses
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    # Save plot
    plt.savefig('loss_plot.png')
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Evaluate on Test
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            preds.append(out.cpu().numpy())
            actuals.append(batch.y.view(-1).cpu().numpy())
    preds = np.concatenate(preds, axis=0).reshape(-1,1)
    actuals = np.concatenate(actuals, axis=0).reshape(-1,1)#
    # De-normalize#
    preds = scaler.inverse_transform(preds)
    actuals = scaler.inverse_transform(actuals)
    mae = np.mean(np.abs(preds - actuals))
    mse = np.mean((preds - actuals)**2)
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    
    # Save train loss to csv
    pd.DataFrame(losses, columns=['loss']).to_csv('train_loss.csv', index=False)
    
if __name__ == '__main__':
    train_model()
