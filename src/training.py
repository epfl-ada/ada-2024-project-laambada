import os
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from scripts.load_and_save import load_data
import tqdm
import rdkit.Chem as Chem
from torch_geometric.data import Data, Batch

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Extract atom features
    features = []
    for atom in mol.GetAtoms():
        features.append([atom.GetAtomicNum()])
    x = torch.tensor(features, dtype=torch.float)

    # Create edge index from molecular graph
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    return Data(x=x, edge_index=edge_index)

class DMPNN(MessagePassing):
    def __init__(self, hidden_dim):
        super(DMPNN, self).__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.node_embedding = nn.Linear(1, hidden_dim)
        self.edge_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.node_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # Initial node embeddings
        x = F.relu(self.node_embedding(x))
        
        # Edge embeddings
        edge_embeddings = self.edge_network(
            torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
        )
        
        # Message passing
        x = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
        
        # Global pooling per molecule
        x = torch.zeros_like(x).scatter_add_(0, batch.unsqueeze(-1).expand(-1, x.size(-1)), x)
        unique_batch, counts = torch.unique(batch, return_counts=True)
        x = x[unique_batch] / counts.float().unsqueeze(-1)
        
        # Final prediction
        out = self.output_layer(x)
        return out.squeeze()

def train():
    # Load your data here
    data = load_data()
    if os.path.exists('data.pt'):
        graphs = torch.load('data.pt')
    else:
        graphs = []
        for smiles in tqdm.tqdm(data['Ligand SMILES']):
            graph = smiles_to_graph(smiles)
            graphs.append(graph)
        torch.save(graphs, 'data.pt')

    print(data.shape)
    target = "pIC50"
    # Only use the lines that have the target values
    data = data.dropna(subset=[target])
    targets = torch.tensor(data[target].values, dtype=torch.float)

    # Create dataset and loader
    dataset = []
    for graph, target_value in zip(graphs, targets):
        graph.y = target_value
        dataset.append(graph)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, optimizer, and loss function
    hidden_dim = 128
    model = DMPNN(hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Training loop
    for epoch in tqdm.tqdm(range(10)):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)  # Add batch
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}')

    # Save the model
    torch.save(model.state_dict(), 'model.pt')
    
    # Evaluate the model on MAE
    eval_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mae = 0
    for batch in eval_loader:
        out = model(batch.x, batch.edge_index, batch.batch)
        mae += (out - batch.y).abs().sum().item()

    print(f'MAE: {mae / len(dataset)}')
if __name__ == '__main__':
    train()