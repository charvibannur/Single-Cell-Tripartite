import pandas as pd
import numpy as np
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
from dgl.nn.pytorch import GraphConv
import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# Tensorboard
# log_dir = "C:/Users/charv/Desktop/Charvi/Sem7/IBAB/Geneformer/logs"
# writer = SummaryWriter(log_dir=log_dir)

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available")
    device = torch.device("cpu")


# Load your data
X = pd.read_csv("X.csv")
y = pd.read_csv("y.csv")

# Create a graph
G = nx.Graph()

cell_ids = X['cell_id'].tolist()
x_columns = X.columns[1:].tolist()
y_columns = y.columns[1:].tolist()

G.add_nodes_from(cell_ids, bipartite=0)
G.add_nodes_from(x_columns, bipartite=1)
G.add_nodes_from(y_columns, bipartite=2)

for index, row in X.iterrows():
    cell_id = row['cell_id']
    for x_col in x_columns:
        value = row[x_col]
        if int(value) != 0:
            G.add_edge(cell_id, x_col, weight=int(value))
        else:
            G.add_edge(cell_id, x_col, weight=0)

for index, row in y.iterrows():
    cell_id = row['cell_id']
    for y_col in y_columns:
        value = row[y_col]
        if int(value) != 0:
            G.add_edge(cell_id, y_col, weight=int(value))
        else:
            G.add_edge(cell_id, x_col, weight=0)

print("Finished Graph construction")

# Convert the networkx graph to a DGL graph
dgl_G = dgl.from_networkx(G)
dgl_G = dgl_G.to(device)

# Separate the node types
cell_nodes = cell_ids
x_nodes = x_columns
y_nodes = y_columns

# Get the edges and their weights
edges = dgl_G.edges()
print(len(edges[1]))
print(edges[1])
# edge_weights = torch.tensor([d['weight'] for u, v, d in G.edges(data=True)])
edge_weights = edges[1]
# edge_weights = dgl_G.edata['weight']
print(edge_weights)
num_edges = dgl_G.number_of_edges()

print(num_edges)
print("Starting Model")
# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GNNModel, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)

    def forward(self, graph, edge_weights):
        h = self.conv1(graph, graph.ndata['feat'])  # Use node features from graph.ndata
        h = torch.relu(h)
        h = self.conv2(graph, h)

        # Calculate edge predictions based on edge weights and node features
        edge_preds = (h[edges[0]] * h[edges[1]]).sum(dim=1)

        return edge_preds



x = len(x_nodes) + len(cell_nodes)
# Initialize the GNN model
in_features = len(x_nodes)
hidden_size = 64
out_features = len(y_nodes)
num_cell_nodes = len(cell_nodes)
gnn_model = GNNModel(x, hidden_size, out_features)
gnn_model.to(device)




# Initialize node feature tensors for x_nodes and y_nodes
x_features = torch.rand(in_features, x)
y_features = torch.rand(out_features, x)
cell_features = torch.randn(num_cell_nodes, x)

x_features = x_features.to(device)
y_features = y_features.to(device)
cell_features = cell_features.to(device)
edge_weights = edge_weights.to(device)

print(x_features.shape)
print(y_features.shape)
print(cell_features.shape)
# Assign node features to the DGL graph
dgl_G.ndata['feat'] = torch.cat([x_features, y_features, cell_features], dim=0)
dgl_G.ndata['feat'] = torch.cat([x_features, y_features, cell_features], dim=0).to(device)
dgl_G = dgl.add_self_loop(dgl_G)

print("Initialized Features")
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(gnn_model.parameters(), lr=0.01)

print("Start Training")
global_step = 0
# Training loop
for epoch in range(10):
    gnn_model.train()
    optimizer.zero_grad()
    y_preds = gnn_model(dgl_G, edge_weights)
    loss = criterion(y_preds, edge_weights.float())  # Use the same tensor for both preds and targets
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/10], Loss: {loss.item():.4f}')

    if (epoch + 1) % 1 == 0:  # You can adjust the frequency of printing
        print("Model Weights:")
        for name, param in gnn_model.state_dict().items():
            if 'weight' in name:  # Check if the parameter is a weight tensor
                print(name, param)

    # TensorBoard
    # writer.add_scalar('Loss/train', loss.item(), global_step)
    # writer.add_histogram('Weights/layer1', gnn_model.conv1.weight, global_step)
    # writer.add_scalar('Accuracy/train', accuracy, global_step)

    global_step += 1  # Increment global step for each iteration

# Get predictions for the y node weights
gnn_model.eval()
with torch.no_grad():
    y_preds = gnn_model(dgl_G, edge_weights)

# Compare predictions to actual y node weights
for i, y_node in enumerate(y_nodes):
    print(f'Predicted weight for edge ({cell_nodes[0]}, {y_node}): {y_preds[i].item():.4f}')
    print(f'Actual weight for edge ({cell_nodes[0]}, {y_node}): {G.edges[cell_nodes[0], y_node]["weight"]:.4f}\n')
