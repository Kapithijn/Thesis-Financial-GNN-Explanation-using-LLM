import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv


class GCN(nn.Module):
	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
		super().__init__()
		self.conv1 = GCNConv(in_channels, hidden_channels)
		self.conv2 = GCNConv(hidden_channels, out_channels)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x
    

class GAT(nn.Module):
	def __init__(
		self,
		in_channels: int,
		hidden_channels: int,
		out_channels: int,
		heads: int = 4,
		dropout: float = 0.6,
	):
		super().__init__()
		self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
		self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv1(x, edge_index)
		x = F.elu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x


class GIN(nn.Module):
	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
		super().__init__()
		mlp1 = nn.Sequential(
			nn.Linear(in_channels, hidden_channels),
			nn.ReLU(),
			nn.Linear(hidden_channels, hidden_channels),
		)
		mlp2 = nn.Sequential(
			nn.Linear(hidden_channels, hidden_channels),
			nn.ReLU(),
			nn.Linear(hidden_channels, out_channels),
		)
		self.conv1 = GINConv(mlp1)
		self.conv2 = GINConv(mlp2)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x


class GraphSAGE(nn.Module):
	def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
		super().__init__()
		self.conv1 = SAGEConv(in_channels, hidden_channels)
		self.conv2 = SAGEConv(hidden_channels, out_channels)
		self.dropout = dropout

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = F.dropout(x, p=self.dropout, training=self.training)
		x = self.conv2(x, edge_index)
		return x


def build_model_bundle(config: dict):
	in_channels = int(config["in_channels"])
	hidden_channels = int(config.get("hidden_channels", 64))
	out_channels = int(config["out_channels"])
	dropout = float(config.get("dropout", 0.5))
	gat_heads = int(config.get("gat_heads", 4))

	return {
		"GCN": GCN(in_channels, hidden_channels, out_channels, dropout=dropout),
		"GAT": GAT(in_channels, hidden_channels, out_channels, heads=gat_heads, dropout=dropout),
		"GIN": GIN(in_channels, hidden_channels, out_channels, dropout=dropout),
		"GraphSAGE": GraphSAGE(in_channels, hidden_channels, out_channels, dropout=dropout),
	}
