import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(conf["num_features"], conf["hidden_channels"], heads=conf["num_heads"])
        self.conv2 = GATConv(conf["hidden_channels"] * conf["num_heads"], conf["num_classes"])

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_weight)
        return x
        

def train_GAT(model, data, optimizer, loss_fn):
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index, data.edge_weight)
      loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss


def test_GAT(model, data, mask):
      model.eval()
      out = model(data.x, data.edge_index, data.edge_weight)
      out = F.softmax(out, dim=1)
      pred = out.argmax(dim=1)
      correct = pred[mask] == data.y[mask]
      acc = int(correct.sum()) / int(mask.sum())
      return acc
