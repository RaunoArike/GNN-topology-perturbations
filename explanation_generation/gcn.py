import torch
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F


class GCN(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(conf["num_features"], conf["hidden_channels"])
        self.conv2 = GCNConv(conf["hidden_channels"], conf["num_classes"])

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
        

def train(model, data, optimizer, loss_fn):
      model.train()
      optimizer.zero_grad()
      out = model(data.x, data.edge_index)
      loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
      return loss


def test(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[mask] == data.y[mask]
    test_acc = int(test_correct.sum()) / int(mask.sum())
    return test_acc
