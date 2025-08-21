import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载 Cora 数据集
dataset = Planetoid(root='./data', name='Cora')  # Cora 有 2708 个节点，7 类

data = dataset[0].to(device)  # Cora 是单图数据集，data 表示图结构数据


# 构建 GCN 模型
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 32)  # 第一层 GCN
        #self.conv1 = GCNConv(3, 32)  # 第一层 GCN
        self.conv2 = GCNConv(32, dataset.num_classes)        # 第二层输出分类

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x



# 模型初始化
model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss()

#训练模型
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_fn(out[data.train_mask], data.y[data.train_mask])  # 只训练 train_mask 的节点
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 测试模型
model.eval()
_, pred = model(data).max(dim=1)
correct = pred[data.test_mask] == data.y[data.test_mask]
acc = int(correct.sum()) / int(data.test_mask.sum())
print(f"\nTest Accuracy: {acc:.4f}")


# edge = zip(data.edge_index[0,:5].tolist(),data.edge_index[1,:5].tolist())
# for e in edge:
#     print(e)

# node_features = torch.tensor([[1.0, 0.0, 0.5],  # Node 0
#                               [0.5, 1.0, 0.5],  # Node 1
#                               [0.8, 0.6, 0.3]],device=device) # Node 2
#
# # 2. 定义边索引（边由源节点和目标节点构成）
# edge_index = torch.tensor([[0, 1],  # 从节点 0 到节点 1
#                            [1, 2]], dtype=torch.long,device=device)  # 从节点 1 到节点 2
#
# # 3. 定义边特征（假设每条边有 2 个特征）
# edge_features = torch.tensor([[1.0, 0.5],  # 边 0 → 1
#                               [0.8, 0.3]], dtype=torch.float,device=device)  # 边 1 → 2
#
# # 4. 定义节点标签（假设是二分类问题）
# node_labels = torch.tensor([1, 2, 0],device=device)  # 节点 0、1、2 的标签
#
# # 5. 创建训练集、验证集和测试集掩码
# train_mask = torch.tensor([True, True, False],device=device)  # 节点 0、1 是训练集
# val_mask = torch.tensor([False, False, True],device=device)  # 节点 2 是验证集
# test_mask = torch.tensor([False, False, False],device=device)  # 没有测试集
#
# # 6. 创建一个 Data 对象
# mydata = Data(x=node_features,  # 节点特征
#             edge_index=edge_index,  # 边索引
#             edge_attr=edge_features,  # 边特征
#             y=node_labels,  # 节点标签
#             train_mask=train_mask,  # 训练集掩码
#             val_mask=val_mask,  # 验证集掩码
#             test_mask=test_mask)  # 测试集掩码
#
# _,pred =model(mydata).max(dim=1)
# print(pred)
