import torch
import torch.nn as nn
import torch.nn.functional as F


# 构建模型
class MyGNN(nn.Module):
    def __init__(self, input_dim, edge_dim, global_dim, output_dim,edges):
        super(MyGNN, self).__init__()

        # 节点输入特征 + 边特征 + 全局特征
        self.node_fc = nn.Linear(input_dim, 64)  # 节点特征映射
        self.edge_fc = nn.Linear(edge_dim, 64)  # 边特征映射
        self.global_fc = nn.Linear(global_dim, 64)  # 全局特征映射
        self.edges = edges

        # 合并后的多层感知机
        self.mlp_edge = nn.Sequential(
            nn.Linear(64*3, 128),  # 64*2 是节点和边特征拼接后的特征长度
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.mlp_node = nn.Sequential(
            nn.Linear(64*3, 128),  # 64*2 是节点和边特征拼接后的特征长度
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, node_features, edge_features, global_features):
        # 节点特征映射
        node_emb = F.relu(self.node_fc(node_features))  # (5, 64)

        # 边特征映射
        edge_emb = F.relu(self.edge_fc(edge_features))  # (4, 64)

        #全局特征映射
        global_emb = F.relu(self.global_fc(global_features))  # (1, 64)

        #如果是对顶点分类
        # 需要将边特征按节点进行汇聚
        edge_to_nodes = edge_emb.new_zeros((node_features.size(0), edge_emb.size(1))) #（5，64）

        # 遍历每条边，累加到对应的节点对   只要第i条边上第k个点，就要把这个矩阵的第k行加上第i条边的特征
        for i, (src, dst) in enumerate(self.edges):
            edge_to_nodes[src] += edge_emb[i]  # 累加到源节点
            edge_to_nodes[dst] += edge_emb[i]  # 累加到目标节点

        combined_features_node = torch.cat([node_emb, edge_to_nodes], dim=1)  # (5, 128)

        combined_features_global_node = torch.cat((combined_features_node, global_emb.expand(combined_features_node.size(0),-1)),dim=1)



        #假如是对边分类
        #汇聚节点信息到边上
        node_to_edge = node_emb.new_zeros((edge_features.size(0),node_emb.size(1)))  # (4, 64)

        for i, (src, dst) in enumerate(self.edges):
            node_to_edge[i] = node_emb[src] + node_emb[dst]

        combined_features_edge = torch.cat([edge_emb, node_to_edge], dim=1)  # (4, 128)
        combined_features_global_edge = torch.cat((combined_features_edge, global_emb.expand(combined_features_edge.size(0),-1)),dim=1) #(4,192)


        return combined_features_global_node,combined_features_global_edge



class MyNet(nn.Module):
    def __init__(self, input_dim, edge_dim, global_dim, output_dim,edges):
        super(MyNet, self).__init__()
        self.gnn1 = MyGNN(input_dim, edge_dim, global_dim, output_dim,edges)
        self.gnn2 = MyGNN(192, 192, global_dim, output_dim,edges)


        self.dropout = nn.Dropout(p=0.1)

        self.mlp_edge = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.mlp_node = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self,node_features, edge_features, global_features):
        n1,e1 = self.gnn1(node_features, edge_features, global_features)
        n1,e1 = self.dropout(n1),self.dropout(e1)

        n2,e2 = self.gnn2(n1,e1,global_features)
        n2,e2 = self.dropout(n2),self.dropout(e2)


        out_node = self.mlp_node(n2)
        out_edge = self.mlp_edge(e2)

        return out_node,out_edge
# 构建数据
node_features = torch.tensor([[1.0, 0.0, 0.5],  # Node 0
                              [0.5, 1.0, 0.5],  # Node 1
                              [0.8, 0.6, 0.3],  # Node 2
                              [0.9, 0.2, 0.1],  # Node 3
                              [0.4, 0.7, 0.9]]) # Node 4

edge_features = torch.tensor([[1.0, 0.3],  # Edge 0-1
                              [0.5, 0.8],  # Edge 1-2
                              [0.7, 0.4],  # Edge 2-3
                              [0.2, 0.9]]) # Edge 3-4

global_features = torch.tensor([[0.5, 0.2, 0.8]])  # 图的全局特征


edges = [(0, 1), (1, 2), (2, 3), (3, 4)]  # 4条边

net = MyNet(input_dim=3, edge_dim=2, global_dim=3, output_dim=2,edges=edges)

out_node,out_edge = net(node_features, edge_features, global_features)
print(out_node.argmax(dim=1))
print(out_edge.argmax(dim=1))

