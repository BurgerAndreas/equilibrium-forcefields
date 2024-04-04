import torch
from e3nn import o3

irreps_node_embedding_injection = "64x0e+32x1e+16x2e"
irreps_in = "64x0e"
irreps_node_embedding = "128x0e+64x1e+32x2e"

irreps_node_embedding = o3.Irreps(irreps_node_embedding)
irreps_node_injection = o3.Irreps(irreps_node_embedding_injection)


# test 1: does addition work as expected?
irreps_node_z = irreps_node_embedding + irreps_node_injection
irreps_node_z = irreps_node_z.simplify()
print(f"irreps_node_z: {irreps_node_z}")

# node_features = torch.cat([node_features, node_features_injection], dim=1)


# torch.zeros_like(node_features_injection)
