import torch
idx = torch.tensor([2, 3, 1])

num_idx, out_dim = max(idx)+1, 5
embedding = torch.nn.Embedding(num_idx, out_dim)

print(embedding.weight)
print(embedding(idx))

