import torch
from retriever import *
retriever = Retriever(10)

problem = torch.randn(3, 6, 10)
logic = torch.randn(8, 5, 10)
attn_mask_p = torch.ones(3, 6)
# attn_mask_p[2,4:] = 0s
attn_mask_l = torch.ones(8, 5)

outputs = retriever(problem, logic, attn_mask_p, attn_mask_l)
print(outputs.shape)
# print(outputs)