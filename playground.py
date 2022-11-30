import torch

t  = torch.tensor([[1.,2.,3.], [4.,5.,6.]])
print("tensor:", t)

t1 = torch.nn.functional.normalize(t, p=1.0, dim = 1)
t2 = torch.nn.functional.normalize(t, p=2.0, dim = 0)

print("Normalized tensor:", t1)
print("Normalized tensor:", t2)