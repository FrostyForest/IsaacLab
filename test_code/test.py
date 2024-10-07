import torch
import torch.nn.functional as F




test_tensor=torch.tensor([[1,-1,-1],[0.5,0.5,0.5],[0,0,1],[0,0,1]],dtype=torch.float)

softmax_tensor=torch.softmax(test_tensor,dim=-1)

num_envs=4

action1 = torch.tensor([1,1],dtype=torch.float)
action1=action1.unsqueeze(0).repeat(num_envs,1)

action2 = torch.tensor([1,-1],dtype=torch.float)
action2=action2.unsqueeze(0).repeat(num_envs,1)

action3 = torch.tensor([1,1],dtype=torch.float)
action3=action3.unsqueeze(0).repeat(num_envs,1)

actions = torch.stack([action1,action2,action3],dim=1)
print(actions,actions.shape)
print(softmax_tensor,softmax_tensor.shape)
result = torch.einsum("ij,ijk->ik", softmax_tensor, actions)

print(result,result.shape)