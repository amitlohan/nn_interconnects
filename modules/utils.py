import torch
import torch.functional as F
import os

def get_dist_mat(coords_org):
    coords=coords_org.clone().detach()
    n=coords.shape[1]
    #print(n)
    #breakpoint()
    tid=torch.eye(n)
    #print("distmat device",coords_org.device)
    tid=tid.to(coords_org.device)
    tid.require_grad=False

    dist_mat=coords.clone()
    dist_mat=dist_mat.unsqueeze(2)
    coords=coords.unsqueeze(2)
    #print(coords.shape,n)
    for i in range(n-1):
        dist_mat=torch.cat([dist_mat,coords],dim=2)
        
    #print(dist_mat.shape)
    #print(tid.shape)
    dist_mat = dist_mat.sub(coords.transpose(1,2))
    dist_mat = torch.mul(dist_mat,dist_mat)
    dist_mat = torch.sum(dist_mat,dim=3)
    dist_mat = torch.sqrt(dist_mat) + 1
    dist_mat = 1/dist_mat
    dist_mat = dist_mat/torch.max(dist_mat)
    dist_mat = dist_mat-tid*dist_mat
    dist_mat = dist_mat+tid*(torch.max(dist_mat))
    
    
    #dist_mat=dist_mat.detach()
    dist_mat.requires_grad=False
    
    return dist_mat
    
    
def make_back_tensor(gout,back_tensor,back_tensor_count,key_locs,grid_size):

    
    key_locs2=key_locs.clone().detach()
    key_locs2.requires_grad=False
    
    key_locs2=torch.floor(key_locs2//(back_tensor.shape[2]))                                                           ##Flag
    
    
    #print("Nan info:",torch.sum(key_locs2.isnan()))
    
    for i in range(key_locs2.shape[0]):
        ctr=0
        for lock in key_locs2[i]:
            back_tensor_count[i,int(lock[0].item()),int(lock[1].item())]+=1
            back_tensor[i,:,int(lock[0].item()),int(lock[1].item())]+=gout[i,ctr]
            ctr+=1
            
    for i in range(key_locs2.shape[0]):
        for lock in key_locs2[i]:
            if back_tensor_count[i,int(lock[0].item()),int(lock[1].item())]>1:
               back_tensor[i,:,int(round(lock[0].item())),int(round(lock[1].item()))]=(back_tensor[i,:,int(round(lock[0].item())),int(round(lock[1].item()))])/back_tensor_count[i,int(lock[0].item()),int(lock[1].item())]
               
               
    return back_tensor
    
def mine(path):
    if not os.path.exists(path):
       os.makedirs(path)
        
    
