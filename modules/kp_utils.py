import torch
import numpy as np

def get_top_kl(c_scores,n,npgc=8):
    csf=c_scores.view(c_scores.shape[0],-1)
    tids=torch.argsort(csf,dim=-1,descending=True)[:,0:n]
    conf_mask=torch.zeros(csf.shape)
    
    nb=csf.shape[0]
    
    for i in range(nb):
        conf_mask[i,tids[i,:]]=1
     
    conf_mask=conf_mask.view(c_scores.shape)
    return conf_mask
    
    
def get_top_coords(c_scores,indices,n):
    csf=c_scores.view(c_scores.shape[0],-1)
    tids=torch.argsort(csf,dim=-1,descending=True)[:,0:n]
    
    nb=csf.shape[0]
    
    
    indices=indices.view(nb,-1)
    key_coords=torch.zeros(nb,n)
    
    #print(indices.shape)
    
    #print(indices.shape,key_coords.shape)
    
    for i in range(nb):
        key_coords[i,:]=indices[i,tids[i,:]]
    return key_coords
    
def get_locs_tensor(coords):
    key_locs=torch.empty(coords.shape).unsqueeze(2)
    key_locs=key_locs.repeat(1,1,2)
    #print(coords.shape,key_locs.shape)
    key_locs[:,:,0]=coords//256
    key_locs[:,:,1]=coords%256
    key_locs=key_locs.clone().detach()
    return key_locs
    
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
