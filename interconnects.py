import sys
sys.path.insert(0,'./modules')

import torch
from torch import nn
from utils import make_back_tensor,get_dist_mat
from kp_utils import get_top_coords,get_locs_tensor
import numpy as np


class CNN_to_Graph(nn.Module):
      def __init__(self):
          super(CNN_to_Graph, self).__init__()
      
      def forward(self,features,top_coords): # features = feature map at the output of CNN, top_coords= the locations to be selected as keypoints represented in form i*Width + j
          key_locs=get_locs_tensor(top_coords)
          adj_mat=get_dist_mat(key_locs)
          top_coords=top_coords.unsqueeze(2)
          top_coords=top_coords.repeat(1,1,128)
          top_coords=top_coords.to(features.device)
          node_features=torch.gather(features,1,top_coords)
          adj_mat=adj_mat.to(features.device)
          return node_features,adj_mat
    
 
class Graph_to_CNN(nn.Module):
      def __init__(self,fv_size=64,fm_width=16,fm_height=16):
          super(Graph_to_CNN, self).__init__()
          self.feature_vector_size=fv_size #Size of feature vector of graph nodes
          self.feature_map_width=fm_width   #Width of CNN feature maps
          self.feature_map_height=fm_height  #Height of CNN feature maps
      
      def forward(self,features,key_locs):
          batch_size=features.shape[0]
          feature_maps=torch.zeros([batch_size,self.feature_vector_size,self.feature_map_height,self.feature_map_width]).to(features.device)
          features_count=np.zeros([batch_size,self.feature_map_height,self.feature_map_width])
          feature_maps=make_back_tensor(features,feature_maps,features_count,key_locs,self.feature_map_height)
          return feature_maps
