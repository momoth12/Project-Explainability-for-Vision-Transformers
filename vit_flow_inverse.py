import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

import maxflow
from tqdm import tqdm

def compute_a_matrices_inverse(attentions, discard_ratio, head_fusion):
    '''Generates the A matrices as in the paper from the attention layers
    '''
    a_matrices = []
    
    for attention in attentions:
        # Getting type of fusion of channels
        if head_fusion == "mean":
            attention_heads_fused = attention.mean(axis=1)
        elif head_fusion == "max":
            attention_heads_fused = attention.max(axis=1)[0]
        elif head_fusion == "min":
            attention_heads_fused = attention.min(axis=1)[0]
        else:
            raise "Attention head fusion type Not supported"
        
        # Drop the lowest attentions, but
        # don't drop the class token
        flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
        _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
        indices = indices[indices != 0]
        flat[0, indices] = 0
        
        
        I = torch.eye(attention_heads_fused.size(-1))
        a = (attention_heads_fused + 1.0*I)/2
        
        # a = a / a.sum(dim=-1)
        # a = a / a.sum(dim=-1, keepdim=True)# -> verify which one is correct
        
        a_matrices.append(a)  
    
    return a_matrices
    
    
def compute_flow_inverse(a_matrices, input_node, output_flow, discard_ratio):
    '''Compute flow of a single input source
    '''

    n_tokens = a_matrices[0].size(-1)
    n_layers = len(a_matrices)
    n_nodes = n_layers * n_tokens
    n_vertices = int((1 - discard_ratio) * n_tokens**2 + 1)

    g = maxflow.Graph[float](n_nodes, n_vertices) 
    nodes = g.add_nodes(n_nodes) 

    ## Setting source nodes (final attention token nodes)
    source_weights = a_matrices[0][0, :, input_node]
    total_input_flow = source_weights.sum()
        
    for node_number in range(n_tokens):
        g.add_tedge(nodes[node_number], source_weights[node_number], 0)

    ## Setting internal edges
    for n_layer, a_matrix in enumerate(a_matrices):
        if n_layer == 0: continue
        
        start_node = (n_layer - 1) * n_tokens # first node of current layer
        start_node_next = (n_layer) * n_tokens # first node of next layer
        
        for idx_x, node_number in enumerate(range(start_node, start_node + n_tokens)):
            weights = a_matrix[0, :, idx_x] # weight from current node to following layer 
            
            for idx_y in range(n_tokens):
                # if idx_y != 0: 
                #     continue
                
                weight = weights[idx_y]
                
                # No need to create link if weight is zero
                if weight == 0: 
                    continue
                
                node_number_next = start_node_next + idx_y
                g.add_edge(nodes[node_number], nodes[node_number_next], weight, 0.) # next layer points to layer before
                
    ## Setting sink nodes (input token nodes)
    for idx, out_node in enumerate(range(n_nodes - n_tokens, n_nodes)):
        if idx > 0: break
        g.add_tedge(nodes[out_node], 0., total_input_flow)
        
    max_flow = g.maxflow()
    
    return max_flow


def compute_all_flows_inverse(a_matrices, discard_ratio, output_flow=200.):
    '''Compute flow for all sources
    '''
    n_tokens = a_matrices[0].size(-1)
    
    mask = torch.Tensor(np.zeros(n_tokens))
    
    for n_token in tqdm(range(n_tokens)):
        mask[n_token] = compute_flow_inverse(a_matrices, n_token, discard_ratio, output_flow)
        
    mask = mask[1:]
    
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    
    return mask

def attention_flow_mask(attentions, discard_ratio, head_fusion, output_flow=200.):
    '''Compute flow for all sources
    '''
    # Getting a_matrices
    a_matrices = compute_a_matrices_inverse(attentions, discard_ratio, head_fusion)
    
    # Getting mask
    n_tokens = a_matrices[0].size(-1)
    
    mask = torch.Tensor(np.zeros(n_tokens))
    
    for n_token in tqdm(range(n_tokens)):
        mask[n_token] = compute_flow_inverse(a_matrices, n_token, discard_ratio, output_flow)
        
    mask = mask[1:]
    
    return mask

def flow_inverse(attentions, discard_ratio, head_fusion, output_flow=200.):
    '''Generates attention flow mask in similar fashion as rollout
    '''
    
    # Getting a_matrices
    a_matrices = compute_a_matrices_inverse(attentions, discard_ratio, head_fusion)
    
    # must generate mask
    mask = compute_all_flows_inverse(a_matrices, discard_ratio, output_flow)

    return mask

class VITAttentionFlowInverse:
    def __init__(self, model, attention_layer_name='attn_drop', head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)

        return flow_inverse(self.attentions, self.discard_ratio, self.head_fusion)
    
    def get_attention_mask(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            output = self.model(input_tensor)
            
        a_matrices = compute_a_matrices_inverse(self.attentions, self.discard_ratio, self.head_fusion)
            
        return attention_flow_mask(self.attentions, self.discard_ratio, self.head_fusion, output_flow=200.)