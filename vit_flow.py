import torch
from PIL import Image
import numpy
import sys
from torchvision import transforms
import numpy as np
import cv2

import maxflow

def compute_a_matrices(attentions, discard_ratio, head_fusion):
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
        
        a = a / a.sum(dim=-1)
        # a = a / a.sum(dim=-1, keepdim=True) -> verify which one is correct
        
        a_matrices.append(a)  
    
    return a_matrices
    
    
def make_graph(a_matrices):
    '''Must make graph such the the source is the class token at final layer (or given layer)
    and the sink are the input embeddings
    Uses PyMaxflow (or not)
    The output must be the weights assigned to the initial tokens...
    '''
    ...

def flow(attentions, discard_ratio, head_fusion):
    '''Generates attention flow mask in similar fashion as rollout
    '''
    
    # Get the a_matrices
    a_matrices = compute_a_matrices(attentions, discard_ratio, head_fusion)
    
    # Generates the graph structure
    graph = make_graph(a_matrices)
    
    # must generate mask
    mask = ... # TODO

    # Adjust for output
    width = int(mask.size(-1)**0.5)
    mask = mask.reshape(width, width).numpy()
    mask = mask / np.max(mask)
    
    return mask

class VITAttentionFlow:
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

        return flow(self.attentions, self.discard_ratio, self.head_fusion)