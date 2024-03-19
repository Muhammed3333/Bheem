import numpy as np
import torch 

from bio_embeddings.embed import ProtTransBertBFDEmbedder,SeqVecEmbedder

seq = 'MVTYDFGSDEMHD' 

embedder = SeqVecEmbedder()
embedding = embedder.embed(seq)
protein_embd = torch.tensor(embedding).sum(dim=0) # Vector with shape [L x 1024]
np_arr = protein_embd.cpu().detach().numpy()
