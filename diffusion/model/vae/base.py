import torch.nn as nn

class IdentityAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()
    
    def encode(self, **kwargs):
        return self.encoder(kwargs)
    
    def decode(self, **kwargs):
        return self.decoder(kwargs)