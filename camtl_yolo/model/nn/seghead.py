from torch import nn

class SegHead(nn.Identity):
    def __init__(self): 
        super().__init__()
        self.is_seg_head = True
