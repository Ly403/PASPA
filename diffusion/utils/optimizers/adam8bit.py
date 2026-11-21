from bitsandbytes.optim import AdamW8bit

class AdamW8bitWrapper(AdamW8bit):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
