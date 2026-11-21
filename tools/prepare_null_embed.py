import os
import pickle
import argparse
import torch
from diffusion.data.datasets.beat2.build_vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/remote-home/share/dmb_nas2/liuyi/BEAT2/beat_english_v2.0.0")
parser.add_argument("--null_embed_path", type=str, default="./tools/null_embed.pth")
parser.add_argument("--length", type=int, default=64)
args = parser.parse_args()

lang_model = None
with open(os.path.join(args.data_path, "weights/vocab.pkl"), 'rb') as f:
    lang_model = pickle.load(f)
res = lang_model.get_word_index("")
res = {
    "uncond_prompt_embeds": torch.tensor(res).repeat(args.length),
}
print(res)
torch.save(res, os.path.join(args.null_embed_path))