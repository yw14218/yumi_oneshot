import rerun as rr
from pathlib import Path
from argparse import ArgumentParser
import torch

from mini_dust3r.api import OptimizedResult, inferece_dust3r, log_optimized_result
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.utils.image import load_images
from PIL import Image
import numpy as np
from typing import Union, List, Literal
import trimesh

mask1 = load_images(
    folder_or_list=["experiments/scissor/demo_head_seg.png"], size=512, verbose=True, norm=False
    )[0]['img']
mask2 = load_images(
    folder_or_list=["experiments/scissor/demo_wrist_seg.png"], size=512, verbose=True, norm=False
    )[0]['img']

gray_mask1 = np.mean(mask1, axis=2).astype(bool)
gray_mask2 = np.mean(mask2, axis=2).astype(bool)

np.expand_dims(~gray_mask1, axis=0)
np.expand_dims(~gray_mask1, axis=0)

def main(image_dir: Path):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = AsymmetricCroCo3DStereo.from_pretrained(
        "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    ).to(device)

    optimized_results: OptimizedResult = inferece_dust3r(
        image_dir_or_list=image_dir,
        model=model,
        device=device,
        batch_size=1,
        # bg_mask1=np.expand_dims(np.stack([~gray_mask2, ~gray_mask1]), axis=0),
        # bg_mask2=np.expand_dims(np.stack([~gray_mask2, ~gray_mask1]), axis=0),
    )
    print(optimized_results.mesh)
    optimized_results.mesh.export('my_mesh.obj')
    log_optimized_result(optimized_results, Path("world"))
    

if __name__ == "__main__":
    parser = ArgumentParser("mini-dust3r rerun demo script")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images to process",
        required=True,
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "mini-dust3r")
    main(args.image_dir)
    rr.script_teardown(args)