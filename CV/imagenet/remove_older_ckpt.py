import os
import glob
import shutil
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument(
        "--ckpt-dir",
		default=None,
        help="The ckpt folder on disk."
)
parser.add_argument(
        "--latest-two",
		action="store_true",
        help=""
)
args = parser.parse_args()

ckpt_dir = args.ckpt_dir
excluded_number = 2 if args.latest_two else 1

def remove_ckpt(ckpt_dir):
	ckpts = glob.glob(os.path.join(ckpt_dir, 'epoch_*'))
	epoch_nums = [int(ckpt.split('_')[-1]) for ckpt in ckpts]
	epoch_nums = sorted(epoch_nums)[:-excluded_number]
	for epoch_num in epoch_nums:
		epoch_path = os.path.join(ckpt_dir, f'epoch_{epoch_num}')
		shutil.rmtree(epoch_path)
		
if ckpt_dir is not None:
	remove_ckpt(ckpt_dir)
else:
	for ckpt_dir in glob.glob('*ViT-*'):
		remove_ckpt(ckpt_dir)