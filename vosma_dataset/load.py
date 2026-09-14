"""Load immutable published NumPy rectangle inputs without rebuilding annotations."""
import json,pathlib
import numpy as np
import pyarrow.parquet as pq
def load_local(directory):
 directory=pathlib.Path(directory)
 info=json.loads((directory/"dataset.json").read_text())
 if info["schema_version"]!="vosma-real-rectangles-v1":raise ValueError("Unsupported schema")
 r1=np.load(directory/"R1.npy",mmap_mode="r",allow_pickle=False)
 r2=np.load(directory/"R2.npy",mmap_mode="r",allow_pickle=False)
 groups=pq.read_table(directory/"groups.parquet")
 return r1,r2,groups,info
def download(dataset,revision,local_dir):
 if not revision:raise ValueError("Pin a Hugging Face commit revision")
 if dataset not in ("doclaynet","mot20","coco_sama"):raise ValueError("Unknown dataset")
 from huggingface_hub import snapshot_download
 root=snapshot_download(repo_id="DannHiroaki/VOSMA-Dataset",repo_type="dataset",revision=revision,allow_patterns=[f"{dataset}/*"],local_dir=local_dir)
 return load_local(pathlib.Path(root)/dataset)

