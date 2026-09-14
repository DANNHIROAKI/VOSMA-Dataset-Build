#!/usr/bin/env python3
"""Render deterministic local-coordinate overlays; no record pairs are saved."""
import argparse,json,pathlib
import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data-root",type=pathlib.Path,required=True);ap.add_argument("--output-root",type=pathlib.Path,required=True);ap.add_argument("--datasets",nargs="+",default=["doclaynet","mot20","coco_sama"]);ap.add_argument("--image-root",type=pathlib.Path);args=ap.parse_args()
 args.output_root.mkdir(parents=True,exist_ok=True)
 for name in args.datasets:
  base=args.data_root/name;rows=pq.read_table(base/"groups.parquet").to_pylist()
  eligible=[g for g in rows if g["r1_count"] and g["r2_count"] and g.get("overlap_count",0)>0]
  chosen=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(3,len(eligible)),dtype=int))]
  arrays=[np.load(base/(s+".npy"),mmap_mode="r") for s in ("R1","R2")]
  fig,axes=plt.subplots(1,len(chosen),figsize=(5*len(chosen),5.5),squeeze=False)
  for ax,g in zip(axes[0],chosen):
   attrs=json.loads(g["attributes"]);impath=args.image_root/attrs.get("file_name","missing") if args.image_root else None
   if name=="doclaynet" and impath and impath.is_file():
    ax.imshow(plt.imread(impath),extent=(0,g["width"],g["height"],0));background="Official page image"
   else:background="Geometry only; source image not shown"
   for a,side,color in zip(arrays,("r1","r2"),("#1570b8","#d96917")):
    boxes=a[g[side+"_start"]:g[side+"_start"]+g[side+"_count"],2:6]-np.array([g["dx"],g["dy"],g["dx"],g["dy"]])
    for x0,y0,x1,y1 in boxes/1000:
     ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,lw=.75,alpha=.65))
   ax.set_xlim(min(0,g["local_min_x"]/1000),max(g["width"],g["local_max_x"]/1000))
   ax.set_ylim(max(g["height"],g["local_max_y"]/1000),min(0,g["local_min_y"]/1000))
   ax.set_aspect("equal");ax.set_title(f'Group {g["group_id"]} | R1 {g["r1_count"]:,} / R2 {g["r2_count"]:,}\n{background}',fontsize=9)
   ax.set_xlabel("Local x (pixels)");ax.set_ylabel("Local y (pixels)")
  fig.suptitle(name+" — local rectangle overlay",fontsize=13)
  fig.legend(handles=[Patch(facecolor="none",edgecolor="#1570b8",label="R1"),Patch(facecolor="none",edgecolor="#d96917",label="R2")],loc="lower center",ncol=2,frameon=False)
  fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(args.output_root/(name+".png"),dpi=140);plt.close(fig)
  (args.output_root/(name+".json")).write_text(json.dumps({"groups":[g["group_id"] for g in chosen],"selection":"First, middle, last positive-overlap group in published order","image_background":bool(args.image_root)},indent=2)+"\n")
if __name__=="__main__":main()
