#!/usr/bin/env python3
"""Build the three frozen real-data rectangle relations."""
import argparse,json,pathlib,shutil
from vosma_dataset.common import sha256
from vosma_dataset import doc,mot,coco
SOURCES={"doclaynet":["doclaynet"],"mot20":["mot20"],"coco_sama":["coco2017","sama_train","sama_val"]}
def verify_sources(raw,names):
 lock_path=pathlib.Path(__file__).with_name("sources.lock.json")
 lock=json.loads(lock_path.read_text()) if lock_path.exists() else {}
 for name in names:
  p=raw/name/"source-manifest.json"
  if not p.exists():raise RuntimeError(f"Fetch is incomplete: {name}")
  manifest=json.loads(p.read_text())
  if lock:
   pinned=lock[name]
   for key in ("url","etag","archive_size"):
    if manifest[key]!=pinned[key]:raise RuntimeError(f"Source manifest does not match lock: {name}/{key}")
   expected={m["path"]:m["sha256"] for m in pinned["members"]}
   actual={m["path"]:m["sha256"] for m in manifest["members"]}
   if expected!=actual:raise RuntimeError(f"Source members do not match lock: {name}")
  for member in manifest["members"]:
   f=raw/member["path"]
   if sha256(f)!=member["sha256"]:raise RuntimeError(f"Source checksum mismatch: {member['path']}")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--raw-root",type=pathlib.Path,required=True);ap.add_argument("--output-root",type=pathlib.Path,required=True)
 ap.add_argument("--datasets",nargs="+",choices=list(SOURCES),default=list(SOURCES));args=ap.parse_args()
 for name in args.datasets:
  verify_sources(args.raw_root,SOURCES[name])
  {"doclaynet":doc.build,"mot20":mot.build,"coco_sama":coco.build}[name](args.raw_root,args.output_root)
  target=args.output_root/name/"sources";target.mkdir(exist_ok=True)
  for source in SOURCES[name]:shutil.copyfile(args.raw_root/source/"source-manifest.json",target/(source+".json"))
if __name__=="__main__":main()

