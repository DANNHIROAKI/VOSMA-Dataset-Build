import collections,pathlib
from .common import Dataset,load_json,dumps
def identity(im):
 name=pathlib.PurePosixPath(im["file_name"]).name
 if not name or name in (".",".."):raise ValueError("Invalid image file identity")
 return name
def catmap(v):
 result={int(c["id"]):str(c["name"]) for c in v["categories"]}
 if len(result)!=len(v["categories"]) or len(set(result.values()))!=len(result):raise ValueError("Ambiguous category mapping")
 return result
def build(raw,out):
 d=Dataset("coco_sama",out);split_stats={};mapping={}
 for split,expected in (("train",118287),("val",5000)):
  coco_path=raw/"coco2017"/"annotations"/f"instances_{split}2017.json"
  sama_dir=raw/("sama_train" if split=="train" else "sama_val")
  if not (sama_dir/"source-manifest.json").exists():raise RuntimeError("Sama download is incomplete")
  manifest=load_json(sama_dir/"source-manifest.json")
  files=sorted(raw/m["path"] for m in manifest["members"])
  if set(files)!=set(sama_dir.glob("sama*.json")):raise ValueError("Sama shard set differs from verified source manifest")
  cv=load_json(coco_path);cc=catmap(cv)
  ci={int(im["id"]):im for im in cv["images"]}
  if len(ci)!=len(cv["images"]):raise ValueError("Duplicate COCO image id")
  c_names={identity(im):im for im in cv["images"]}
  if len(c_names)!=len(ci):raise ValueError("Duplicate COCO file identity")
  s_names={};s_cat=None;repeat_image_entries=0
  for p in files:
   v=load_json(p);cats=catmap(v)
   if s_cat is None:s_cat=cats
   if cats!=s_cat:raise ValueError("Sama category maps differ between shards")
   local_ids=set()
   for im in v["images"]:
    if im["id"] in local_ids:raise ValueError("Duplicate image id within Sama shard")
    local_ids.add(im["id"]);name=identity(im)
    if name in s_names:
     prior=s_names[name]
     if (prior["width"],prior["height"],prior["id"])!=(im["width"],im["height"],im["id"]):raise ValueError("Conflicting Sama image identity")
     repeat_image_entries+=1
    else:s_names[name]=im
  if set(cc.values())!=set(s_cat.values()) or len(cc)!=80:raise ValueError("COCO/Sama class names do not agree")
  common=set(c_names)&set(s_names)
  missing_s=sorted(set(c_names)-set(s_names));missing_c=sorted(set(s_names)-set(c_names))
  d.stats["images.missing_in_sama"]+=len(missing_s);d.stats["images.missing_in_coco"]+=len(missing_c)
  (d.out/f"{split}_image_pairing.json").write_text(dumps({"missing_in_sama":missing_s,"missing_in_coco":missing_c})+"\n")
  if missing_s or missing_c or len(common)!=expected:raise ValueError(f"Image coverage mismatch in {split}: {len(common)}")
  image_counts={n:collections.Counter() for n in common}
  for n in common:
   a,b=c_names[n],s_names[n]
   if (a["width"],a["height"])!=(b["width"],b["height"]):raise ValueError("COCO/Sama image dimensions disagree")
   for im in (a,b):
    if "coco_url" in im and identity({"file_name":im["coco_url"]})!=n:raise ValueError("File identity disagrees with source COCO URL")
  def process(v,side,p,cats):
   images={int(im["id"]):im for im in v["images"]};source=p.relative_to(raw).as_posix()
   for ordinal,a in enumerate(v["annotations"]):
    d.stats[f"{side}.raw"]+=1
    if int(a["image_id"]) not in images:raise ValueError("Orphan annotation image")
    im=images[int(a["image_id"])];name=identity(im)
    if name not in common:raise ValueError("Unpaired annotation image")
    cid=int(a["category_id"])
    if cid not in cats:raise ValueError("Unknown category")
    category=cats[cid];image_counts[name][side+"_raw"]+=1
    key=d.group([split,name,category],image_key=f"{split}/{name}",split=split,width=int(im["width"]),height=int(im["height"]),category=category,file_name=name,coco_image_id=str(c_names[name]["id"]),sama_image_id=str(s_names[name]["id"]))
    crowd=a.get("iscrowd")
    if crowd not in (0,1):d.reject(side,"invalid_iscrowd",source,ordinal,annotation_id=str(a.get("id","")));continue
    if crowd==1:d.reject(side,"crowd",source,ordinal,annotation_id=str(a.get("id","")));continue
    if d.add(key,side,a.get("bbox"),source,ordinal,annotation_id=a.get("id",""),image_id=a["image_id"],category_id=cid,iscrowd=crowd,category=category):
     image_counts[name][side+"_retained"]+=1
  process(cv,"r1",coco_path,cc)
  del cv
  for p in files:process(load_json(p),"r2",p,s_cat)
  for name in sorted(common):
   n=image_counts[name];a,b=c_names[name],s_names[name]
   d.images.append(dict(image_key=f"{split}/{name}",split=split,file_name=name,width=int(a["width"]),height=int(a["height"]),r1_image_id=str(a["id"]),r2_image_id=str(b["id"]),
    r1_raw=n["r1_raw"],r2_raw=n["r2_raw"],r1_retained=n["r1_retained"],r2_retained=n["r2_retained"],
    no_raw_annotations_on_both_sides=not(n["r1_raw"] or n["r2_raw"]),no_retained_boxes_on_both_sides=not(n["r1_retained"] or n["r2_retained"])))
  split_stats[split]={"common_images":len(common),"sama_json_shards":len(files),"repeated_sama_image_metadata_entries":repeat_image_entries,
    "no_raw_annotations_on_both_sides":sum(not(n["r1_raw"] or n["r2_raw"]) for n in image_counts.values()),
    "no_retained_boxes_on_both_sides":sum(not(n["r1_retained"] or n["r2_retained"]) for n in image_counts.values())}
  mapping[split]={"coco":cc,"sama":s_cat}
 (d.out/"category_mapping.json").write_text(dumps(mapping)+"\n")
 return d.finish({"source_roles":{"R1":"COCO 2017 non-crowd instance bounding boxes","R2":"Sama-COCO non-crowd instance bounding boxes"},
  "splits":split_stats,"group_universe":"Union of source image-category mentions before filtering; annotation-free images retained separately in images.parquet",
  "comparison":"Annotation-version comparison, not independent blind annotation and not object matching","license":"CC-BY-4.0"})

