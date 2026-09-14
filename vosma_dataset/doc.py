import collections
from .common import Dataset,load_json,dumps
def build(raw,out):
 d=Dataset("doclaynet",out);pages={};split_stats={};prec_global=collections.Counter()
 for split in ("train","val","test"):
  path=raw/"doclaynet"/"COCO"/(split+".json");v=load_json(path);source=path.relative_to(raw).as_posix()
  ims={im["id"]:im for im in v["images"]}
  if len(ims)!=len(v["images"]):raise ValueError("Duplicate image id in DocLayNet split")
  for im in v["images"]:
   key=(im["doc_category"],im["collection"],im["doc_name"],im["page_no"])
   if key not in pages:pages[key]={"images":[],"layers":collections.Counter(),"records":[],"layer_refs":collections.defaultdict(set)}
   pages[key]["images"].append({"split":split,"source_image_id":str(im["id"]),"file_name":im["file_name"],"width":im["width"],"height":im["height"],"image_precedence":im.get("precedence")})
  pc=collections.Counter()
  for ordinal,a in enumerate(v["annotations"]):
   d.stats["annotations.raw"]+=1
   if a["image_id"] not in ims:raise ValueError("Orphan DocLayNet annotation")
   im=ims[a["image_id"]];key=(im["doc_category"],im["collection"],im["doc_name"],im["page_no"]);page=pages[key]
   p=a.get("precedence")
   if type(p) is not int or p not in (0,1,2):
    d.reject("source","invalid_annotation_precedence",source,ordinal,annotation_id=str(a.get("id","")));continue
   pc[p]+=1;prec_global[p]+=1;page["layers"][p]+=1;page["layer_refs"][p].add((split,str(a["image_id"])))
   if p in (0,1):page["records"].append((p,a.get("bbox"),source,ordinal,a.get("id",""),a["image_id"],a["category_id"],im.get("precedence")))
  split_stats[split]={"images":len(v["images"]),"annotations":len(v["annotations"]),"annotation_precedence":dict(pc)}
  del v
 paired=0;identity_conflicts=[];ambiguous=[]
 for physical,page in sorted(pages.items()):
  has01=page["layers"][0]>0 and page["layers"][1]>0
  if not has01:
   d.stats["pages.without_both_annotation_layers"]+=1
   d.stats["annotations.outside_primary_duplicate_subset"]+=sum(page["layers"].values())
   continue
  paired+=1;images=page["images"];names=sorted({im["file_name"] for im in images});sizes={(im["width"],im["height"]) for im in images}
  if len(names)!=1 or len(sizes)!=1:
   identity_conflicts.append({"physical_key":physical,"images":images});continue
  if any(len(page["layer_refs"][p])>1 for p in (0,1)):
   ambiguous.append({"physical_key":physical,"layer_refs":{str(p):sorted(page["layer_refs"][p]) for p in (0,1)}});continue
  width,height=next(iter(sizes))
  key=d.group(list(physical),image_key=dumps(physical),split=",".join(sorted({im["split"] for im in images})),width=width,height=height,
   doc_category=physical[0],collection=physical[1],doc_name=physical[2],page_no=physical[3],file_name=names[0],image_refs=images,
   identity_check="Same official PNG filename and declared dimensions",raw_layer_counts=dict(page["layers"]))
  if page["layers"][2]:d.stats["pages.triple_annotated"]+=1
  else:d.stats["pages.double_annotated"]+=1
  d.stats["annotations.precedence_2_not_in_primary"]+=page["layers"][2]
  for p,bbox,source,ordinal,aid,iid,cid,imgp in page["records"]:
   side="r1" if p==0 else "r2";d.stats[f"{side}.raw"]+=1
   d.add(key,side,bbox,source,ordinal,annotation_id=aid,image_id=iid,category_id=cid,annotation_precedence=p,image_precedence=imgp)
  page["records"].clear()
 (d.out/"pairing_conflicts.json").write_text(dumps({"image_identity_conflicts":identity_conflicts,"layer_identity_ambiguous":ambiguous})+"\n")
 if identity_conflicts or ambiguous:raise RuntimeError(f"Unresolved DocLayNet pairing: identity={len(identity_conflicts)}, layers={len(ambiguous)}")
 audit={"source_physical_pages":len(pages),"splits":split_stats,"annotation_precedence":dict(prec_global),"paired_pages":paired,"filter_statistics":dict(d.stats)}
 (d.out/"source-audit.json").write_text(dumps(audit)+"\n")
 if not paired:
  d.excluded.close()
  raise RuntimeError("No original precedence 0/1 pages present; refuse substitute data")
 return d.finish({"source_roles":{"R1":"Original annotation precedence=0","R2":"Original annotation precedence=1"},
  "source_physical_pages":len(pages),"raw_paired_pages":paired,"reference_paired_pages_from_paper":8650,"splits":split_stats,
  "annotation_precedence":dict(prec_global),"license":"CDLA-Permissive-1.0"})

