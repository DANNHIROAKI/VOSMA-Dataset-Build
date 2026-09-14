from __future__ import annotations
import collections, gzip, hashlib, json, pathlib, io
from decimal import Decimal
from fractions import Fraction
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

Q=1000
COLUMNS=["record_id","group_id","x0","y0","x1","y1","weight"]
I64MAX=2**63-1
def dumps(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),default=str)
def load_json(p):
 with pathlib.Path(p).open() as f:return json.load(f,parse_float=Decimal,parse_constant=Decimal)
def round_even(v):
 n,d=v.numerator*Q,v.denominator
 q,r=divmod(n,d)
 return q+int(2*r>d or (2*r==d and q%2))
def quantize(bbox):
 if not isinstance(bbox,(list,tuple)) or len(bbox)!=4:return None,"malformed_bbox"
 try:v=[x if isinstance(x,Decimal) else Decimal(str(x)) for x in bbox]
 except Exception:return None,"malformed_bbox"
 if not all(x.is_finite() for x in v):return None,"nonfinite_geometry"
 if v[2]<=0 or v[3]<=0:return None,"nonpositive_extent"
 x,y,w,h=map(Fraction,v)
 b=tuple(round_even(z) for z in (x,y,x+w,y+h))
 if b[0]>=b[2] or b[1]>=b[3]:return None,"quantized_degenerate"
 if any(not -(2**63)<=k<=I64MAX for k in b):raise OverflowError("Local rectangle exceeds int64")
 return b,None
def sha256(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
META_SCHEMA=pa.schema([
 ("record_id",pa.int64()),("group_id",pa.int64()),("source_file",pa.string()),
 ("source_record_index",pa.int64()),("source_annotation_id",pa.string()),("source_image_id",pa.string()),
 ("source_category_id",pa.string()),("local_x0",pa.int64()),("local_y0",pa.int64()),
 ("local_x1",pa.int64()),("local_y1",pa.int64()),("source_bbox",pa.string()),("attributes",pa.string())])
GROUP_SCHEMA=pa.schema([
 ("group_id",pa.int64()),("semantic_key",pa.string()),("image_key",pa.string()),("split",pa.string()),
 ("width",pa.int64()),("height",pa.int64()),("category",pa.string()),("dx",pa.int64()),("dy",pa.int64()),
 ("local_min_x",pa.int64()),("local_min_y",pa.int64()),("local_max_x",pa.int64()),("local_max_y",pa.int64()),
 ("global_min_x",pa.int64()),("global_max_x",pa.int64()),("r1_start",pa.int64()),("r1_count",pa.int64()),
 ("r2_start",pa.int64()),("r2_count",pa.int64()),("state",pa.string()),("attributes",pa.string())])
class ParquetBatches:
 def __init__(self,path,schema):self.writer=pq.ParquetWriter(path,schema,compression="zstd");self.rows=[];self.schema=schema
 def add(self,row):
  self.rows.append(row)
  if len(self.rows)>=20000:self.flush()
 def flush(self):
  if self.rows:self.writer.write_table(pa.Table.from_pylist(self.rows,schema=self.schema));self.rows.clear()
 def close(self):self.flush();self.writer.close()
class Dataset:
 def __init__(self,name,out):
  self.name=name;self.out=pathlib.Path(out)/name;self.out.mkdir(parents=True,exist_ok=True)
  self.groups={};self.stats=collections.Counter();self.excluded=io.TextIOWrapper(gzip.GzipFile(filename=str(self.out/"exclusions.jsonl.gz"),mode="wb",mtime=0),encoding="utf-8")
  self.images=[]
 def group(self,key,**meta):
  key=dumps(key)
  if key not in self.groups:self.groups[key]={"meta":meta,"r1":[],"r2":[]}
  return key
 def reject(self,side,reason,source,index,**extra):
  self.stats[f"{side}.excluded.{reason}"]+=1
  self.excluded.write(dumps(dict(side=side,reason=reason,source_file=source,source_record_index=index,**extra))+"\n")
 def add(self,key,side,bbox,source,index,annotation_id="",image_id="",category_id="",**attrs):
  self.stats[f"{side}.geometry_candidates"]+=1
  b,reason=quantize(bbox)
  if reason:self.reject(side,reason,source,index,group=key,annotation_id=str(annotation_id),bbox=bbox);return False
  g=self.groups[key];m=g["meta"]
  if min(b[0],b[1])<0:self.stats[f"{side}.negative_local_coordinate"]+=1
  origin=m.get("pixel_origin",0)
  if b[0]<origin*Q or b[1]<origin*Q or b[2]>(m["width"]+origin)*Q or b[3]>(m["height"]+origin)*Q:self.stats[f"{side}.out_of_bounds_retained"]+=1
  meta={"source_file":source,"source_record_index":index,"source_annotation_id":str(annotation_id),
   "source_image_id":str(image_id),"source_category_id":str(category_id),
   "source_bbox":dumps([str(x) for x in bbox]),"attributes":dumps(attrs)}
  g[side].append((b,meta));self.stats[f"{side}.retained"]+=1
  return True
 def finish(self,extra=None):
  self.excluded.close()
  totals={s:sum(len(g[s]) for g in self.groups.values()) for s in ("r1","r2")}
  arrays={s:np.lib.format.open_memmap(self.out/(s.upper()+".npy"),mode="w+",dtype="<i8",shape=(totals[s],7)) for s in totals}
  meta={s:ParquetBatches(self.out/(s.upper()+"_metadata.parquet"),META_SCHEMA) for s in totals}
  gw=ParquetBatches(self.out/"groups.parquet",GROUP_SCHEMA)
  pos={"r1":0,"r2":0};cursor=1;states=collections.Counter();naive=0
  for gid,(key,g) in enumerate(sorted(self.groups.items())):
   m=g["meta"];counts={s:len(g[s]) for s in totals}
   boxes=[v[0] for s in totals for v in g[s]]
   if boxes:
    bounds=(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))
    dx,dy=cursor-bounds[0],1-bounds[1];lo,hi=cursor,cursor+bounds[2]-bounds[0]
    if max(hi,1+bounds[3]-bounds[1],abs(dx),abs(dy))>I64MAX:raise OverflowError("Packed coordinates exceed int64")
    cursor=hi+1
   else:bounds=(0,0,0,0);dx=dy=0;lo=hi=cursor
   state="both_nonempty" if counts["r1"] and counts["r2"] else "r1_only" if counts["r1"] else "r2_only" if counts["r2"] else "both_empty"
   states[state]+=1;naive+=counts["r1"]*counts["r2"]
   row=dict(group_id=gid,semantic_key=key,image_key=m.get("image_key",""),split=m.get("split",""),
    width=m["width"],height=m["height"],category=m.get("category",""),dx=dx,dy=dy,
    local_min_x=bounds[0],local_min_y=bounds[1],local_max_x=bounds[2],local_max_y=bounds[3],
    global_min_x=lo,global_max_x=hi,r1_start=pos["r1"],r1_count=counts["r1"],r2_start=pos["r2"],r2_count=counts["r2"],state=state,
    attributes=dumps({k:v for k,v in m.items() if k not in ("width","height","category","image_key","split")}))
   gw.add(row)
   for side in totals:
    for b,md in g[side]:
     rid=pos[side];arr=(rid,gid,b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy,1)
     arrays[side][rid]=arr
     meta[side].add(dict(record_id=rid,group_id=gid,local_x0=b[0],local_y0=b[1],local_x1=b[2],local_y1=b[3],**md))
     pos[side]+=1
    g[side].clear()
  for a in arrays.values():a.flush()
  for w in meta.values():w.close()
  gw.close()
  if self.images:pq.write_table(pa.Table.from_pylist(self.images),self.out/"images.parquet",compression="zstd")
  info={"schema_version":"vosma-real-rectangles-v1","dataset":self.name,"quantization_scale":Q,"quantization":"Exact source decimal endpoints; nearest integer, ties to even",
   "columns":COLUMNS,"dtype":"little-endian int64","rectangle_semantics":"half-open","record_weights":"unit",
   "group_packing":"Shared R1/R2 integer translation; disjoint x slabs with a one-unit gap, based on retained box extents",
   "r1_count":totals["r1"],"r2_count":totals["r2"],"group_count":len(self.groups),"group_states":dict(states),
   "within_group_cartesian_pairs":naive,"filter_statistics":dict(sorted(self.stats.items())),
   "no_algorithm_index":True,"no_pair_matching":True,**(extra or {})}
  (self.out/"dataset.json").write_text(json.dumps(info,ensure_ascii=False,indent=2,default=str)+"\n")
  print(self.name,totals,"groups",len(self.groups),flush=True)
  return info

