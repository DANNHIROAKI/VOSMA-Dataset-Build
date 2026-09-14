#!/usr/bin/env python3
"""Validate every record and count within-group overlaps without materializing pairs."""
import argparse,ctypes,json,pathlib,subprocess,collections
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from vosma_dataset.common import sha256,COLUMNS

def load_counter(libpath):
 lib=ctypes.CDLL(str(libpath.resolve()))
 fn=lib.count_overlaps;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64,ctypes.c_int64,ctypes.POINTER(ctypes.c_uint64)];fn.restype=ctypes.c_int
 return fn
def check_counter(fn):
 rng=np.random.default_rng(42017)
 for _ in range(120):
  arrays=[]
  for side in range(2):
   n=int(rng.integers(0,18));a=np.zeros((n,7),dtype="<i8")
   a[:,2:4]=rng.integers(-5,6,size=(n,2));a[:,4:6]=a[:,2:4]+rng.integers(1,5,size=(n,2));a[:,6]=1;arrays.append(a)
  a,b=arrays;result=ctypes.c_uint64()
  assert fn(a.ctypes.data,b.ctypes.data,len(a),len(b),ctypes.byref(result))==0
  brute=sum(max(x[2],y[2])<min(x[4],y[4]) and max(x[3],y[3])<min(x[5],y[5]) for x in a for y in b)
  assert result.value==brute
def validate(path,fn):
 info=json.loads((path/"dataset.json").read_text());table=pq.read_table(path/"groups.parquet")
 g={name:np.asarray(table[name]) for name in ("group_id","dx","dy","global_min_x","global_max_x","r1_start","r1_count","r2_start","r2_count")}
 n=len(table);assert np.array_equal(g["group_id"],np.arange(n))
 arrays={}
 for side in ("r1","r2"):
  a=np.load(path/(side.upper()+".npy"),mmap_mode="r",allow_pickle=False);arrays[side]=a
  assert a.dtype==np.dtype("<i8") and a.shape==(info[side+"_count"],7)
  assert np.array_equal(a[:,0],np.arange(len(a))) and np.all(a[:,6]==1)
  assert np.all(a[:,2:4]>=0) and np.all(a[:,2]<a[:,4]) and np.all(a[:,3]<a[:,5])
  assert np.all(a[:,1]>=0) and np.all(a[:,1]<n)
  starts,counts=g[side+"_start"],g[side+"_count"]
  assert np.array_equal(starts,np.r_[0,np.cumsum(counts)[:-1]]) and counts.sum()==len(a)
  expected_ids=np.repeat(np.arange(n),counts);assert np.array_equal(a[:,1],expected_ids)
  offset=0
  for batch in pq.ParquetFile(path/(side.upper()+"_metadata.parquet")).iter_batches(batch_size=50000):
   m=batch.to_pydict();count=len(m["record_id"]);sub=a[offset:offset+count];ids=np.asarray(m["group_id"],dtype=np.int64)
   assert np.array_equal(m["record_id"],sub[:,0]) and np.array_equal(ids,sub[:,1])
   local=np.column_stack([m[k] for k in ("local_x0","local_y0","local_x1","local_y1")])
   shift=np.column_stack([g["dx"][ids],g["dy"][ids],g["dx"][ids],g["dy"][ids]])
   assert np.array_equal(local+shift,sub[:,2:6])
   assert np.all(sub[:,2]>=g["global_min_x"][ids]) and np.all(sub[:,4]<=g["global_max_x"][ids])
   assert np.array_equal(local[:,2:4]-local[:,0:2],sub[:,4:6]-sub[:,2:4])
   offset+=count
  assert offset==len(a)
  stats=info["filter_statistics"];excluded=sum(v for k,v in stats.items() if k.startswith(side+".excluded."))
  assert stats[side+".raw"]==len(a)+excluded
 occupied=(g["r1_count"]+g["r2_count"])>0
 assert np.all(g["global_max_x"][occupied][:-1]<g["global_min_x"][occupied][1:])
 overlaps=np.zeros(n,dtype=np.uint64)
 for i in range(n):
  if not g["r1_count"][i] or not g["r2_count"][i]:continue
  result=ctypes.c_uint64()
  pa_=arrays["r1"].ctypes.data+int(g["r1_start"][i])*56;pb_=arrays["r2"].ctypes.data+int(g["r2_start"][i])*56
  code=fn(pa_,pb_,int(g["r1_count"][i]),int(g["r2_count"][i]),ctypes.byref(result))
  if code:raise RuntimeError(f"Overlap counter failed {code} at group {i}")
  overlaps[i]=result.value
 if "overlap_count" in table.column_names:table=table.drop(["overlap_count"])
 pq.write_table(table.append_column("overlap_count",pa.array(overlaps)),path/"groups.parquet",compression="zstd")
 sample_indices=np.flatnonzero(overlaps>0)
 rng=np.random.default_rng(20260915)
 selected=np.sort(rng.choice(sample_indices,size=min(20,len(sample_indices)),replace=False))
 sampled_ious=[];sampled_areas=[];shared_geometry=0
 for i in selected:
  aa=arrays["r1"][g["r1_start"][i]:g["r1_start"][i]+g["r1_count"][i],2:6]
  bb=arrays["r2"][g["r2_start"][i]:g["r2_start"][i]+g["r2_count"][i],2:6]
  ca=collections.Counter(map(tuple,aa));cb=collections.Counter(map(tuple,bb));shared_geometry+=sum((ca&cb).values())
  kept=0
  for start in range(0,len(aa),64):
   a=aa[start:start+64];x=np.maximum(0,np.minimum(a[:,None,2],bb[None,:,2])-np.maximum(a[:,None,0],bb[None,:,0]))
   y=np.maximum(0,np.minimum(a[:,None,3],bb[None,:,3])-np.maximum(a[:,None,1],bb[None,:,1]))
   inter=x*y;mask=inter>0
   if not mask.any():continue
   ar=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);br=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1])
   union=ar[:,None]+br[None,:]-inter
   vals=(inter[mask]/union[mask])[:max(0,5000-kept)];areas=inter[mask][:len(vals)]
   assert np.all(vals>0) and np.all(vals<=1)
   sampled_ious.extend(vals.tolist());sampled_areas.extend((areas/1e6).tolist());kept+=len(vals)
   if kept>=5000:break
 distribution=lambda values:dict(zip(("min","median","p95","max"),map(float,np.quantile(values,[0,.5,.95,1])))) if len(values) else {}
 validation={"passed":True,"every_record_checked":True,"cross_group_intersections":0,"within_group_overlap_count":int(sum(map(int,overlaps))),
  "groups_with_positive_mass":int(np.count_nonzero(overlaps)),"groups_with_zero_mass":int(np.count_nonzero(overlaps==0)),
  "counter_randomized_bruteforce_cases":120,"r1_group_sizes":distribution(g["r1_count"]),"r2_group_sizes":distribution(g["r2_count"]),
  "spot_check":{"seed":20260915,"groups":selected.tolist(),"positive_pairs_examined":len(sampled_ious),"scope":"Deterministic geometric spot checks; not a population distribution estimate",
   "intersection_area_pixel_squared":distribution(sampled_areas),"iou":distribution(sampled_ious),"shared_exact_geometry_occurrences_in_selected_groups":shared_geometry},
  "numeric_range":{"max_coordinate":max(int(a[:,2:6].max()) for a in arrays.values() if len(a)),"unit_weights":True,
   "note":"Algorithm-specific area and cumulative-mass overflow checks remain required."}}
 (path/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
 hashes={p.relative_to(path).as_posix():sha256(p) for p in sorted(path.rglob("*")) if p.is_file() and p.name not in ("SHA256SUMS","files.json")}
 (path/"SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name,h in hashes.items()))
 print(path.name,"validated",info["r1_count"],info["r2_count"],"overlaps",validation["within_group_overlap_count"],flush=True)
 return validation
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--data-root",type=pathlib.Path,required=True);ap.add_argument("--work-root",type=pathlib.Path,required=True);ap.add_argument("--datasets",nargs="+",default=["doclaynet","mot20","coco_sama"]);args=ap.parse_args()
 args.work_root.mkdir(parents=True,exist_ok=True);lib=args.work_root/"overlap_count.so";src=pathlib.Path(__file__).with_name("overlap_count.cpp")
 subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",str(src),"-o",str(lib)],check=True)
 fn=load_counter(lib);check_counter(fn)
 for name in args.datasets:validate(args.data_root/name,fn)
if __name__=="__main__":main()

