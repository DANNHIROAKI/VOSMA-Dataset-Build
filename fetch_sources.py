#!/usr/bin/env python3
"""Download only annotation members from official ZIP/ZIP64 archives."""
import argparse, io, json, pathlib, hashlib, re, shutil, time, zipfile, zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOCK={}
SOURCES={
 "doclaynet":{"url":"https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip","pattern":r"(^|/)COCO/(train|val|test)\.json$","license":"CDLA-Permissive-1.0","version":"1.0.0"},
 "mot20":{"url":"https://motchallenge.net/data/MOT20Labels.zip","pattern":r"(^|/)train/MOT20-(01|02|03|05)/(gt/gt\.txt|det/det\.txt|seqinfo\.ini)$","license":"CC-BY-NC-SA-3.0","version":"official-2020"},
 "coco2017":{"url":"https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip","pattern":r"(^|/)instances_(train|val)2017\.json$","license":"CC-BY-4.0","version":"2017"},
 "sama_train":{"url":"https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-train.zip","pattern":r"\.json$","license":"CC-BY-4.0","version":"official-2022"},
 "sama_val":{"url":"https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip","pattern":r"\.json$","license":"CC-BY-4.0","version":"official-2022"},
}
class RangeFile(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.pos=0;self.cache_start=0;self.cache=b""
  self.session=requests.Session()
  self.session.mount("https://",HTTPAdapter(max_retries=Retry(total=6,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
  with self.session.get(url,headers={"Range":"bytes=0-0","Accept-Encoding":"identity"},stream=True,timeout=(20,90)) as r:
   if r.status_code!=206:raise RuntimeError(f"Range unsupported: {r.status_code} {url}")
   m=re.fullmatch(r"bytes 0-0/(\d+)",r.headers.get("Content-Range",""))
   if not m:raise RuntimeError("Invalid initial Content-Range")
   self.size=int(m[1]);self.etag=r.headers.get("ETag");self.modified=r.headers.get("Last-Modified")
   if len(r.content)!=1:raise RuntimeError("Invalid initial byte response")
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  p=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if p<0:raise ValueError("Negative seek")
  self.pos=p;return p
 def read(self,n=-1):
  n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
  if n<=0:return b""
  if n>1024**3:raise RuntimeError("Unexpected >1 GiB single ZIP range")
  begin=self.pos;end=begin+n-1
  if self.cache_start<=begin and end<self.cache_start+len(self.cache):
   self.pos+=n;return self.cache[begin-self.cache_start:begin-self.cache_start+n]
  headers={"Range":f"bytes={begin}-{end}","Accept-Encoding":"identity"}
  if self.etag:headers["If-Match"]=self.etag
  with self.session.get(self.url,headers=headers,stream=True,timeout=(20,120)) as r:
   if r.status_code!=206:raise RuntimeError(f"Range failed {r.status_code}")
   if r.headers.get("Content-Range")!=f"bytes {begin}-{end}/{self.size}":raise RuntimeError("Mismatched range")
   data=r.content
   if len(data)!=n:raise RuntimeError("Truncated range")
  self.pos+=n;return data
 def close(self):
  if hasattr(self,"session"):self.session.close()
  super().close()
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024**2),b""):h.update(chunk)
 return h.hexdigest()
def crc32(path):
 v=0
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):v=zlib.crc32(b,v)
 return v
def fetch(name,root):
 spec=SOURCES[name];out=root/name;out.mkdir(parents=True,exist_ok=True)
 with RangeFile(spec["url"]) as remote,zipfile.ZipFile(remote) as z:
  locked=LOCK.get(name)
  if locked and any(locked[k]!=v for k,v in (("url",spec["url"]),("etag",remote.etag),("archive_size",remote.size))):raise RuntimeError(f"{name}: source archive differs from the pinned lock")
  catalog=[{"name":x.filename,"size":x.file_size,"compressed_size":x.compress_size,"crc32":f"{x.CRC:08x}"} for x in z.infolist()]
  (out/"archive-members.json").write_text(json.dumps(catalog,indent=2)+"\n")
  selected=[x for x in z.infolist() if not x.is_dir() and re.search(spec["pattern"],x.filename)]
  if not selected:raise RuntimeError(f"{name}: no matching members, inspect archive-members.json")
  if locked and {x.filename for x in selected}!={x["archive_member"] for x in locked["members"]}:raise RuntimeError(f"{name}: archive member set differs from lock")
  manifest={**spec,"archive_size":remote.size,"etag":remote.etag,"last_modified":remote.modified,"retrieved_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"members":[]}
  for member in selected:
   rel=pathlib.PurePosixPath(member.filename)
   if rel.is_absolute() or ".." in rel.parts:raise RuntimeError("Unsafe ZIP member")
   dest=out/pathlib.Path(*rel.parts);dest.parent.mkdir(parents=True,exist_ok=True)
   if not dest.exists() or dest.stat().st_size!=member.file_size or crc32(dest)!=member.CRC:
    remote.cache=b""
    remote.seek(member.header_offset)
    remote.cache_start=member.header_offset
    remote.cache=remote.read(min(member.compress_size+65565,remote.size-member.header_offset))
    partial=dest.with_name(dest.name+".partial")
    with z.open(member) as src,partial.open("wb") as target:
     shutil.copyfileobj(src,target,8*1024**2)
    if partial.stat().st_size!=member.file_size:raise RuntimeError("Extracted size mismatch")
    partial.replace(dest)
   member_sha=digest(dest)
   if locked:
    expected={x["archive_member"]:x["sha256"] for x in locked["members"]}
    if expected.get(member.filename)!=member_sha:raise RuntimeError(f"{name}: extracted member checksum differs from lock")
   manifest["members"].append({"path":dest.relative_to(root).as_posix(),"archive_member":member.filename,"size":member.file_size,"compressed_size":member.compress_size,"crc32":f"{member.CRC:08x}","sha256":member_sha})
   print(name,member.filename,member.file_size,flush=True)
  (out/"source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 return name
def main():
 global LOCK
 ap=argparse.ArgumentParser();ap.add_argument("--raw-root",type=pathlib.Path,required=True);ap.add_argument("--sources",nargs="+",default=list(SOURCES));ap.add_argument("--lock",type=pathlib.Path,default=pathlib.Path(__file__).with_name("sources.lock.json"));args=ap.parse_args()
 if args.lock.exists():LOCK=json.loads(args.lock.read_text())
 with ThreadPoolExecutor(max_workers=3) as pool:
  tasks={pool.submit(fetch,n,args.raw_root):n for n in args.sources}
  failures=[]
  for f in as_completed(tasks):
   try:print("DONE",f.result(),flush=True)
   except Exception as e:failures.append(tasks[f]);print("FAILED",tasks[f],repr(e),flush=True)
 if failures:raise SystemExit("Failed sources: "+", ".join(failures))
if __name__=="__main__":main()

