#!/usr/bin/env python3
"""Fetch a fixed DocLayNet page selection from the pinned official image archive."""
import argparse,collections,concurrent.futures,hashlib,io,json,pathlib,re,struct,threading,zipfile,zlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from PIL import Image
from fetch_sources import RangeFile
from vosma_dataset.common import load_json,sha256

def prepare(raw,out,splits,workers):
 out.mkdir(parents=True,exist_ok=True);image_dir=out/"images";image_dir.mkdir(exist_ok=True)
 lock=json.loads(pathlib.Path(__file__).with_name("sources.lock.json").read_text())["doclaynet"]
 pages=[];physical=set();names=set()
 for split in splits:
  source=raw/"doclaynet"/"COCO"/(split+".json")
  expected=next(m["sha256"] for m in lock["members"] if m["archive_member"]=="COCO/"+split+".json")
  if sha256(source)!=expected:raise RuntimeError("Annotation source differs from lock")
  v=load_json(source)
  for im in v["images"]:
   key=[im["doc_category"],im["collection"],im["doc_name"],im["page_no"]]
   identity=tuple(key);name=im["file_name"]
   if identity in physical or name in names:raise RuntimeError("Ambiguous physical page or file identity")
   if not re.fullmatch(r"[0-9a-f]+\.png",name):raise RuntimeError("Unexpected PNG filename")
   physical.add(identity);names.add(name)
   pages.append({"page_id":f'{split}/{im["id"]}',"split":split,"source_image_id":im["id"],"file_name":name,"width":im["width"],"height":im["height"],"physical_key":key})
 pages.sort(key=lambda p:(p["split"],p["source_image_id"]))
 page_doc={"schema_version":"doclaynet-pages-v1","splits":splits,"selection":"All images in each explicitly selected official split; no annotation-dependent selection","pages":pages}
 (out/"pages.json").write_text(json.dumps(page_doc,indent=2)+"\n")
 with RangeFile(lock["url"]) as remote,zipfile.ZipFile(remote) as z:
  if remote.etag!=lock["etag"] or remote.size!=lock["archive_size"]:raise RuntimeError("Image archive differs from pinned source")
  members={p["file_name"]:z.getinfo("PNG/"+p["file_name"]) for p in pages}
  archive_size=remote.size
 local=threading.local()
 def fetch(p):
  name=p["file_name"];m=members[name];dest=image_dir/name;cached=False
  if dest.exists():
   content=dest.read_bytes()
   cached=len(content)==m.file_size and zlib.crc32(content)==m.CRC
  if not cached:
   if not hasattr(local,"session"):
    local.session=requests.Session()
    local.session.mount("https://",HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
   start=m.header_offset;end=min(archive_size-1,start+m.compress_size+30+len(m.filename.encode())+65535-1)
   r=local.session.get(lock["url"],headers={"Range":f"bytes={start}-{end}","If-Match":lock["etag"],"Accept-Encoding":"identity"},timeout=(20,120))
   r.raise_for_status()
   if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{archive_size}":raise RuntimeError("Invalid PNG byte-range response")
   b=r.content;header=struct.unpack("<4s5H3I2H",b[:30]);signature,version,flags,method,mtime,mdate,crc,csize,usize,nlen,xlen=header
   if signature!=b"PK\x03\x04" or flags&1 or method!=m.compress_type:raise RuntimeError("Invalid ZIP member header")
   if b[30:30+nlen].decode("utf-8")!=m.filename:raise RuntimeError("ZIP member name mismatch")
   offset=30+nlen+xlen;compressed=b[offset:offset+m.compress_size]
   if len(compressed)!=m.compress_size:raise RuntimeError("Truncated PNG member")
   if method==zipfile.ZIP_DEFLATED:content=zlib.decompress(compressed,-15)
   elif method==zipfile.ZIP_STORED:content=compressed
   else:raise RuntimeError("Unsupported ZIP compression")
   if len(content)!=m.file_size or zlib.crc32(content)!=m.CRC:raise RuntimeError("PNG size/CRC mismatch")
   partial=dest.with_suffix(".partial");partial.write_bytes(content);partial.replace(dest)
  with Image.open(io.BytesIO(content)) as im:
   if im.size!=(p["width"],p["height"]):raise RuntimeError("PNG dimensions disagree with source")
   im.verify()
  return {"page_id":p["page_id"],"file_name":name,"archive_member":m.filename,"size":len(content),"compressed_size":m.compress_size,"crc32":f"{m.CRC:08x}","sha256":hashlib.sha256(content).hexdigest(),"width":p["width"],"height":p["height"]}
 results={};failures=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(fetch,p):p for p in pages}
  for future in concurrent.futures.as_completed(futures):
   p=futures[future]
   try:results[p["page_id"]]=future.result()
   except Exception as e:failures.append({"page_id":p["page_id"],"error":type(e).__name__+": "+str(e).split("?")[0]})
   done=len(results)+len(failures)
   if done%100==0 or done==len(pages):print("IMAGES",done,"/",len(pages),"failures",len(failures),flush=True)
 manifest={"source_url":lock["url"],"archive_etag":lock["etag"],"archive_size":archive_size,"pages_sha256":sha256(out/"pages.json"),"selected_pages":len(pages),"verified_images":len(results),"failures":failures,"members":[results[p["page_id"]] for p in pages if p["page_id"] in results]}
 (out/"images-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
 if failures:raise RuntimeError(f"{len(failures)} images failed; selection not silently shortened")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--raw-root",type=pathlib.Path,required=True);ap.add_argument("--output-root",type=pathlib.Path,required=True);ap.add_argument("--splits",nargs="+",choices=["train","val","test"],default=["test"]);ap.add_argument("--workers",type=int,default=8);a=ap.parse_args()
 if len(set(a.splits))!=len(a.splits):ap.error("Duplicate split selection")
 prepare(a.raw_root,a.output_root,a.splits,a.workers)
if __name__=="__main__":main()
