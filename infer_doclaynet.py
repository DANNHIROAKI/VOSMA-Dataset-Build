#!/usr/bin/env python3
"""Generate fixed model predictions without reading ground-truth annotations."""
import argparse,concurrent.futures,gzip,hashlib,importlib.metadata,io,json,math,os,pathlib,random,time
import numpy as np
from PIL import Image
import torch
from transformers import DeformableDetrConfig,DeformableDetrForObjectDetection,AutoImageProcessor

MODEL=None;PROCESSOR=None;CONFIG=None
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open("rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
def initialize(config):
 global MODEL,PROCESSOR,CONFIG
 CONFIG=config;torch.set_num_threads(config["threads"]);torch.set_num_interop_threads(1)
 random.seed(20260915);np.random.seed(20260915);torch.manual_seed(20260915);torch.use_deterministic_algorithms(True)
 cfg=DeformableDetrConfig.from_pretrained(config["model"],local_files_only=True)
 cfg.use_pretrained_backbone=False;cfg.disable_custom_kernels=True;cfg.use_timm_backbone=True
 MODEL,info=DeformableDetrForObjectDetection.from_pretrained(config["model"],config=cfg,local_files_only=True,use_safetensors=True,output_loading_info=True)
 if any(info.get(k) for k in ("missing_keys","unexpected_keys","mismatched_keys","error_msgs")):raise RuntimeError("Incomplete checkpoint load: "+str(info))
 MODEL=MODEL.to(device="cpu",dtype=torch.float32).eval()
 PROCESSOR=AutoImageProcessor.from_pretrained(config["model"],local_files_only=True)
def predict(page):
 out=pathlib.Path(CONFIG["output"])/"pages"/(page["page_id"].replace("/","__")+".json")
 image_path=pathlib.Path(CONFIG["images"])/page["file_name"]
 try:
  deadline=time.monotonic()+7200
  while not image_path.exists():
   if time.monotonic()>deadline:raise RuntimeError("Selected source image missing after download deadline")
   time.sleep(.5)
  image_sha=sha(image_path)
  if out.exists():
   prior=json.loads(out.read_text())
   if prior.get("status")=="success" and prior.get("protocol_sha256")==CONFIG["protocol_sha256"] and prior.get("image_sha256")==image_sha:return page["page_id"],True,prior["inference_seconds"]
  start=time.monotonic()
  with Image.open(image_path) as source:
   if source.size!=(page["width"],page["height"]):raise RuntimeError("Image dimensions differ from fixed page manifest")
   im=source.convert("RGB")
  inputs=PROCESSOR(images=im,return_tensors="pt")
  with torch.inference_mode():
   outputs=MODEL(**inputs)
   values,indices=torch.topk(outputs.logits.sigmoid().flatten(1),100,dim=1)
   values=values[0];indices=indices[0];labels=indices%outputs.logits.shape[-1];queries=indices//outputs.logits.shape[-1]
   native=PROCESSOR.post_process_object_detection(outputs,threshold=.7,top_k=100,target_sizes=torch.tensor([[page["height"],page["width"]]]))[0]
   finite=torch.isfinite(values);keep=values>.7;ranks=torch.nonzero(keep).flatten().tolist()
   assert len(ranks)==len(native["scores"])
   assert torch.equal(labels[keep],native["labels"]) and torch.equal(values[keep],native["scores"])
   predictions=[]
   for i,rank in enumerate(ranks):
    label=int(labels[rank])
    if not finite[rank] or label==0:continue
    coords=native["boxes"][i].tolist();coords=[v if math.isfinite(v) else str(v) for v in coords]
    predictions.append({"prediction_index":rank,"query_index":int(queries[rank]),"label":label,"score":float(native["scores"][i]),**dict(zip(("x0","y0","x1","y1"),coords))})
  result={"page_id":page["page_id"],"status":"success","protocol_sha256":CONFIG["protocol_sha256"],"image_sha256":image_sha,"width":page["width"],"height":page["height"],"model_input_shape":list(inputs["pixel_values"].shape),"top_k_candidates":100,"nonfinite_score_excluded":int((~finite).sum()),"threshold_excluded":int((finite&~keep).sum()),"background_excluded":int((finite&keep&(labels==0)).sum()),"predictions":predictions,"inference_seconds":time.monotonic()-start}
  assert sum(result[k] for k in ("nonfinite_score_excluded","threshold_excluded","background_excluded"))+len(predictions)==100
  if CONFIG.get("pilot"):
   np.savez_compressed(pathlib.Path(CONFIG["output"])/"pilot.npz",logits=outputs.logits.numpy(),boxes=outputs.pred_boxes.numpy())
   (pathlib.Path(CONFIG["output"])/"pilot.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
  tmp=out.with_suffix(".partial");tmp.write_text(json.dumps(result,separators=(",",":"),allow_nan=False)+"\n");tmp.replace(out)
  return page["page_id"],True,result["inference_seconds"]
 except Exception as e:
  failure={"page_id":page["page_id"],"status":"failed","protocol_sha256":CONFIG["protocol_sha256"],"error":type(e).__name__+": "+str(e)}
  out.write_text(json.dumps(failure)+"\n");return page["page_id"],False,0

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--pages",type=pathlib.Path,required=True);ap.add_argument("--images",type=pathlib.Path,required=True);ap.add_argument("--model",type=pathlib.Path,required=True);ap.add_argument("--model-manifest",type=pathlib.Path,required=True);ap.add_argument("--output-root",type=pathlib.Path,required=True);ap.add_argument("--workers",type=int,default=4);ap.add_argument("--threads",type=int,default=4);ap.add_argument("--pilot",action="store_true");a=ap.parse_args()
 a.output_root.mkdir(parents=True,exist_ok=True);(a.output_root/"pages").mkdir(exist_ok=True)
 model_manifest=json.loads(a.model_manifest.read_text())
 for name,m in model_manifest["files"].items():
  if sha(a.model/name)!=m["sha256"]:raise RuntimeError("Model file does not match manifest: "+name)
 pages_doc=json.loads(a.pages.read_text());pages=pages_doc["pages"]
 protocol={"schema_version":"doclaynet-predictions-v1","model":model_manifest,"page_manifest_sha256":sha(a.pages),"source_page_splits":pages_doc["splits"],"inference":{"device":"cpu","dtype":"float32","batch_size":1,"threads_per_worker":a.threads,"interop_threads":1,"seed":20260915,"deterministic_algorithms":True,"configuration_overrides":{"use_pretrained_backbone":False,"disable_custom_kernels":True,"use_timm_backbone":True}},"preprocessing":json.loads((a.model/"preprocessor_config.json").read_text()),"postprocessing":{"implementation":"DeformableDetrImageProcessor.post_process_object_detection","top_k":100,"top_k_scope":"Flattened 200 queries x 12 sigmoid class scores","confidence_threshold":.7,"threshold_operator":">","threshold_basis":"Fixed author model-card example; no GT-based tuning","excluded_label_ids":[0],"exclusion_order":["nonfinite_score","score_at_most_threshold","N/A_background"],"nms":False,"clipping":False,"deduplication":False,"coordinates":"float32 xyxy restored to each official PNG width/height before Q1000; no decimal printing round-off","prediction_index":"Original top-100 rank; can have gaps after filtering"},"id2label":json.loads((a.model/"config.json").read_text())["id2label"],"ground_truth_used_for_prediction":False,"software":{"python":os.sys.version.split()[0],**{n:importlib.metadata.version(n) for n in ("torch","torchvision","transformers","timm","safetensors","Pillow","numpy","tokenizers","huggingface-hub")}}}
 protocol_path=a.output_root/"protocol.json";protocol_path.write_text(json.dumps(protocol,indent=2)+"\n")
 config={"model":str(a.model.resolve()),"images":str(a.images.resolve()),"output":str(a.output_root.resolve()),"threads":a.threads,"protocol_sha256":sha(protocol_path),"pilot":a.pilot}
 begin=time.monotonic();failures=[]
 if a.pilot:
  initialize(config)
  chosen=next((p for p in pages if (a.images/p["file_name"]).exists()),pages[0])
  result=predict(chosen);print("PILOT",result,flush=True)
  if not result[1]:raise RuntimeError("Pilot inference failed; inspect output page status")
  return
 import multiprocessing
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context("spawn"),initializer=initialize,initargs=(config,)) as pool:
  futures={pool.submit(predict,p):p["page_id"] for p in pages}
  for i,future in enumerate(concurrent.futures.as_completed(futures),1):
   page_id,passed,seconds=future.result()
   if not passed:failures.append(page_id)
   if i%50==0 or i==len(pages):print("INFERENCE",i,"/",len(pages),"failures",len(failures),"elapsed",round(time.monotonic()-begin,1),flush=True)
 summary={"selected_pages":len(pages),"successful_pages":len(pages)-len(failures),"failed_pages":failures,"elapsed_seconds":time.monotonic()-begin,"workers":a.workers,"protocol_sha256":sha(protocol_path)}
 (a.output_root/"inference-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
 if failures:raise RuntimeError(f"{len(failures)} pages failed; not a valid completed dataset")
 with io.TextIOWrapper(gzip.GzipFile(filename=str(a.output_root/"predictions.jsonl.gz"),mode="wb",mtime=0),encoding="utf-8") as target:
  for page in pages:
   record=json.loads((a.output_root/"pages"/(page["page_id"].replace("/","__")+".json")).read_text())
   if record.get("status")!="success" or record.get("protocol_sha256")!=summary["protocol_sha256"]:raise RuntimeError("Invalid or stale inference record")
   record.pop("inference_seconds",None)
   target.write(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n")
 print("PREDICTIONS FROZEN",sha(a.output_root/"predictions.jsonl.gz"),flush=True)
if __name__=="__main__":main()
