import unittest,tempfile,pathlib,json
import numpy as np
from vosma_dataset.doc import build
class DocLayNetTests(unittest.TestCase):
 def inputs(self,root,paired):
  base=root/"raw"/"doclaynet"/"COCO";base.mkdir(parents=True)
  im={"id":1,"width":10,"height":10,"file_name":"same.png","doc_category":"manuals","collection":"c","doc_name":"book.pdf","page_no":1,"precedence":0}
  anns=[{"id":1,"image_id":1,"category_id":1,"bbox":[0,0,5,5],"precedence":0}]
  if paired:
   anns += [{"id":2,"image_id":1,"category_id":2,"bbox":[1,1,5,5],"precedence":1},{"id":3,"image_id":1,"category_id":1,"bbox":[2,2,5,5],"precedence":2}]
  for split in ("train","val","test"):
   (base/(split+".json")).write_text(json.dumps({"images":[im] if split=="train" else [],"annotations":anns if split=="train" else []}))
 def test_annotation_layers_override_image_marker(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);self.inputs(root,True);info=build(root/"raw",root/"out")
   self.assertEqual((info["r1_count"],info["r2_count"],info["group_count"]),(1,1,1))
   self.assertEqual(info["filter_statistics"]["annotations.precedence_2_not_in_primary"],1)
   self.assertEqual(info["filter_statistics"]["pages.triple_annotated"],1)
 def test_missing_layers_refuses_substitute(self):
  with tempfile.TemporaryDirectory() as td:
   root=pathlib.Path(td);self.inputs(root,False)
   with self.assertRaisesRegex(RuntimeError,"No original precedence"):build(root/"raw",root/"out")
   self.assertFalse((root/"out"/"doclaynet"/"R1.npy").exists())
if __name__=="__main__":unittest.main()
