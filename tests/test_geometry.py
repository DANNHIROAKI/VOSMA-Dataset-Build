import tempfile,pathlib,unittest,ctypes,subprocess
from decimal import Decimal
from fractions import Fraction
import numpy as np
import pyarrow.parquet as pq
from vosma_dataset.common import quantize,round_even,Dataset
class GeometryTests(unittest.TestCase):
 def test_ties_and_negative(self):
  for x,want in [("0.0005",0),("0.0015",2),("-0.0005",0),("-0.0015",-2),("-0.0025",-2)]:
   self.assertEqual(round_even(Fraction(Decimal(x))),want)
 def test_endpoints_before_rounding(self):
  b,e=quantize(["0.0006","-0.0006","0.0006","0.002"])
  self.assertIsNone(b);self.assertEqual(e,"quantized_degenerate")
 def test_endpoint_sum(self):
  b,e=quantize(["0.0006","0","0.0016","1"])
  self.assertIsNone(e);self.assertEqual(b,(1,0,2,1000))
 def test_filtering(self):
  for b,reason in [(["nan",0,1,1],"nonfinite_geometry"),([0,0,0,1],"nonpositive_extent"),([0,0,".0001",1],"quantized_degenerate"),([0,0,1],"malformed_bbox")]:
   self.assertEqual(quantize(b)[1],reason)
 def test_one_based_bounds(self):
  with tempfile.TemporaryDirectory() as td:
   d=Dataset("mot_origin",td);key=d.group(["frame"],width=1920,height=1080,pixel_origin=1)
   d.add(key,"r1",[1,1,1920,1080],"source",1)
   self.assertEqual(d.stats["r1.out_of_bounds_retained"],0)
   d.excluded.close()
 def test_group_translation_and_duplicates(self):
  with tempfile.TemporaryDirectory() as td:
   d=Dataset("toy",td)
   a=d.group(["a"],width=10,height=10);b=d.group(["b"],width=10,height=10);d.group(["empty"],width=10,height=10)
   for side in ("r1","r2"):
    d.add(a,side,[-2,-3,5,7],"source",0)
    d.add(b,side,[-200,0,400,20],"source",1)
   d.add(a,"r1",[-2,-3,5,7],"source",2)
   d.finish()
   r=np.load(pathlib.Path(td)/"toy"/"R1.npy");s=np.load(pathlib.Path(td)/"toy"/"R2.npy")
   self.assertEqual(r.shape,(3,7));self.assertTrue(np.all(r[:,6]==1))
   groups=pq.read_table(pathlib.Path(td)/"toy"/"groups.parquet").to_pylist()
   self.assertEqual(groups[0]["r1_count"],2);self.assertEqual(groups[-1]["state"],"both_empty")
   self.assertLess(groups[0]["global_max_x"],groups[1]["global_min_x"])
   self.assertEqual(tuple(r[0,2:6]),tuple(s[0,2:6]))
   self.assertEqual(tuple(r[0,2:6]),tuple(r[1,2:6]))
if __name__=="__main__":unittest.main()

