import tempfile
from pathlib import Path
import unittest
import numpy as np
from fit import bounds, counts_and_ids

class TestBounds(unittest.TestCase):
    def test_bounds(self):
        self.assertEqual(bounds(.5,.5),(0,.5))
        self.assertEqual(bounds(0,.7),(0,0))
        self.assertEqual(bounds(1,.7),(.7,.7))
        lo,hi = bounds(.7,.6)
        self.assertAlmostEqual(lo,.3); self.assertAlmostEqual(hi,.6)
        for r in [lo,hi]:
            cells = [r,.7-r,.6-r,1-.7-.6+r]
            self.assertGreaterEqual(min(cells),-1e-12); self.assertAlmostEqual(sum(cells),1)
        with self.assertRaises(ValueError): bounds(float('nan'), .5)
    def test_validation_and_reversal(self):
        ids = [f'f{i}' for i in range(300)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'x.npz'; x = np.arange(600).reshape(2,300)
            np.savez(path,raw_counts=x,feature_ids=ids)
            a,_=counts_and_ids(path,ids)
            np.testing.assert_array_equal((a>0).mean(0),(a[::-1]>0).mean(0))
            with self.assertRaises(ValueError): counts_and_ids(path,ids[::-1])
            x[0,0]=-1; np.savez(path,raw_counts=x,feature_ids=ids)
            with self.assertRaises(ValueError): counts_and_ids(path,ids)

if __name__ == '__main__': unittest.main()
