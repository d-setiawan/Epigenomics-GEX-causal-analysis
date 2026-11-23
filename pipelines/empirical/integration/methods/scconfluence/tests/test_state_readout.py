"""Only pool-integrity checks; the unchanged metric implementation has its own tests."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import state_readout as scorer


def test_exact_pool_order_is_retained_and_missing_cells_cannot_shrink_it():
    views = {a: {"cell_ids": np.asarray([a + str(i) for i in (2, 1, 0)]),
                 "shared": np.asarray([[2.0, 1.0], [1.0, 1.0], [0.0, 1.0]])}
             for a in scorer.pilot.ASSAYS}
    pools = {a: [a + "0", a + "2"] for a in scorer.pilot.ASSAYS}
    result = scorer.subset_frozen(views, pools)
    for a in pools:
        assert result[a]["cell_ids"].tolist() == pools[a]
        assert result[a]["shared"][:, 0].tolist() == [0.0, 2.0]
    missing = {a: {"cell_ids": v["cell_ids"][1:], "shared": v["shared"][1:]} for a, v in views.items()}
    with pytest.raises(ValueError, match="misses frozen"):
        scorer.subset_frozen(missing, pools)
    with pytest.raises(ValueError, match="no shrinking"):
        scorer.require_same_pools({"val": {a: ids[:1] for a, ids in pools.items()}}, {"val": pools})
