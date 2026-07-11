"""Shared fixtures for step-column and transient tests.

This module centralises the 2-tile model builder that was duplicated across
test_step_column_reuse.py and test_chunked_direct_scatter.py (both of which
dropped the ``_owns_pkl_dir`` handoff present in the canonical version in
tests/validation/test_equivalence.py::_build_synthetic_two_tile_model).

Import pattern (following tests/distributed/test_interface_iterative.py:39):
    from tests.distributed._fixtures import build_two_tile_model

or from within the same package:
    from _fixtures import build_two_tile_model
"""

from __future__ import annotations

import os
import tempfile


def build_two_tile_model(with_caps: bool = True):
    """Build a minimal 2-tile DistributedPowerGridModel for DC + transient tests.

    This is the canonical version that matches _build_synthetic_two_tile_model
    in tests/validation/test_equivalence.py.  Pad node 'pad' is placed ONLY in
    PackageData (not in tile boundary_nodes) so that solve_dc() works correctly.

    Parameters
    ----------
    with_caps:
        When True (default) capacitive edges are included (a1-0 10 fF, b1-0
        5 fF).  Pass False to build a purely resistive model.

    Returns
    -------
    model : DistributedPowerGridModel
        The model.  Call ``model.shutdown()`` when done — shutdown() calls
        shutil.rmtree on the temp directory owned by ``model._owns_pkl_dir``.
    workers : list[TileWorker]
        The two in-process workers ([tile_A, tile_B]).

    Topology
    --------
    Tile A : a1 --[1 mS]-- B   (cap: a1-0 10 fF when with_caps=True)
    Tile B : B --[3 mS]-- b1 --[1 mS]-- 0   (cap: b1-0 5 fF when with_caps=True)
    Package: pad --[10 mS]-- B;  pad at 1.0 V (Dirichlet)
    Loads  : a1 = 0.5 mA, b1 = 0.3 mA  (via current_injections)
    """
    from distributed.backend import LocalBackend
    from distributed.model import DistributedPowerGridModel
    from distributed.parser import PackageData, PowerGridMetaData, TileConfig
    from distributed.tile_worker import TileData, TileWorker

    tmp_dir = tempfile.mkdtemp(prefix="test_tw_tile_")

    cap_a = [("a1", "0", 10.0)] if with_caps else []
    cap_b = [("b1", "0", 5.0)] if with_caps else []

    tile_a = TileData(
        tile_id=(0, 0),
        resistive_edges=[("a1", "B", 1.0)],
        all_nodes={"a1", "B"},
        boundary_nodes={"B"},   # pad is NOT placed here (F13 fix)
        current_injections={"a1": 0.5},
        capacitive_edges=cap_a,
    )
    tile_b = TileData(
        tile_id=(0, 1),
        resistive_edges=[("B", "b1", 3.0), ("b1", "0", 1.0)],
        all_nodes={"B", "b1"},
        boundary_nodes={"B"},
        current_injections={"b1": 0.3},
        capacitive_edges=cap_b,
    )

    interface_nodes = {"B"}

    be = LocalBackend()
    be.initialize()

    wa = TileWorker()
    wa.setup_from_tile_data(tile_a, interface_nodes)
    wb = TileWorker()
    wb.setup_from_tile_data(tile_b, interface_nodes)

    workers = [wa, wb]

    pkg = PackageData(
        vsrc_dict={"V1": {"node_pos": "pad", "node_neg": "0", "net": "VDD", "value": 1.0}},
        package_edges=[("pad", "B", 10.0)],
        pad_nodes={"pad"},
        tap_nodes=set(),
        die_attachment_nodes=set(),
        vdd=1.0,
        net_name="VDD",
        package_cap_edges=[],
    )
    dummy_ckt = os.path.join(tmp_dir, "dummy.ckt")
    tile_configs = [
        TileConfig(tile_id=(0, 0), ckt_path=dummy_ckt, nd_path=None,
                   instance_path=None, net_filter=None),
        TileConfig(tile_id=(0, 1), ckt_path=dummy_ckt, nd_path=None,
                   instance_path=None, net_filter=None),
    ]
    metadata = PowerGridMetaData(
        tile_grid=(1, 2),
        parameters={},
        tile_configs=tile_configs,
        package_data=pkg,
        net_name="VDD",
        vdd=1.0,
    )

    model = DistributedPowerGridModel(
        backend=be,
        workers=workers,
        interface_nodes=interface_nodes,
        tile_boundary_nodes={(0, 0): ["B"], (0, 1): ["B"]},
        tile_interior_counts={(0, 0): wa.n_interior, (0, 1): wb.n_interior},
        package_data=pkg,
        metadata=metadata,
        # Hand ownership of tmp_dir to the model so shutdown() cleans it up.
        _owns_pkl_dir=tmp_dir,
    )
    return model, workers
