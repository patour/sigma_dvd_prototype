import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import analysis.dynamic_irdrop_decomposition as dynamic_overlay_module
from analysis.dynamic_irdrop_decomposition import (
    AggressorResult,
    DecompositionResult,
    InstanceDecomposition,
    PEAK_IR_DROP_DATA_FILENAME,
    generate_peak_ir_drop_plots,
    generate_plots,
    plot_decomposition_overlay_peak_ir_drop_heatmaps,
    plot_victim_aggressor_bar_chart,
    plot_victim_overlay_peak_ir_drop_heatmap,
    prepare_decomposition_aggressor_bar_plot_data,
    postprocess_saved_decomposition_outputs,
    prepare_victim_aggressor_bar_plot_data,
    prepare_decomposition_overlay_plot_data,
    prepare_victim_overlay_plot_data,
)

pytestmark = pytest.mark.unit


def _make_instance(top_aggressors):
    return InstanceDecomposition(
        instance_name="inst0",
        node="100_200_M1",
        x=100.0,
        y=200.0,
        window_bounds=(80.0, 120.0, 170.0, 230.0),
        n_near_sources=1,
        n_far_sources=2,
        t_array=np.array([0.0]),
        ir_drop_total=np.array([0.0]),
        ir_drop_near=np.array([0.0]),
        ir_drop_far=np.array([0.0]),
        peak_total_mV=10.0,
        peak_near_mV=6.0,
        peak_far_mV=4.0,
        peak_time_ns=1.0,
        avg_total_mV=5.0,
        avg_near_mV=3.0,
        avg_far_mV=2.0,
        near_fraction_at_peak=60.0,
        far_fraction_at_peak=40.0,
        avg_near_fraction=60.0,
        avg_far_fraction=40.0,
        top_aggressors=top_aggressors,
    )


def test_prepare_victim_overlay_plot_data_limits_top10_and_encodes_marker_semantics():
    top_aggressors = [
        AggressorResult(
            node=f"{100 + idx}_{200 + idx}_M1",
            contribution_mV=float(20 - idx),
            contribution_pct=1.0,
            source_names=[],
            distance_um=float(idx + 1),
        )
        for idx in range(12)
    ]
    top_aggressors.insert(
        3,
        AggressorResult(
            node="140_240_M2",
            contribution_mV=99.0,
            contribution_pct=1.0,
            source_names=[],
            distance_um=5.0,
        ),
    )

    overlay = prepare_victim_overlay_plot_data(_make_instance(top_aggressors))

    assert overlay is not None
    assert overlay.victim_layer == "M1"
    assert overlay.victim_marker.marker == "x"
    assert overlay.victim_marker.edge_color == "red"
    assert overlay.victim_marker.filled is False
    assert overlay.near_window_box.edge_color == "red"
    assert len(overlay.aggressor_markers) == 10
    assert all(marker.layer == "M1" for marker in overlay.aggressor_markers)
    assert all(marker.marker == "o" for marker in overlay.aggressor_markers)
    assert all(marker.filled for marker in overlay.aggressor_markers)
    assert overlay.victim_marker.face_color is None
    assert overlay.victim_marker.face_color_mode == "none"
    assert all(marker.face_color is None for marker in overlay.aggressor_markers)
    assert all(
        marker.face_color_mode == "strength_mapped"
        for marker in overlay.aggressor_markers
    )
    assert overlay.aggressor_markers[0].node == "100_200_M1"
    assert overlay.aggressor_markers[-1].node == "109_209_M1"
    assert overlay.aggressor_markers[0].color_value == 1.0
    assert overlay.aggressor_markers[-1].color_value == 0.0

    x_min, x_max, y_min, y_max = overlay.zoom_bounds
    assert (x_min + x_max) / 2 == overlay.victim_location[0]
    assert (y_min + y_max) / 2 == overlay.victim_location[1]
    assert x_min <= overlay.near_window_box.x_min
    assert x_max >= overlay.near_window_box.x_max
    assert y_min <= overlay.near_window_box.y_min
    assert y_max >= overlay.near_window_box.y_max


def test_prepare_decomposition_overlay_plot_data_accepts_saved_result_mappings():
    saved_result = {
        "worst_instances": [
            {
                "instance_name": "inst1",
                "node": "500_600_24",
                "location": [500.0, 600.0],
                "window_bounds": [450.0, 550.0, 575.0, 625.0],
                "aggressor_analysis": {
                    "top_aggressors": [
                        {"node": "490_600_24", "contribution_mV": 8.0},
                        {"node": "510_620_23", "contribution_mV": 7.0},
                        {"node": "520_610_24", "contribution_mV": 4.0},
                    ]
                },
            }
        ]
    }

    overlays = prepare_decomposition_overlay_plot_data(saved_result)

    assert len(overlays) == 1
    overlay = overlays[0]
    assert overlay.instance_name == "inst1"
    assert overlay.victim_layer == "24"
    assert overlay.victim_location == (500.0, 600.0)
    assert [marker.node for marker in overlay.aggressor_markers] == [
        "490_600_24",
        "520_610_24",
    ]
    assert [marker.contribution_mV for marker in overlay.aggressor_markers] == [8.0, 4.0]


def test_prepare_victim_aggressor_bar_plot_data_limits_top10_and_includes_self_contribution():
    top_aggressors = [
        AggressorResult(
            node=f"{100 + idx}_{200 + idx}_M1",
            contribution_mV=float(idx + 1),
            contribution_pct=float(idx + 1),
            source_names=[],
            distance_um=float(20 - idx),
        )
        for idx in range(12)
    ]
    instance = _make_instance(top_aggressors)
    instance.self_contribution_mV = 2.5

    plot_data = prepare_victim_aggressor_bar_plot_data(instance)

    assert plot_data is not None
    assert plot_data.self_contribution_mV == pytest.approx(2.5)
    assert len(plot_data.aggressors) == 10
    assert [aggressor.contribution_mV for aggressor in plot_data.aggressors] == pytest.approx(
        [12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
    )
    assert plot_data.aggressors[0].color_value == pytest.approx(1.0)
    assert plot_data.aggressors[-1].color_value == pytest.approx(0.0)


def test_prepare_decomposition_aggressor_bar_plot_data_accepts_saved_result_mappings():
    saved_result = {
        "worst_instances": [
            {
                "instance_name": "inst1",
                "node": "500_600_24",
                "peak_ir_drop": {"total_mV": 10.0},
                "aggressor_analysis": {
                    "self_contribution_mV": 1.5,
                    "top_aggressors": [
                        {"node": "490_600_24", "contribution_mV": 8.0},
                        {"node": "520_610_24", "contribution_mV": 4.0},
                    ],
                },
            }
        ]
    }

    prepared = prepare_decomposition_aggressor_bar_plot_data(saved_result)

    assert len(prepared) == 1
    assert prepared[0].peak_total_mV == pytest.approx(10.0)
    assert prepared[0].self_contribution_mV == pytest.approx(1.5)
    assert [aggressor.node for aggressor in prepared[0].aggressors] == [
        "490_600_24",
        "520_610_24",
    ]


def test_plot_victim_overlay_peak_ir_drop_heatmap_renders_zoomed_overlay():
    top_aggressors = [
        AggressorResult(
            node="90_190_M1",
            contribution_mV=8.0,
            contribution_pct=1.0,
            source_names=[],
            distance_um=14.0,
        ),
        AggressorResult(
            node="110_210_M1",
            contribution_mV=4.0,
            contribution_pct=1.0,
            source_names=[],
            distance_um=14.0,
        ),
        AggressorResult(
            node="100_200_M2",
            contribution_mV=20.0,
            contribution_pct=1.0,
            source_names=[],
            distance_um=0.0,
        ),
    ]
    peak_ir_drop_per_node = {
        "80_180_M1": 0.001,
        "90_190_M1": 0.004,
        "100_200_M1": 0.006,
        "110_210_M1": 0.003,
        "120_220_M1": 0.002,
        "100_200_M2": 0.025,
        "bad_node": 0.010,
    }

    fig, ax = plot_victim_overlay_peak_ir_drop_heatmap(
        peak_ir_drop_per_node,
        _make_instance(top_aggressors),
        show=False,
    )

    try:
        overlay = prepare_victim_overlay_plot_data(_make_instance(top_aggressors))
        assert overlay is not None

        assert ax.get_xlim() == pytest.approx(overlay.zoom_bounds[:2])
        assert ax.get_ylim() == pytest.approx(overlay.zoom_bounds[2:])

        assert len(ax.collections) == 2
        heatmap_collection, aggressor_collection = ax.collections
        assert heatmap_collection.get_offsets().shape[0] == 5
        assert aggressor_collection.get_offsets().shape[0] == 2
        assert aggressor_collection.get_array().tolist() == pytest.approx([1.0, 0.0])

        assert len(ax.lines) == 1
        victim_line = ax.lines[0]
        assert victim_line.get_marker() == "x"
        assert victim_line.get_color() == "red"
        assert victim_line.get_xdata()[0] == pytest.approx(overlay.victim_marker.x)
        assert victim_line.get_ydata()[0] == pytest.approx(overlay.victim_marker.y)

        assert len(ax.patches) == 1
        window_patch = ax.patches[0]
        assert window_patch.get_x() == pytest.approx(overlay.near_window_box.x_min)
        assert window_patch.get_y() == pytest.approx(overlay.near_window_box.y_min)
        assert window_patch.get_width() == pytest.approx(
            overlay.near_window_box.x_max - overlay.near_window_box.x_min
        )
        assert window_patch.get_height() == pytest.approx(
            overlay.near_window_box.y_max - overlay.near_window_box.y_min
        )

        assert len(fig.axes) == 3
    finally:
        plt.close(fig)


def test_plot_victim_aggressor_bar_chart_uses_plasma_strength_colors_and_keeps_labels_in_bounds():
    top_aggressors = [
        AggressorResult(
            node="110_210_M1",
            contribution_mV=10.0,
            contribution_pct=50.0,
            source_names=[],
            distance_um=5.0,
        ),
        AggressorResult(
            node="120_220_M1",
            contribution_mV=6.0,
            contribution_pct=30.0,
            source_names=[],
            distance_um=15.0,
        ),
        AggressorResult(
            node="130_230_M1",
            contribution_mV=2.0,
            contribution_pct=10.0,
            source_names=[],
            distance_um=25.0,
        ),
    ]
    instance = _make_instance(top_aggressors)
    instance.self_contribution_mV = 2.1

    fig, ax = plot_victim_aggressor_bar_chart(instance, show=False)

    try:
        assert len(fig.axes) == 1
        assert ax.get_xlabel() == "Contribution (mV)"
        assert ax.get_ylabel() == "Aggressor Node"
        assert "Distance" not in ax.get_xlabel()
        assert "Distance" not in ax.get_ylabel()
        assert "Self: 2.100 mV" in ax.get_title()
        assert [text.get_text() for text in ax.texts] == [
            "10.00 mV | 50.0%",
            "6.00 mV | 30.0%",
            "2.00 mV | 10.0%",
        ]

        expected_colors = plt.get_cmap("plasma")([1.0, 0.5, 0.0])
        observed_colors = np.array([patch.get_facecolor() for patch in ax.patches])
        assert observed_colors.shape[0] == 3
        assert np.allclose(observed_colors, expected_colors)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axis_bbox = ax.get_window_extent(renderer=renderer)
        for text in ax.texts:
            text_bbox = text.get_window_extent(renderer=renderer)
            assert text_bbox.x0 >= axis_bbox.x0 - 1.0
            assert text_bbox.x1 <= axis_bbox.x1 + 1.0
    finally:
        plt.close(fig)


def test_plot_decomposition_overlay_peak_ir_drop_heatmaps_uses_prepared_overlays():
    result = {
        "worst_instances": [
            {
                "instance_name": "inst1",
                "node": "500_600_24",
                "location": [500.0, 600.0],
                "window_bounds": [450.0, 550.0, 575.0, 625.0],
                "aggressor_analysis": {
                    "top_aggressors": [
                        {"node": "490_600_24", "contribution_mV": 8.0},
                        {"node": "510_620_23", "contribution_mV": 7.0},
                        {"node": "520_610_24", "contribution_mV": 4.0},
                    ]
                },
            }
        ]
    }
    peak_ir_drop_per_node = {
        "470_590_24": 0.002,
        "490_600_24": 0.006,
        "500_600_24": 0.010,
        "520_610_24": 0.005,
        "540_620_24": 0.003,
        "510_620_23": 0.040,
    }

    rendered = plot_decomposition_overlay_peak_ir_drop_heatmaps(
        peak_ir_drop_per_node,
        result,
        show=False,
    )

    try:
        assert len(rendered) == 1
        overlay, fig, ax = rendered[0]
        assert overlay.instance_name == "inst1"
        assert overlay.victim_layer == "24"
        assert ax.collections[1].get_offsets().shape[0] == 2
    finally:
        for _, fig, _ in rendered:
            plt.close(fig)


def test_plot_decomposition_overlay_peak_ir_drop_heatmaps_parses_peak_map_once(
    monkeypatch,
):
    result = {
        "worst_instances": [
            {
                "instance_name": "inst1",
                "node": "500_600_24",
                "location": [500.0, 600.0],
                "window_bounds": [450.0, 550.0, 575.0, 625.0],
                "aggressor_analysis": {"top_aggressors": [{"node": "490_600_24"}]},
            },
            {
                "instance_name": "inst2",
                "node": "700_800_24",
                "location": [700.0, 800.0],
                "window_bounds": [650.0, 750.0, 775.0, 825.0],
                "aggressor_analysis": {"top_aggressors": [{"node": "710_810_24"}]},
            },
        ]
    }
    peak_ir_drop_per_node = {
        "490_600_24": 0.006,
        "500_600_24": 0.010,
        "510_610_24": 0.005,
        "700_800_24": 0.009,
        "710_810_24": 0.006,
        "720_820_24": 0.004,
    }
    call_count = 0
    original_extract = dynamic_overlay_module.extract_node_data_vectorized

    def counting_extract(node_values):
        nonlocal call_count
        call_count += 1
        assert node_values is peak_ir_drop_per_node
        return original_extract(node_values)

    monkeypatch.setattr(
        dynamic_overlay_module,
        "extract_node_data_vectorized",
        counting_extract,
    )

    rendered = plot_decomposition_overlay_peak_ir_drop_heatmaps(
        peak_ir_drop_per_node,
        result,
        show=False,
    )

    try:
        assert len(rendered) == 2
        assert call_count == 1
    finally:
        for _, fig, _ in rendered:
            plt.close(fig)


def test_plot_decomposition_overlay_peak_ir_drop_heatmaps_saves_unique_files(
    tmp_path,
):
    result = {
        "worst_instances": [
            {
                "instance_name": "inst1/core",
                "node": "500_600_24",
                "location": [500.0, 600.0],
                "window_bounds": [450.0, 550.0, 575.0, 625.0],
                "aggressor_analysis": {
                    "top_aggressors": [
                        {"node": "490_600_24", "contribution_mV": 8.0},
                    ]
                },
            },
            {
                "instance_name": "inst2 core",
                "node": "700_800_24",
                "location": [700.0, 800.0],
                "window_bounds": [650.0, 750.0, 775.0, 825.0],
                "aggressor_analysis": {
                    "top_aggressors": [
                        {"node": "710_810_24", "contribution_mV": 6.0},
                    ]
                },
            },
        ]
    }
    peak_ir_drop_per_node = {
        "490_600_24": 0.006,
        "500_600_24": 0.010,
        "510_610_24": 0.005,
        "700_800_24": 0.009,
        "710_810_24": 0.006,
        "720_820_24": 0.004,
    }
    base_save_path = tmp_path / "peak_ir_drop_overlays"

    rendered = plot_decomposition_overlay_peak_ir_drop_heatmaps(
        peak_ir_drop_per_node,
        result,
        save_path=base_save_path,
        show=False,
    )

    try:
        assert len(rendered) == 2
        expected_paths = [
            tmp_path / "peak_ir_drop_overlays" / "peak_ir_drop_overlay_1_inst1_core_24.png",
            tmp_path / "peak_ir_drop_overlays" / "peak_ir_drop_overlay_2_inst2_core_24.png",
        ]
        for path in expected_paths:
            assert path.is_file()
            assert path.stat().st_size > 0
    finally:
        for _, fig, _ in rendered:
            plt.close(fig)


def test_plot_victim_overlay_peak_ir_drop_heatmap_raises_when_zoom_window_has_no_samples():
    peak_ir_drop_per_node = {
        "500_600_M1": 0.010,
        "520_620_M1": 0.008,
    }
    instance = _make_instance([])

    with pytest.raises(ValueError, match="No peak IR-drop samples found inside zoom window"):
        plot_victim_overlay_peak_ir_drop_heatmap(
            peak_ir_drop_per_node,
            instance,
            show=False,
        )


def test_plot_decomposition_overlay_peak_ir_drop_heatmaps_skips_failed_victims():
    result = {
        "worst_instances": [
            {
                "instance_name": "inst-miss",
                "node": "100_200_M1",
                "location": [100.0, 200.0],
                "window_bounds": [80.0, 120.0, 170.0, 230.0],
                "aggressor_analysis": {"top_aggressors": []},
            },
            {
                "instance_name": "inst-hit",
                "node": "500_600_M1",
                "location": [500.0, 600.0],
                "window_bounds": [480.0, 520.0, 570.0, 630.0],
                "aggressor_analysis": {
                    "top_aggressors": [{"node": "510_610_M1", "contribution_mV": 4.0}]
                },
            },
        ]
    }
    peak_ir_drop_per_node = {
        "500_600_M1": 0.010,
        "510_610_M1": 0.008,
        "520_620_M1": 0.006,
    }

    with pytest.warns(RuntimeWarning, match="Skipping peak IR-drop overlay render for inst-miss"):
        rendered = plot_decomposition_overlay_peak_ir_drop_heatmaps(
            peak_ir_drop_per_node,
            result,
            show=False,
        )

    try:
        assert len(rendered) == 1
        assert rendered[0][0].instance_name == "inst-hit"
    finally:
        for _, fig, _ in rendered:
            plt.close(fig)


def test_generate_plots_wires_overlay_heatmaps_into_standard_flow(
    monkeypatch,
    tmp_path,
):
    result = DecompositionResult(
        netlist_dir="dummy",
        net_name="VDD",
        method="transient",
        integration_method="trap",
        t_start_ns=0.0,
        t_end_ns=1.0,
        dt_ns=1.0,
        window_percent=10.0,
        grid_bounds=(0.0, 1000.0, 0.0, 1000.0),
        worst_instances=[_make_instance([])],
        peak_ir_drop_per_node={"100_200_M1": 0.010},
    )

    stripe_calls = []
    aggressor_plot_calls = []
    overlay_calls = []
    overlay_figures = []

    def fake_generate_aggressor_contribution_plots(result_arg, plot_dir, **kwargs):
        aggressor_plot_calls.append(
            {
                "result": result_arg,
                "plot_dir": plot_dir,
                **kwargs,
            }
        )

    def fake_plot_stripe_heatmap(**kwargs):
        stripe_calls.append(kwargs)

    def fake_plot_decomposition_overlay_peak_ir_drop_heatmaps(
        peak_ir_drop_per_node,
        result_arg,
        **kwargs,
    ):
        fig, ax = plt.subplots()
        overlay_figures.append(fig)
        overlay_calls.append(
            {
                "peak_ir_drop_per_node": peak_ir_drop_per_node,
                "result": result_arg,
                **kwargs,
            }
        )
        return [(object(), fig, ax)]

    monkeypatch.setattr(
        dynamic_overlay_module,
        "generate_aggressor_contribution_plots",
        fake_generate_aggressor_contribution_plots,
    )
    monkeypatch.setattr(
        "visualization.stripe_heatmap.plot_stripe_heatmap",
        fake_plot_stripe_heatmap,
    )
    monkeypatch.setattr(
        dynamic_overlay_module,
        "plot_decomposition_overlay_peak_ir_drop_heatmaps",
        fake_plot_decomposition_overlay_peak_ir_drop_heatmaps,
    )

    generate_plots(result, str(tmp_path), show=False)

    assert len(aggressor_plot_calls) == 1
    assert aggressor_plot_calls[0]["result"] is result
    assert aggressor_plot_calls[0]["plot_dir"] == str(tmp_path)
    assert aggressor_plot_calls[0]["show"] is False
    assert aggressor_plot_calls[0]["max_aggressors"] == 10
    assert len(stripe_calls) == 1
    assert len(overlay_calls) == 1
    assert overlay_calls[0]["peak_ir_drop_per_node"] is result.peak_ir_drop_per_node
    assert overlay_calls[0]["result"] is result
    assert overlay_calls[0]["save_path"] == str(tmp_path / "peak_ir_drop_overlays")
    assert overlay_calls[0]["show"] is False
    assert callable(overlay_calls[0]["on_error"])
    assert not plt.fignum_exists(overlay_figures[0].number)


def test_generate_peak_ir_drop_plots_cleans_stale_heatmaps_and_overlay_files(tmp_path):
    result = DecompositionResult(
        netlist_dir="dummy",
        net_name="VDD",
        method="transient",
        integration_method="trap",
        t_start_ns=0.0,
        t_end_ns=1.0,
        dt_ns=1.0,
        window_percent=10.0,
        grid_bounds=(0.0, 1000.0, 0.0, 1000.0),
        worst_instances=[
            _make_instance([
                AggressorResult(
                    node="110_210_M1",
                    contribution_mV=4.0,
                    contribution_pct=40.0,
                    source_names=[],
                    distance_um=14.0,
                )
            ]),
            InstanceDecomposition(
                instance_name="inst1b",
                node="300_400_M1",
                x=300.0,
                y=400.0,
                window_bounds=(280.0, 320.0, 380.0, 420.0),
                n_near_sources=1,
                n_far_sources=0,
                t_array=np.array([0.0]),
                ir_drop_total=np.array([0.0]),
                ir_drop_near=np.array([0.0]),
                ir_drop_far=np.array([0.0]),
                peak_total_mV=12.0,
                peak_near_mV=7.0,
                peak_far_mV=5.0,
                peak_time_ns=1.0,
                avg_total_mV=6.0,
                avg_near_mV=3.0,
                avg_far_mV=3.0,
                near_fraction_at_peak=58.0,
                far_fraction_at_peak=42.0,
                avg_near_fraction=55.0,
                avg_far_fraction=45.0,
                top_aggressors=[
                    AggressorResult(
                        node="310_410_M1",
                        contribution_mV=5.0,
                        contribution_pct=41.0,
                        source_names=[],
                        distance_um=14.0,
                    )
                ],
            ),
        ],
        peak_ir_drop_per_node={
            "100_200_M1": 0.010,
            "110_210_M1": 0.008,
            "120_220_M1": 0.006,
            "300_400_M1": 0.011,
            "310_410_M1": 0.007,
            "320_420_M1": 0.005,
            "500_600_M2": 0.020,
            "520_620_M2": 0.018,
        },
    )

    plot_dir = tmp_path / "plots"
    generate_peak_ir_drop_plots(result, result.peak_ir_drop_per_node, str(plot_dir), show=False)

    assert (plot_dir / "peak_ir-drop_heatmap_M1.png").is_file()
    assert (plot_dir / "peak_ir-drop_heatmap_M2.png").is_file()
    assert (
        plot_dir
        / "peak_ir_drop_overlays"
        / "peak_ir_drop_overlay_2_inst1b_M1.png"
    ).is_file()

    stale_heatmap = plot_dir / "peak_ir-drop_heatmap_M2.png"
    stale_overlay = (
        plot_dir
        / "peak_ir_drop_overlays"
        / "peak_ir_drop_overlay_2_inst1b_M1.png"
    )
    assert stale_heatmap.is_file()
    assert stale_overlay.is_file()

    result.worst_instances = result.worst_instances[:1]
    result.peak_ir_drop_per_node = {
        "100_200_M1": 0.010,
        "110_210_M1": 0.008,
        "120_220_M1": 0.006,
    }

    generate_peak_ir_drop_plots(
        result,
        result.peak_ir_drop_per_node,
        str(plot_dir),
        show=False,
        heatmap_layers=["M1"],
    )

    assert (plot_dir / "peak_ir-drop_heatmap_M1.png").is_file()
    assert not stale_heatmap.exists()
    assert (
        plot_dir
        / "peak_ir_drop_overlays"
        / "peak_ir_drop_overlay_1_inst0_M1.png"
    ).is_file()
    assert not stale_overlay.exists()


def test_save_json_persists_peak_ir_drop_sidecar(tmp_path):
    result = DecompositionResult(
        netlist_dir="dummy",
        net_name="VDD",
        method="transient",
        integration_method="trap",
        t_start_ns=0.0,
        t_end_ns=1.0,
        dt_ns=1.0,
        window_percent=10.0,
        grid_bounds=(0.0, 1000.0, 0.0, 1000.0),
        worst_instances=[_make_instance([])],
        peak_ir_drop_per_node={"100_200_M1": 0.010, "110_210_M1": 0.008},
    )

    json_path = tmp_path / "results.json"
    result.save_json(str(json_path))

    saved_json = json.loads(json_path.read_text())
    sidecar_path = tmp_path / PEAK_IR_DROP_DATA_FILENAME

    assert saved_json["peak_ir_drop_data_file"] == PEAK_IR_DROP_DATA_FILENAME
    assert sidecar_path.is_file()
    with gzip.open(sidecar_path, "rt", encoding="utf-8") as f:
        assert json.load(f) == result.peak_ir_drop_per_node


def test_postprocess_saved_decomposition_outputs_reuses_peak_plotting_flow(
    monkeypatch,
    tmp_path,
):
    saved_result = {
        "netlist_dir": "netlist/netlist_minion/distributed_pkl",
        "net_name": "VDD_XLV",
        "worst_instances": [
            {
                "instance_name": "inst1",
                "node": "500_600_24",
                "location": [500.0, 600.0],
                "window_bounds": [450.0, 550.0, 575.0, 625.0],
                "aggressor_analysis": {
                    "top_aggressors": [
                        {"node": "490_600_24", "contribution_mV": 8.0},
                        {"node": "520_610_24", "contribution_mV": 4.0},
                    ]
                },
            }
        ],
    }
    peak_ir_drop_per_node = {
        "470_590_24": 0.002,
        "490_600_24": 0.006,
        "500_600_24": 0.010,
        "520_610_24": 0.005,
        "540_620_24": 0.003,
    }

    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps(saved_result))
    sidecar_path = tmp_path / PEAK_IR_DROP_DATA_FILENAME
    with gzip.open(sidecar_path, "wt", encoding="utf-8") as f:
        json.dump(peak_ir_drop_per_node, f)

    stripe_calls = []
    aggressor_plot_calls = []
    overlay_calls = []
    overlay_figures = []

    def fake_generate_aggressor_contribution_plots(result_arg, plot_dir, **kwargs):
        aggressor_plot_calls.append(
            {
                "result": result_arg,
                "plot_dir": plot_dir,
                **kwargs,
            }
        )

    def fake_plot_stripe_heatmap(**kwargs):
        stripe_calls.append(kwargs)

    def fake_plot_decomposition_overlay_peak_ir_drop_heatmaps(
        peak_ir_drop_arg,
        result_arg,
        **kwargs,
    ):
        fig, ax = plt.subplots()
        overlay_figures.append(fig)
        overlay_calls.append(
            {
                "peak_ir_drop_per_node": peak_ir_drop_arg,
                "result": result_arg,
                **kwargs,
            }
        )
        return [(object(), fig, ax)]

    monkeypatch.setattr(
        dynamic_overlay_module,
        "generate_aggressor_contribution_plots",
        fake_generate_aggressor_contribution_plots,
    )
    monkeypatch.setattr(
        "visualization.stripe_heatmap.plot_stripe_heatmap",
        fake_plot_stripe_heatmap,
    )
    monkeypatch.setattr(
        dynamic_overlay_module,
        "plot_decomposition_overlay_peak_ir_drop_heatmaps",
        fake_plot_decomposition_overlay_peak_ir_drop_heatmaps,
    )

    plot_dir = postprocess_saved_decomposition_outputs(
        results_json,
        verbose=True,
    )

    assert plot_dir == str(tmp_path / "plots")
    assert len(aggressor_plot_calls) == 1
    assert aggressor_plot_calls[0]["result"] == saved_result
    assert aggressor_plot_calls[0]["plot_dir"] == tmp_path / "plots"
    assert len(stripe_calls) == 1
    assert stripe_calls[0]["node_values"] == peak_ir_drop_per_node
    assert len(overlay_calls) == 1
    assert overlay_calls[0]["peak_ir_drop_per_node"] == peak_ir_drop_per_node
    assert overlay_calls[0]["result"]["worst_instances"][0]["instance_name"] == "inst1"
    assert overlay_calls[0]["save_path"] == str(tmp_path / "plots" / "peak_ir_drop_overlays")
    assert not plt.fignum_exists(overlay_figures[0].number)


def test_postprocess_saved_decomposition_outputs_recovers_legacy_peak_plots_without_sidecar(
    tmp_path,
):
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps({"worst_instances": []}))

    legacy_plot_dir = tmp_path / "plots"
    legacy_plot_dir.mkdir()
    legacy_heatmap = legacy_plot_dir / "peak_ir-drop_heatmap_24.png"
    legacy_heatmap.write_bytes(b"legacy-heatmap")

    custom_plot_dir = tmp_path / "copied-plots"
    plot_dir = postprocess_saved_decomposition_outputs(
        results_json,
        plot_dir=custom_plot_dir,
    )

    copied_heatmap = custom_plot_dir / "peak_ir-drop_heatmap_24.png"
    assert plot_dir == str(custom_plot_dir)
    assert copied_heatmap.read_bytes() == b"legacy-heatmap"


def test_postprocess_saved_decomposition_outputs_reports_partial_legacy_recovery(
    capsys,
    tmp_path,
):
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps({"worst_instances": []}))

    legacy_plot_dir = tmp_path / "plots"
    legacy_plot_dir.mkdir()
    (legacy_plot_dir / "peak_ir-drop_heatmap_24.png").write_bytes(b"legacy-heatmap")

    postprocess_saved_decomposition_outputs(results_json, verbose=True)

    stdout = capsys.readouterr().out
    assert "reusing legacy plot artifacts" in stdout
    assert "Recovered legacy peak IR-drop artifacts: 1 total (1 heatmap, 0 overlays)" in stdout


def test_remove_stale_peak_ir_drop_outputs_warns_when_deletions_fail(
    monkeypatch,
    tmp_path,
):
    heatmap = tmp_path / "peak_ir-drop_heatmap_24.png"
    heatmap.write_bytes(b"stale-heatmap")
    overlay_dir = tmp_path / "peak_ir_drop_overlays"
    overlay_dir.mkdir()
    overlay = overlay_dir / "peak_ir_drop_overlay_1_inst0_core_24.png"
    overlay.write_bytes(b"stale-overlay")

    original_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):
        if self in {heatmap, overlay}:
            raise OSError("permission denied")
        return original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.warns(RuntimeWarning, match="Failed to remove stale peak IR-drop artifact") as record:
        dynamic_overlay_module._remove_stale_peak_ir_drop_outputs(tmp_path)

    assert len(record) == 2
    assert str(heatmap) in str(record[0].message)
    assert str(overlay) in str(record[1].message)
    assert heatmap.exists()
    assert overlay.exists()


def test_postprocess_saved_decomposition_outputs_errors_when_peak_data_and_legacy_plots_missing(
    tmp_path,
):
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps({"worst_instances": []}))

    with pytest.raises(FileNotFoundError, match="missing persisted peak IR-drop data"):
        postprocess_saved_decomposition_outputs(results_json)


def test_postprocess_saved_decomposition_outputs_explicit_invalid_sidecar_fails_fast(
    tmp_path,
):
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps({"worst_instances": []}))

    legacy_plot_dir = tmp_path / "plots"
    legacy_plot_dir.mkdir()
    (legacy_plot_dir / "peak_ir-drop_heatmap_24.png").write_bytes(b"legacy-heatmap")

    with pytest.raises(
        FileNotFoundError,
        match="Persisted peak IR-drop data file not found",
    ):
        postprocess_saved_decomposition_outputs(
            results_json,
            peak_ir_drop_data_path=tmp_path / "missing-sidecar.json.gz",
        )


def test_postprocess_saved_decomposition_outputs_prefers_explicit_valid_sidecar(
    monkeypatch,
    tmp_path,
):
    results_json = tmp_path / "results.json"
    results_json.write_text(
        json.dumps(
            {
                "peak_ir_drop_data_file": "stale-sidecar.json.gz",
                "worst_instances": [],
            }
        )
    )
    default_sidecar = tmp_path / PEAK_IR_DROP_DATA_FILENAME
    explicit_sidecar = tmp_path / "explicit-sidecar.json.gz"
    with gzip.open(default_sidecar, "wt", encoding="utf-8") as f:
        json.dump({"100_200_M1": 0.010}, f)
    with gzip.open(explicit_sidecar, "wt", encoding="utf-8") as f:
        json.dump({"100_200_M1": 0.020}, f)

    captured = {}

    def fake_generate_peak_ir_drop_plots(result_arg, peak_ir_drop_per_node_arg, plot_dir_arg, **kwargs):
        captured["result"] = result_arg
        captured["peak_ir_drop_per_node"] = peak_ir_drop_per_node_arg
        captured["plot_dir"] = plot_dir_arg
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        dynamic_overlay_module,
        "generate_peak_ir_drop_plots",
        fake_generate_peak_ir_drop_plots,
    )

    postprocess_saved_decomposition_outputs(
        results_json,
        peak_ir_drop_data_path=explicit_sidecar,
    )

    assert captured["result"]["peak_ir_drop_data_file"] == "stale-sidecar.json.gz"
    assert captured["peak_ir_drop_per_node"] == {"100_200_M1": 0.020}
    assert captured["plot_dir"] == str(tmp_path / "plots")


def test_main_saved_results_entrypoint_dispatches_to_postprocess(
    monkeypatch,
    tmp_path,
):
    results_json = tmp_path / "results.json"
    results_json.write_text("{}")
    called = {}

    def fake_postprocess_saved_decomposition_outputs(
        results_json_path,
        **kwargs,
    ):
        called["results_json_path"] = results_json_path
        called.update(kwargs)
        return "done"

    monkeypatch.setattr(
        dynamic_overlay_module,
        "postprocess_saved_decomposition_outputs",
        fake_postprocess_saved_decomposition_outputs,
    )
    monkeypatch.setattr(
        dynamic_overlay_module.sys,
        "argv",
        [
            "dynamic_irdrop_decomposition",
            "--saved-results",
            str(results_json),
            "--peak-ir-drop-data",
            str(tmp_path / "custom.json.gz"),
            "--output-dir",
            str(tmp_path / "custom-plots"),
            "--heatmap-layers",
            "24,25",
            "--max-stripes",
            "123",
            "--verbose",
        ],
    )

    assert dynamic_overlay_module.main() == "done"
    assert called["results_json_path"] == str(results_json)
    assert called["peak_ir_drop_data_path"] == str(tmp_path / "custom.json.gz")
    assert called["plot_dir"] == str(tmp_path / "custom-plots")
    assert called["heatmap_layers"] == ["24", "25"]
    assert called["max_stripes"] == 123
    assert called["verbose"] is True
