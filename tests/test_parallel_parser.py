#!/usr/bin/env python3
"""
Unit tests for Parallel PDN Parser (pdn/parallel_parser.py)

Tests cover:
- MMapSpiceLineReader: Memory-mapped and gzip file reading
- Worker functions: Tile and instance model parsing in isolation
- Parallel parsing: Equivalence with sequential parsing
- Performance: Speedup measurements with multiple workers
"""

import gzip
import multiprocessing
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'pdn'))

from pdn.parallel_parser import (
    MMapSpiceLineReader,
    TileParseResult,
    InstanceParseResult,
    _parse_tile_worker,
    _parse_instance_worker,
    _build_tile_worker_args,
    _build_instance_worker_args,
    _load_nd_file_worker,
    _parse_element_line,
)
from pdn.pdn_parser import NetlistParser


class TestMMapSpiceLineReader(unittest.TestCase):
    """Test memory-mapped SPICE file reading."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_plain_text_file(self):
        """Test reading plain text SPICE file."""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1 n1 n2 100\n")
            f.write("C1 n3 n4 1e-12\n")

        with MMapSpiceLineReader(str(filepath)) as reader:
            self.assertFalse(reader.is_gzipped)
            line1 = reader.read_line()
            self.assertEqual(line1, "R1 n1 n2 100")
            line2 = reader.read_line()
            self.assertEqual(line2, "C1 n3 n4 1e-12")
            line3 = reader.read_line()
            self.assertIsNone(line3)

    def test_gzipped_file(self):
        """Test automatic gzip detection and reading."""
        filepath = Path(self.temp_dir) / "test.sp.gz"
        with gzip.open(filepath, 'wt') as f:
            f.write("R1 n1 n2 100\n")
            f.write("C1 n3 n4 1e-12\n")

        with MMapSpiceLineReader(str(filepath)) as reader:
            self.assertTrue(reader.is_gzipped)
            line1 = reader.read_line()
            self.assertEqual(line1, "R1 n1 n2 100")
            line2 = reader.read_line()
            self.assertEqual(line2, "C1 n3 n4 1e-12")

    def test_line_continuation(self):
        """Test line continuation with + prefix."""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1 n1 n2\n")
            f.write("+ 100\n")

        with MMapSpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 n1 n2 100")

    def test_comment_removal(self):
        """Test comment line removal."""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("* This is a comment\n")
            f.write("R1 n1 n2 100\n")
            f.write("* Another comment\n")

        with MMapSpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 n1 n2 100")
            self.assertIsNone(reader.read_line())

    def test_whitespace_normalization(self):
        """Test multiple whitespace normalization."""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1    n1   n2    100\n")

        with MMapSpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 n1 n2 100")

    def test_chunk_reading(self):
        """Test chunk-based reading."""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            for i in range(25):
                f.write(f"R{i} n{i} n{i+1} {100+i}\n")

        with MMapSpiceLineReader(str(filepath), chunk_size=10) as reader:
            chunk1 = reader.read_chunk()
            self.assertEqual(len(chunk1), 10)
            self.assertEqual(chunk1[0], "R0 n0 n1 100")

            chunk2 = reader.read_chunk()
            self.assertEqual(len(chunk2), 10)
            self.assertEqual(chunk2[0], "R10 n10 n11 110")

            chunk3 = reader.read_chunk()
            self.assertEqual(len(chunk3), 5)

            chunk4 = reader.read_chunk()
            self.assertEqual(len(chunk4), 0)

    def test_empty_file(self):
        """Test reading empty file."""
        filepath = Path(self.temp_dir) / "empty.sp"
        with open(filepath, 'w') as f:
            pass  # Create empty file

        with MMapSpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertIsNone(line)

    def test_boundary_marker(self):
        """Test that boundary markers (*node) are preserved."""
        filepath = Path(self.temp_dir) / "test.sp"
        with open(filepath, 'w') as f:
            f.write("R1 *n1 n2 100\n")

        with MMapSpiceLineReader(str(filepath)) as reader:
            line = reader.read_line()
            self.assertEqual(line, "R1 *n1 n2 100")


class TestNodeNetMapLoader(unittest.TestCase):
    """Test .nd file loading."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_load_nd_file(self):
        """Test loading node-net mapping from .nd file."""
        nd_path = Path(self.temp_dir) / "tile_0_0.nd"
        with open(nd_path, 'w') as f:
            f.write("node1 100 200 M1 0 VDD\n")
            f.write("node2 100 300 M1 0 VSS\n")

        node_map, node_map_lower, warnings = _load_nd_file_worker(str(nd_path))

        self.assertEqual(node_map['node1'], 'VDD')
        self.assertEqual(node_map['node2'], 'VSS')
        self.assertEqual(node_map_lower['node1'], 'vdd')
        self.assertEqual(len(warnings), 0)

    def test_load_nd_file_gzipped(self):
        """Test loading gzipped .nd file."""
        nd_path = Path(self.temp_dir) / "tile_0_0.nd.gz"
        with gzip.open(nd_path, 'wt') as f:
            f.write("node1 100 200 M1 0 VDD\n")

        node_map, node_map_lower, warnings = _load_nd_file_worker(str(nd_path))
        self.assertEqual(node_map['node1'], 'VDD')


class TestGzipSupport(unittest.TestCase):
    """Test gzip file support in parallel parser."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_gzipped_tile_parsing(self):
        """Test parsing gzipped .ckt and .nd files."""
        from pdn.parallel_parser import _parse_tile_worker

        # Create gzipped .ckt file
        ckt_content = "R1 node1 node2 100\nR2 node2 node3 200\nC1 node1 0 1e-15\n"
        ckt_path = Path(self.temp_dir) / "tile_0_0.ckt.gz"
        with gzip.open(ckt_path, 'wt') as f:
            f.write(ckt_content)

        # Create gzipped .nd file
        nd_content = "node1 0 0 M1 0 VDD\nnode2 100 0 M1 0 VDD\nnode3 200 0 M1 0 VDD\n"
        nd_path = Path(self.temp_dir) / "tile_0_0.nd.gz"
        with gzip.open(nd_path, 'wt') as f:
            f.write(nd_content)

        # Parse gzipped files
        args = (0, 0, str(ckt_path), str(nd_path), None, 1000, {})
        result = _parse_tile_worker(args)

        self.assertEqual(result.tile_id, (0, 0))
        self.assertEqual(len(result.edges), 3)
        self.assertEqual(result.stats['R'], 2)
        self.assertEqual(result.stats['C'], 1)
        self.assertEqual(len(result.errors), 0)

    def test_gzipped_instance_models(self):
        """Test parsing gzipped instance model files."""
        from pdn.parallel_parser import _parse_instance_worker

        # Create gzipped instance models file
        inst_content = "I_test1 node1 0 1e-3\nI_test2 node2 0 2e-3 pulse(0, 1e-3, 0, 1n, 1n, 5n, 10n)\n"
        inst_path = Path(self.temp_dir) / "instanceModels_0_0.sp.gz"
        with gzip.open(inst_path, 'wt') as f:
            f.write(inst_content)

        # Parse gzipped file
        args = (0, 0, str(inst_path), None, 1000, {})
        result = _parse_instance_worker(args)

        self.assertEqual(result.tile_id, (0, 0))
        self.assertEqual(result.stats['total'], 2)
        self.assertEqual(result.stats['with_waveforms'], 1)

    def test_mmap_reader_gzip_detection(self):
        """Test that MMapSpiceLineReader correctly detects gzip files."""
        # Plain text file
        plain_path = Path(self.temp_dir) / "plain.sp"
        with open(plain_path, 'w') as f:
            f.write("R1 n1 n2 100\n")

        with MMapSpiceLineReader(str(plain_path)) as reader:
            self.assertFalse(reader.is_gzipped)

        # Gzipped file
        gz_path = Path(self.temp_dir) / "gzipped.sp.gz"
        with gzip.open(gz_path, 'wt') as f:
            f.write("R1 n1 n2 100\n")

        with MMapSpiceLineReader(str(gz_path)) as reader:
            self.assertTrue(reader.is_gzipped)


class TestTileWorker(unittest.TestCase):
    """Test tile worker function in isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _create_tile_files(self, tile_x: int, tile_y: int, elements: List[str], nd_entries: List[str]):
        """Helper to create tile .ckt and .nd files."""
        ckt_path = Path(self.temp_dir) / f"tile_{tile_x}_{tile_y}.ckt"
        nd_path = Path(self.temp_dir) / f"tile_{tile_x}_{tile_y}.nd"

        with open(ckt_path, 'w') as f:
            for elem in elements:
                f.write(elem + '\n')

        with open(nd_path, 'w') as f:
            for entry in nd_entries:
                f.write(entry + '\n')

        return str(ckt_path), str(nd_path)

    def test_parse_resistors(self):
        """Test parsing resistors in tile worker."""
        ckt_path, nd_path = self._create_tile_files(
            0, 0,
            ["R1 node1 node2 1000"],  # 1000 Ohm = 1 KOhm
            ["node1 0 0 M1 0 VDD", "node2 0 1 M1 0 VDD"]
        )

        args = (0, 0, ckt_path, nd_path, None, 1000, {})
        result = _parse_tile_worker(args)

        self.assertEqual(result.tile_id, (0, 0))
        self.assertEqual(len(result.edges), 1)
        self.assertEqual(result.stats['R'], 1)

        node1, node2, elem_type, value, name, attrs = result.edges[0]
        self.assertEqual(elem_type, 'R')
        self.assertAlmostEqual(value, 1.0)  # 1 KOhm

    def test_parse_capacitors(self):
        """Test parsing capacitors in tile worker."""
        ckt_path, nd_path = self._create_tile_files(
            0, 0,
            ["C1 node1 node2 1e-15"],  # 1 fF
            ["node1 0 0 M1 0 VDD", "node2 0 1 M1 0 VDD"]
        )

        args = (0, 0, ckt_path, nd_path, None, 1000, {})
        result = _parse_tile_worker(args)

        self.assertEqual(len(result.edges), 1)
        node1, node2, elem_type, value, name, attrs = result.edges[0]
        self.assertEqual(elem_type, 'C')
        self.assertAlmostEqual(value, 1.0)  # 1 fF

    def test_net_filter(self):
        """Test net filtering in tile worker."""
        ckt_path, nd_path = self._create_tile_files(
            0, 0,
            [
                "R1 vdd_node1 vdd_node2 1000",
                "R2 vss_node1 vss_node2 1000"
            ],
            [
                "vdd_node1 0 0 M1 0 VDD",
                "vdd_node2 0 1 M1 0 VDD",
                "vss_node1 0 2 M1 0 VSS",
                "vss_node2 0 3 M1 0 VSS"
            ]
        )

        # Filter only VDD
        args = (0, 0, ckt_path, nd_path, 'vdd', 1000, {})
        result = _parse_tile_worker(args)

        self.assertEqual(len(result.edges), 1)
        node1, node2, elem_type, value, name, attrs = result.edges[0]
        self.assertIn('vdd', node1)

    def test_worker_isolation(self):
        """Verify workers don't share state between invocations."""
        ckt_path1, nd_path1 = self._create_tile_files(
            0, 0,
            ["R1 n1 n2 100"],
            ["n1 0 0 M1 0 VDD", "n2 0 1 M1 0 VDD"]
        )
        ckt_path2, nd_path2 = self._create_tile_files(
            1, 0,
            ["R2 n3 n4 200"],
            ["n3 0 0 M1 0 VDD", "n4 0 1 M1 0 VDD"]
        )

        args1 = (0, 0, ckt_path1, nd_path1, None, 1000, {})
        args2 = (1, 0, ckt_path2, nd_path2, None, 1000, {})

        result1 = _parse_tile_worker(args1)
        result2 = _parse_tile_worker(args2)

        # Each result should only contain its own data
        self.assertEqual(result1.tile_id, (0, 0))
        self.assertEqual(result2.tile_id, (1, 0))
        self.assertEqual(len(result1.edges), 1)
        self.assertEqual(len(result2.edges), 1)


class TestInstanceWorker(unittest.TestCase):
    """Test instance model worker function in isolation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_parse_current_source(self):
        """Test parsing current sources in instance worker."""
        inst_path = Path(self.temp_dir) / "instanceModels_0_0.sp"
        with open(inst_path, 'w') as f:
            f.write("I_test node1 node2 1e-3\n")  # 1 mA

        args = (0, 0, str(inst_path), None, 1000, {})
        result = _parse_instance_worker(args)

        self.assertEqual(result.tile_id, (0, 0))
        self.assertEqual(result.stats['total'], 1)
        self.assertIn('I_test', result.current_sources)

    def test_parse_current_with_waveform(self):
        """Test parsing current source with pulse waveform."""
        inst_path = Path(self.temp_dir) / "instanceModels_0_0.sp"
        with open(inst_path, 'w') as f:
            f.write("I_test node1 node2 0 pulse(0, 1e-3, 0, 1n, 1n, 10n, 20n)\n")

        args = (0, 0, str(inst_path), None, 1000, {})
        result = _parse_instance_worker(args)

        self.assertEqual(result.stats['with_waveforms'], 1)


class TestParallelSequentialEquivalence(unittest.TestCase):
    """Test that parallel parsing produces same results as sequential."""

    @classmethod
    def setUpClass(cls):
        """Check if test netlist exists."""
        cls.netlist_test = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        cls.netlist_small = Path(__file__).parent.parent / 'pdn' / 'netlist_small'
        cls.netlist_multi_tile = Path(__file__).parent.parent / 'pdn' / 'netlist_multi_tile'

    def test_parallel_vs_sequential_netlist_small(self):
        """Verify parallel parse produces same graph as sequential for netlist_small."""
        if not self.netlist_small.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_small}")

        # Sequential parse
        parser_seq = NetlistParser(str(self.netlist_small), parallel=False)
        graph_seq = parser_seq.parse()

        # Parallel parse
        parser_par = NetlistParser(str(self.netlist_small), parallel=True, n_workers=2)
        graph_par = parser_par.parse()

        # Compare node counts
        self.assertEqual(
            graph_seq.number_of_nodes(),
            graph_par.number_of_nodes(),
            "Node counts should match"
        )

        # Compare edge counts
        self.assertEqual(
            graph_seq.number_of_edges(),
            graph_par.number_of_edges(),
            "Edge counts should match"
        )

        # Compare statistics
        seq_stats = graph_seq.graph.get('stats', {})
        par_stats = graph_par.graph.get('stats', {})

        self.assertEqual(seq_stats.get('resistors'), par_stats.get('resistors'))
        self.assertEqual(seq_stats.get('capacitors'), par_stats.get('capacitors'))

    def test_parallel_vs_sequential_netlist_test(self):
        """Verify parallel parse produces same graph as sequential for netlist_test."""
        if not self.netlist_test.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_test}")

        # Sequential parse
        parser_seq = NetlistParser(str(self.netlist_test), parallel=False)
        graph_seq = parser_seq.parse()

        # Parallel parse
        parser_par = NetlistParser(str(self.netlist_test), parallel=True, n_workers=4)
        graph_par = parser_par.parse()

        # Compare node counts
        self.assertEqual(
            graph_seq.number_of_nodes(),
            graph_par.number_of_nodes(),
            "Node counts should match"
        )

        # Compare edge counts
        self.assertEqual(
            graph_seq.number_of_edges(),
            graph_par.number_of_edges(),
            "Edge counts should match"
        )

    def test_parallel_with_net_filter(self):
        """Test parallel parsing with net filter produces same results."""
        if not self.netlist_test.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_test}")

        # Sequential parse with filter
        parser_seq = NetlistParser(str(self.netlist_test), parallel=False, net_filter='VDD')
        graph_seq = parser_seq.parse()

        # Parallel parse with filter
        parser_par = NetlistParser(str(self.netlist_test), parallel=True, n_workers=2, net_filter='VDD')
        graph_par = parser_par.parse()

        # Compare counts
        self.assertEqual(
            graph_seq.number_of_nodes(),
            graph_par.number_of_nodes(),
            "Node counts should match with net filter"
        )

    def test_parallel_vs_sequential_multi_tile(self):
        """Verify parallel parse produces same graph as sequential for netlist_multi_tile (3x3 tiles)."""
        if not self.netlist_multi_tile.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_multi_tile}")

        # Sequential parse
        parser_seq = NetlistParser(str(self.netlist_multi_tile), parallel=False)
        graph_seq = parser_seq.parse()

        # Parallel parse with multiple workers
        parser_par = NetlistParser(str(self.netlist_multi_tile), parallel=True, n_workers=4)
        graph_par = parser_par.parse()

        # Compare node counts
        self.assertEqual(
            graph_seq.number_of_nodes(),
            graph_par.number_of_nodes(),
            "Node counts should match for multi-tile netlist"
        )

        # Compare edge counts
        self.assertEqual(
            graph_seq.number_of_edges(),
            graph_par.number_of_edges(),
            "Edge counts should match for multi-tile netlist"
        )

        # Compare statistics
        seq_stats = graph_seq.graph.get('stats', {})
        par_stats = graph_par.graph.get('stats', {})

        self.assertEqual(seq_stats.get('resistors'), par_stats.get('resistors'),
                        "Resistor counts should match")
        self.assertEqual(seq_stats.get('capacitors'), par_stats.get('capacitors'),
                        "Capacitor counts should match")
        self.assertEqual(seq_stats.get('inductors'), par_stats.get('inductors'),
                        "Inductor counts should match")
        self.assertEqual(seq_stats.get('isources'), par_stats.get('isources'),
                        "Current source counts should match")
        self.assertEqual(seq_stats.get('tiles_parsed'), par_stats.get('tiles_parsed'),
                        "Tiles parsed should match")

        # Verify tiles parsed equals expected (3x3 = 9)
        self.assertEqual(seq_stats.get('tiles_parsed'), 9, "Should parse 9 tiles")

    def test_multi_tile_overlapping_current_sources(self):
        """Test that overlapping current sources (multiple on same node) are handled correctly."""
        if not self.netlist_multi_tile.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_multi_tile}")

        # Parse with parallel
        parser = NetlistParser(str(self.netlist_multi_tile), parallel=True, n_workers=4)
        graph = parser.parse()

        # Get instance sources
        instance_sources = graph.graph.get('_instance_sources_objects', {})
        instance_node_map = graph.graph.get('instance_node_map', {})

        # Count nodes with multiple current sources
        node_isrc_count = {}
        for name, nodes in instance_node_map.items():
            if nodes:
                node = nodes[0]  # Positive terminal
                node_isrc_count[node] = node_isrc_count.get(node, 0) + 1

        overlapping_nodes = {n: c for n, c in node_isrc_count.items() if c > 1}

        # Verify some nodes have multiple current sources
        self.assertGreater(len(overlapping_nodes), 0,
                          "Should have nodes with overlapping current sources")

        # Verify total current sources matches
        stats = graph.graph.get('stats', {})
        self.assertEqual(stats.get('isources'), 225, "Should have 225 current sources")


class TestParallelSpeedup(unittest.TestCase):
    """Performance benchmarks for parallel parsing."""

    @classmethod
    def setUpClass(cls):
        cls.netlist_test = Path(__file__).parent.parent / 'pdn' / 'netlist_test'
        cls.netlist_multi_tile = Path(__file__).parent.parent / 'pdn' / 'netlist_multi_tile'

    def test_parallel_speedup(self):
        """Measure speedup with multiple workers."""
        if not self.netlist_test.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_test}")

        # Warm up (first parse may include import overhead)
        parser_warmup = NetlistParser(str(self.netlist_test), parallel=False)
        _ = parser_warmup.parse()

        # Sequential timing
        t0 = time.perf_counter()
        parser_seq = NetlistParser(str(self.netlist_test), parallel=False)
        _ = parser_seq.parse()
        seq_time = time.perf_counter() - t0

        # Parallel timing with 4 workers
        t0 = time.perf_counter()
        parser_par = NetlistParser(str(self.netlist_test), parallel=True, n_workers=4)
        _ = parser_par.parse()
        par_time = time.perf_counter() - t0

        speedup = seq_time / par_time if par_time > 0 else 0

        print(f"\nSpeedup benchmark:")
        print(f"  Sequential: {seq_time:.3f}s")
        print(f"  Parallel (4 workers): {par_time:.3f}s")
        print(f"  Speedup: {speedup:.2f}x")

        # Note: For small test netlists, parallel may be slower due to overhead
        # We don't assert a minimum speedup here as it depends on netlist size

    def test_parallel_speedup_multi_tile(self):
        """Measure speedup on multi-tile netlist (3x3 = 9 tiles)."""
        if not self.netlist_multi_tile.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_multi_tile}")

        # Warm up
        parser_warmup = NetlistParser(str(self.netlist_multi_tile), parallel=False)
        _ = parser_warmup.parse()

        # Sequential timing
        t0 = time.perf_counter()
        parser_seq = NetlistParser(str(self.netlist_multi_tile), parallel=False)
        graph_seq = parser_seq.parse()
        seq_time = time.perf_counter() - t0

        # Parallel timing with varying worker counts
        worker_counts = [2, 4, 8]
        par_times = {}

        for n_workers in worker_counts:
            t0 = time.perf_counter()
            parser_par = NetlistParser(str(self.netlist_multi_tile), parallel=True, n_workers=n_workers)
            graph_par = parser_par.parse()
            par_times[n_workers] = time.perf_counter() - t0

            # Verify equivalence
            self.assertEqual(graph_seq.number_of_nodes(), graph_par.number_of_nodes())
            self.assertEqual(graph_seq.number_of_edges(), graph_par.number_of_edges())

        print(f"\nMulti-tile speedup benchmark (9 tiles):")
        print(f"  Sequential: {seq_time:.3f}s")
        for n_workers, par_time in par_times.items():
            speedup = seq_time / par_time if par_time > 0 else 0
            print(f"  Parallel ({n_workers} workers): {par_time:.3f}s (speedup: {speedup:.2f}x)")

    def test_chunk_size_variation(self):
        """Test parsing with different chunk sizes produces same results."""
        if not self.netlist_multi_tile.exists():
            self.skipTest(f"Test netlist not found: {self.netlist_multi_tile}")

        # Parse with different chunk sizes
        chunk_sizes = [1000, 5000, 10000, 50000]
        results = {}

        for chunk_size in chunk_sizes:
            parser = NetlistParser(str(self.netlist_multi_tile), parallel=True,
                                   n_workers=4, chunk_size=chunk_size)
            graph = parser.parse()
            results[chunk_size] = {
                'nodes': graph.number_of_nodes(),
                'edges': graph.number_of_edges(),
            }

        # Verify all chunk sizes produce same results
        first_result = results[chunk_sizes[0]]
        for chunk_size, result in results.items():
            self.assertEqual(result['nodes'], first_result['nodes'],
                           f"Node count should be same with chunk_size={chunk_size}")
            self.assertEqual(result['edges'], first_result['edges'],
                           f"Edge count should be same with chunk_size={chunk_size}")


class TestDataClassSerialization(unittest.TestCase):
    """Test that result data classes are pickle-serializable."""

    def test_tile_result_pickle(self):
        """Test TileParseResult can be pickled."""
        import pickle

        result = TileParseResult(
            tile_id=(1, 2),
            edges=[('n1', 'n2', 'R', 1.0, 'R1', {'net_type': 'VDD'})],
            nodes={'n1': {'x': 100}},
            node_net_map={'n1': 'VDD'},
            stats={'R': 1}
        )

        # Serialize and deserialize
        data = pickle.dumps(result)
        restored = pickle.loads(data)

        self.assertEqual(restored.tile_id, (1, 2))
        self.assertEqual(len(restored.edges), 1)
        self.assertEqual(restored.nodes['n1']['x'], 100)

    def test_instance_result_pickle(self):
        """Test InstanceParseResult can be pickled."""
        import pickle

        result = InstanceParseResult(
            tile_id=(0, 0),
            current_sources={'I1': {'name': 'I1', 'dc_value': 1.0}},
            instance_node_map={'I1': ['n1', 'n2']},
            stats={'total': 1}
        )

        # Serialize and deserialize
        data = pickle.dumps(result)
        restored = pickle.loads(data)

        self.assertEqual(restored.tile_id, (0, 0))
        self.assertEqual(restored.stats['total'], 1)


if __name__ == '__main__':
    unittest.main()
