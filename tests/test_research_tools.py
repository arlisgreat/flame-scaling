import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_crossover import first_crossover, load_paired_differences
from scripts.analyze_identity_scaling_followups import paired_method_effects
from scripts.generate_run_matrix import stable_id
from scripts.run_animation_oracle import (
    build_correspondence_permutations,
    nested_uniform_subset,
    tracking_codes,
)
from scripts.run_geometry_oracle import build_density_scaffold
from scripts.run_identity_scaling import (
    augment_region_indices,
    augment_subject_density,
    augmented_mesh_edges,
    bounded_vector,
    greedy_farthest_camera_order,
    released_means,
    surface_sampling_plan,
    unique_mesh_edges,
)
from flame_oracle.flame import apply_tracking_world_transform, batch_rodrigues


class CrossoverTest(unittest.TestCase):
    def test_observed_crossing(self):
        result = first_crossover({32: -0.1, 128: -0.02, 512: 0.04})
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["bracket"], [128, 512])
        self.assertGreater(result["log_interpolated_estimate"], 128)
        self.assertLess(result["log_interpolated_estimate"], 512)

    def test_right_censoring(self):
        result = first_crossover({32: -0.1, 128: -0.02})
        self.assertEqual(result, {"status": "right_censored", "above": 128})

    def test_stable_run_id_ignores_key_order(self):
        left = stable_id({"method": "hard", "seed": 0})
        right = stable_id({"seed": 0, "method": "hard"})
        self.assertEqual(left, right)

    def test_tidy_rows_are_paired_by_identity(self):
        fields = ["method", "n_id", "n_obs", "capacity", "region", "seed", "metric", "value", "identity_group"]
        rows = [
            ["hard", 32, 16, "M", "tongue", 0, "lpips", 0.30, "id_a"],
            ["free", 32, 16, "M", "tongue", 0, "lpips", 0.20, "id_a"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerows(rows)
            differences, identity_column = load_paired_differences(path, "lpips", "hard", "free", True)
        self.assertEqual(identity_column, "identity_group")
        self.assertEqual(len(differences), 1)
        self.assertAlmostEqual(differences[0]["delta_free_advantage"], 0.10)

    def test_nested_uniform_subset_has_requested_unique_count(self):
        indices = np.arange(17, dtype=np.int64)
        selected = nested_uniform_subset(indices, 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(np.unique(selected)), 6)
        self.assertEqual(selected[0], indices[0])
        self.assertEqual(selected[-1], indices[-1])

    def test_tracking_codes_are_fixed_width_and_image_independent(self):
        rng = np.random.default_rng(0)
        tracking = {
            "expression": rng.normal(size=(20, 100)).astype(np.float32),
            "neck": rng.normal(size=(20, 3)).astype(np.float32),
            "jaw": rng.normal(size=(20, 3)).astype(np.float32),
            "eyes": rng.normal(size=(20, 6)).astype(np.float32),
        }
        codes, report = tracking_codes(tracking, 8)
        self.assertEqual(codes.shape, (20, 9))
        self.assertTrue(np.isfinite(codes).all())
        self.assertTrue(np.all(codes[:, 0] == 1))
        self.assertFalse(report["uses_image_targets"])

    def test_correspondence_intervention_preserves_each_geometry_set(self):
        vertices = np.stack(
            np.meshgrid(np.arange(5), np.arange(4), np.arange(2), indexing="ij"), axis=-1
        ).reshape(-1, 3).astype(np.float32) * 0.001
        permutations, effective = build_correspondence_permutations(
            vertices, frame_count=4, fraction=0.5, mode="global", local_bin_mm=10, seed=0
        )
        for row in permutations:
            np.testing.assert_array_equal(np.sort(row), np.arange(len(vertices)))
        self.assertAlmostEqual(effective, 0.5)

    def test_zero_rodrigues_and_world_transform(self):
        rotvec = torch.zeros((2, 3), requires_grad=True)
        rotation = batch_rodrigues(rotvec)
        torch.testing.assert_close(rotation, torch.eye(3)[None].expand(2, -1, -1))
        rotation[:, 2, 1].sum().backward()
        self.assertTrue(torch.isfinite(rotvec.grad).all())
        self.assertGreater(abs(float(rotvec.grad[:, 0].sum())), 0.0)
        vertices = torch.tensor([[[1.0, 2.0, 3.0]]])
        transformed = apply_tracking_world_transform(
            vertices,
            torch.eye(3)[None],
            torch.tensor([[0.5, -0.5, 1.0]]),
            torch.tensor([[2.0]]),
        )
        torch.testing.assert_close(transformed, torch.tensor([[[2.5, 3.5, 7.0]]]))

    def test_density_scaffold_is_nested_and_stays_on_surface(self):
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        colors = vertices.copy()
        normals = np.tile(np.array([[0, 0, 1]], dtype=np.float32), (4, 1))
        low, _, _, low_report = build_density_scaffold(
            vertices, faces, colors, normals, multiplier=0.5, seed=7
        )
        full, _, _, full_report = build_density_scaffold(
            vertices, faces, colors, normals, multiplier=1.0, seed=7
        )
        dense, _, dense_normals, dense_report = build_density_scaffold(
            vertices, faces, colors, normals, multiplier=2.0, seed=7
        )
        self.assertEqual(low.shape, (2, 3))
        np.testing.assert_array_equal(full, vertices)
        np.testing.assert_array_equal(dense[: len(vertices)], vertices)
        self.assertTrue(np.all(dense[:, 0] >= 0) and np.all(dense[:, 0] <= 1))
        self.assertTrue(np.all(dense[:, 1] >= 0) and np.all(dense[:, 1] <= 1))
        self.assertTrue(np.allclose(dense[:, 2], 0))
        self.assertTrue(np.allclose(dense_normals, normals[0]))
        self.assertEqual(low_report["construction"], "nested_vertex_fps_prefix")
        self.assertEqual(full_report["surface_sample_count"], 0)
        self.assertEqual(dense_report["surface_sample_count"], 4)

    def test_camera_observation_order_is_source_first_and_complete(self):
        camera_ids = ["left", "source", "right", "back"]
        centers = np.asarray(
            [[-1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 0, -1]], dtype=np.float32
        )
        viewmats = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        # With identity rotations, t=-center gives the requested camera center.
        viewmats[:, :3, 3] = -centers
        order = greedy_farthest_camera_order(camera_ids, viewmats, "source")
        self.assertEqual(order[0], 1)
        self.assertEqual(sorted(order), list(range(4)))
        self.assertEqual(len(set(order[:3])), 3)

    def test_geometry_release_has_exact_hard_and_bounded_support(self):
        base = torch.zeros((5, 3))
        raw = torch.tensor(
            [[100.0, 0.0, 0.0], [0.0, -100.0, 0.0], [0.0, 0.0, 100.0]]
            + [[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]
        )
        gate = torch.zeros((5, 1))
        hard, hard_offset, _ = released_means(base, raw, gate, "hard", 30.0, 150.0)
        adaptive, adaptive_offset, _ = released_means(
            base, raw, gate, "adaptive", 30.0, 150.0
        )
        released, released_offset, _ = released_means(
            base, raw, gate, "released", 30.0, 150.0
        )
        torch.testing.assert_close(hard, base)
        torch.testing.assert_close(hard_offset, torch.zeros_like(base))
        self.assertLessEqual(float(adaptive_offset.norm(dim=1).max()), 0.015001)
        self.assertLessEqual(float(released_offset.norm(dim=1).max()), 0.150001)
        self.assertGreater(float(released_offset.norm(dim=1).max()), 0.149)
        self.assertLessEqual(float(bounded_vector(raw, 0.03).norm(dim=1).max()), 0.030001)

    def test_mesh_edges_are_undirected_and_unique(self):
        faces = np.asarray([[0, 1, 2], [2, 1, 3]], dtype=np.int64)
        edges = unique_mesh_edges(faces)
        np.testing.assert_array_equal(
            edges,
            np.asarray([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]], dtype=np.int64),
        )

    def test_followup_pairs_at_identity_level_without_duplicate_method_cells(self):
        rows = []
        for subject, hard, adaptive, released in [
            ("a", 10.0, 11.0, 12.0),
            ("b", 20.0, 20.5, 23.0),
        ]:
            for seed in [0, 1]:
                for method, value in [
                    ("hard", hard),
                    ("adaptive", adaptive),
                    ("released", released),
                ]:
                    rows.append(
                        {
                            "variant": "control",
                            "n_id": 8,
                            "method": method,
                            "seed": seed,
                            "subject": subject,
                            "n_obs": 4,
                            "region": "head_foreground",
                            "masked_psnr": value,
                        }
                    )
        effects = paired_method_effects(
            rows, "masked_psnr", np.random.default_rng(0), samples=100
        )
        self.assertEqual(len(effects), 2)
        lookup = {row["comparison"]: row for row in effects}
        self.assertAlmostEqual(lookup["adaptive"]["advantage_over_hard_mean"], 0.75)
        self.assertAlmostEqual(lookup["released"]["advantage_over_hard_mean"], 2.5)

    def test_population_density_slots_are_nested_and_topology_consistent(self):
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        normals = np.tile(np.asarray([[0, 0, 1]], dtype=np.float32), (4, 1))
        face_indices, barycentric = surface_sampling_plan(
            faces, len(vertices), density_multiplier=2, seed=9
        )
        np.testing.assert_allclose(barycentric.sum(axis=1), 1.0)
        self.assertTrue(np.all(barycentric >= 0))
        subject = {
            "base": torch.from_numpy(vertices.copy()),
            "normals": torch.from_numpy(normals.copy()),
            "faces": faces,
        }
        augment_subject_density(subject, face_indices, barycentric)
        self.assertEqual(len(subject["base"]), 8)
        torch.testing.assert_close(subject["base"][:4], torch.from_numpy(vertices))
        self.assertTrue(torch.allclose(subject["base"][4:, 2], torch.zeros(4)))
        edges = augmented_mesh_edges(faces, len(vertices), face_indices)
        for extra in range(4, 8):
            self.assertEqual(int(np.sum(np.any(edges == extra, axis=1))), 3)
        groups = {"region": np.asarray([0, 1, 2], dtype=np.int64)}
        augmented = augment_region_indices(
            groups, faces, len(vertices), face_indices
        )["region"]
        expected_extra = 4 + np.flatnonzero(
            np.isin(faces[face_indices], groups["region"]).sum(axis=1) >= 2
        )
        np.testing.assert_array_equal(augmented[3:], expected_extra)


if __name__ == "__main__":
    unittest.main()
