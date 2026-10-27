from pathlib import Path

import numpy as np

from brainxdl.xai.st_fc_interpretation import (
    CANONICAL_NETWORKS,
    aggregate_network_pairs,
    edge_vector_to_matrix,
    group_importance,
    load_schaefer_labels,
    node_involvement,
    rank_edges,
)
from brainxdl.xai.st_fidelity import undirected_edge_pairs


def test_edge_vector_and_group_importance_are_symmetric() -> None:
    pairs = undirected_edge_pairs(3)
    subjects = np.asarray([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    mean, standard_deviation, matrix = group_importance(subjects, 3, pairs)

    assert np.array_equal(mean, np.asarray([2.0, 3.0, 4.0]))
    assert np.allclose(standard_deviation, np.sqrt(2.0))
    assert np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), np.zeros(3))
    assert np.array_equal(edge_vector_to_matrix(mean, 3, pairs), matrix)


def test_schaefer_label_loading_and_edge_ranking(tmp_path: Path) -> None:
    labels = tmp_path / "labels.txt"
    labels.write_text(
        "1 7Networks_LH_Vis_1 0 0 0 0\n"
        "2 7Networks_RH_Default_1 0 0 0 0\n",
        encoding="utf-8",
    )
    atlas_rows = load_schaefer_labels(labels, expected_rois=2)
    ranking = rank_edges(
        np.asarray([0.25]),
        np.asarray([0.05]),
        undirected_edge_pairs(2),
        atlas_rows,
        n_subjects=120,
    )

    assert atlas_rows[0]["network_display"] == "Visual"
    assert atlas_rows[1]["network_display"] == "Default Mode"
    assert ranking[0]["region_i"] == "7Networks_LH_Vis_1"
    assert ranking[0]["network_j"] == "Default Mode"
    assert ranking[0]["n_subjects"] == 120


def test_network_pair_aggregation_covers_every_edge_once() -> None:
    atlas_rows = []
    for network in CANONICAL_NETWORKS:
        for hemisphere in ("LH", "RH"):
            atlas_rows.append(
                {
                    "network": network,
                    "network_display": network,
                    "label": f"7Networks_{hemisphere}_{network}_1",
                }
            )
    pairs = undirected_edge_pairs(len(atlas_rows))
    matrix = edge_vector_to_matrix(np.arange(1, len(pairs) + 1), len(atlas_rows), pairs)
    rows = aggregate_network_pairs(matrix, atlas_rows)
    score, normalised = node_involvement(matrix)

    assert len(rows) == 28
    assert sum(row["n_edges"] for row in rows) == len(pairs)
    assert np.isfinite(score).all()
    assert normalised.max() == 1.0
