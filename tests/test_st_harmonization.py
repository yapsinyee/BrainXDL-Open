from pathlib import Path

import numpy as np
import pandas as pd

from brainxdl.st import harmonization


def test_align_abide_phenotype_to_roi_order_uses_sorted_roi_files(tmp_path: Path) -> None:
    rois_dir = tmp_path / "rois"
    rois_dir.mkdir()
    (rois_dir / "b_rois_aal.1D").write_text("1 2\n", encoding="utf-8")
    (rois_dir / "a_rois_aal.1D").write_text("3 4\n", encoding="utf-8")

    pheno_path = tmp_path / "pheno.csv"
    pd.DataFrame(
        [
            {"FILE_ID": "b", "SITE_ID": "S2", "DX_GROUP": 2},
            {"FILE_ID": "a", "SITE_ID": "S1", "DX_GROUP": 1},
        ]
    ).to_csv(pheno_path, index=False)

    aligned = harmonization.align_abide_phenotype_to_roi_order(rois_dir, pheno_path)

    assert aligned["FILE_ID"].tolist() == ["a", "b"]
    assert aligned["SITE_ID"].tolist() == ["S1", "S2"]


def test_harmonize_ldw_connectivity_skips_constant_features(monkeypatch) -> None:
    calls = []

    def fake_neurocombat(**kwargs):
        calls.append(kwargs)
        return {"data": kwargs["dat"] + 10.0}

    monkeypatch.setattr(harmonization, "_import_neurocombat", lambda: fake_neurocombat)

    base = np.array(
        [
            [1.0, 0.2],
            [0.2, 1.0],
        ]
    )
    node_feats = [
        [base.copy()],
        [np.array([[1.0, 0.4], [0.4, 1.0]])],
    ]
    covars = pd.DataFrame(
        {
            "SITE_ID": ["A", "B"],
            "DX_GROUP": [1, 2],
            "SEX": [1, 2],
            "AGE_AT_SCAN": [10.0, 11.0],
        }
    )

    harmonized, adj, reports = harmonization.harmonize_ldw_connectivity(
        node_feats,
        covars,
        threshold_p=0.5,
    )

    assert len(calls) == 1
    assert calls[0]["dat"].shape == (2, 2)
    assert calls[0]["categorical_cols"] == ["DX_GROUP", "SEX"]
    assert calls[0]["continuous_cols"] == ["AGE_AT_SCAN"]
    np.testing.assert_allclose(np.diag(harmonized[0][0]), [1.0, 1.0])
    assert harmonized[0][0][0, 1] > 10.0
    assert reports[0].n_harmonized_features == 2
    assert reports[0].n_constant_or_invalid_features == 2
    assert adj[0][0].shape == (2, 2)
