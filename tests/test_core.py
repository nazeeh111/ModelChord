"""Behavioral checks for artifact identity, merge algebra and constrained repair."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from modelchord.artifacts import (
    Artifact, CompatibilityError, digest, export_artifact, validate_compatibility,
)
from modelchord.evaluate import classification
from modelchord.merge import _tsv, merge
from modelchord.models import MLP
from modelchord.repair import BuildBudget, Capability, repair


def contract(classes=None):
    return {"family": "classification", "representation": "two-features/v1",
            "preprocessing": {"version": "identity/v1"}, "tokenizer": None,
            "labels": classes or ["zero", "one"]}


def model(value=0.0, input_dim=2, classes=None):
    result = MLP(input_dim, [], classes or ["zero", "one"])
    with torch.no_grad():
        for parameter in result.parameters():
            parameter.fill_(value)
    return result


@pytest.fixture
def base(tmp_path):
    return export_artifact(tmp_path / "base", model(), contract())


def expert(tmp_path, base, name="expert", value=1.0):
    return export_artifact(tmp_path / name, model(value), contract(),
                           base_id=base.manifest["base_id"])


def mutate_manifest(artifact, change):
    value = copy.deepcopy(artifact.manifest)
    change(value)
    (artifact.path / "manifest.json").write_text(json.dumps(value))


@pytest.mark.parametrize("field,value", [
    ("tokenizer", {"vocab_sha256": "different"}),
    ("preprocessing", {"version": "different"}),
    ("representation", "another-embedding"),
    ("family", "another-task"),
])
def test_matching_shapes_do_not_override_contract(tmp_path, base, field, value):
    item = expert(tmp_path, base)
    mutate_manifest(item, lambda m: m["contract"].update({field: value}))
    with pytest.raises(CompatibilityError, match="contract"):
        validate_compatibility(base, [Artifact.open(item.path)])


def test_reordered_task_head_is_incompatible(tmp_path, base):
    item = expert(tmp_path, base)
    def change(m):
        m["architecture"]["classes"].reverse()
        m["contract"]["labels"].reverse()
    mutate_manifest(item, change)
    with pytest.raises(CompatibilityError):
        validate_compatibility(base, [Artifact.open(item.path)])


@pytest.mark.parametrize("change", [
    lambda m: m["architecture"].update({"kind": "remote-code-model"}),
    lambda m: m.update({"ties": [["layers.0.weight", "layers.0.bias"]]}),
    lambda m: m["tensors"]["layers.0.weight"].update({"dtype": "float16"}),
    lambda m: m.update({"license": "unknown-proprietary-license"}),
    lambda m: m["contract"].update({"labels": ["one", "zero"]}),
])
def test_unsupported_artifact_contracts_rejected(tmp_path, base, change):
    item = expert(tmp_path, base)
    mutate_manifest(item, change)
    with pytest.raises(CompatibilityError):
        Artifact.open(item.path)


def test_changed_base_cannot_claim_old_ancestry(tmp_path, base):
    replacement = export_artifact(tmp_path / "replacement", model(2), contract(),
                                   base_id=base.manifest["base_id"])
    item = expert(tmp_path, base)
    with pytest.raises(CompatibilityError):
        validate_compatibility(replacement, [item])


def test_valid_but_changed_source_weights_rejected_after_open(tmp_path, base):
    item = expert(tmp_path, base)
    changed = {name: tensor + 4 for name, tensor in item.state().items()}
    save_file(changed, item.path / "model.safetensors")
    with pytest.raises(CompatibilityError):
        merge(base, [item], tmp_path / "bad-merge")
    assert not (tmp_path / "bad-merge").exists()


def test_changed_source_manifest_rejected_after_open(tmp_path, base):
    item = expert(tmp_path, base)
    mutate_manifest(item, lambda m: m["provenance"].update({"changed": True}))
    with pytest.raises(CompatibilityError):
        item.tensor("layers.0.weight")


def test_corrupt_bytes_fail_before_loading(tmp_path, base):
    item = expert(tmp_path, base)
    weights = item.path / "model.safetensors"
    contents = bytearray(weights.read_bytes())
    contents[-1] ^= 1
    weights.write_bytes(contents)
    with pytest.raises(CompatibilityError, match="hash"):
        Artifact.open(item.path)


def test_nonfinite_tensors_rejected_even_with_matching_file_hash(tmp_path, base):
    item = expert(tmp_path, base)
    state = item.state()
    state["layers.0.weight"][0, 0] = float("nan")
    save_file(state, item.path / "model.safetensors")
    mutate_manifest(item, lambda m: m.update({"weights_sha256": digest(item.path / "model.safetensors")}))
    with pytest.raises(CompatibilityError, match="Nonfinite"):
        Artifact.open(item.path)


@pytest.mark.parametrize("method,scale,expected", [
    ("task_arithmetic", 0.25, 1.0),
    ("average", 1.0, 2.0),
    ("ties", 1.0, 2.0),
    ("dare", 1.0, 4.0),
])
def test_ordinary_merges_have_declared_arithmetic(tmp_path, base, method, scale, expected):
    items = [expert(tmp_path, base, "one", 1), expert(tmp_path, base, "three", 3)]
    source_hashes = [digest(a.path / "model.safetensors") for a in [base, *items]]
    result = merge(base, items, tmp_path / "merged", method=method, scale=scale, density=1)
    for value in result.state().values():
        torch.testing.assert_close(value, torch.full_like(value, expected))
    assert source_hashes == [digest(a.path / "model.safetensors") for a in [base, *items]]
    with pytest.raises(FileExistsError):
        merge(base, items, result.path, method=method)
    assert result.manifest == Artifact.open(result.path).manifest


def test_ties_does_not_average_opposed_signs(tmp_path, base):
    items = [expert(tmp_path, base, "positive", 3), expert(tmp_path, base, "negative", -1)]
    result = merge(base, items, tmp_path / "ties", method="ties", scale=1, density=1)
    torch.testing.assert_close(result.tensor("layers.0.weight"), torch.full((2, 2), 3.0))


def test_tsv_preserves_already_orthogonal_subspaces():
    deltas = torch.zeros(3, 6, 6)
    for index in range(3):
        deltas[index, index * 2, index * 2] = 2
        deltas[index, index * 2 + 1, index * 2 + 1] = 1
    torch.testing.assert_close(_tsv(deltas), deltas.sum(0), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(_tsv(torch.zeros_like(deltas))).all()


def test_regmean_matches_average_when_statistics_match(tmp_path, base):
    items = [expert(tmp_path, base, "one", 1), expert(tmp_path, base, "three", 3)]
    x = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    result = merge(base, items, tmp_path / "regmean", method="regmean",
                   calibration=[x, x.clone()], ridge=1e-8)
    for value in result.state().values():
        torch.testing.assert_close(value, torch.full_like(value, 2.0), atol=1e-5, rtol=1e-5)


def test_regmean_singular_statistics_stay_finite(tmp_path, base):
    items = [expert(tmp_path, base, "one", 1), expert(tmp_path, base, "three", 3)]
    result = merge(base, items, tmp_path / "singular", method="regmean",
                   calibration=[torch.zeros(3, 2), torch.zeros(3, 2)])
    assert all(torch.isfinite(v).all() for v in result.state().values())
    with pytest.raises(ValueError, match="RegMean"):
        merge(base, items, tmp_path / "no-calibration", method="regmean")


def test_export_failure_is_atomic_and_existing_output_immutable(tmp_path, base):
    before = {p.name: p.read_bytes() for p in base.path.iterdir()}
    with pytest.raises(FileExistsError):
        export_artifact(base.path, model(4), contract())
    assert before == {p.name: p.read_bytes() for p in base.path.iterdir()}
    invalid = contract()
    invalid["labels"] = ["mismatched", "head"]
    with pytest.raises(CompatibilityError):
        export_artifact(tmp_path / "invalid", model(), invalid)
    assert not (tmp_path / "invalid").exists()
    assert not list(tmp_path.glob(".modelchord-*"))


@pytest.mark.parametrize("x", [
    torch.zeros(0, 2), torch.zeros(1, 3), torch.zeros(2),
    torch.tensor([[float("nan"), 0.]]), torch.tensor([[float("inf"), 0.]]),
    torch.ones(1, 2, dtype=torch.float64), torch.tensor([[1e7, 0.]]),
])
def test_inference_rejects_invalid_input(base, x):
    with pytest.raises(ValueError):
        base.predict(x)


def test_calibration_bins_count_boundary_confidence_once():
    metrics = classification(model(), torch.tensor([[1., 0.]]), torch.tensor([0]))
    assert metrics["ece_10"] == pytest.approx(0.5)


def test_finite_weights_and_inputs_cannot_silently_overflow(tmp_path):
    item = export_artifact(tmp_path / "overflow", model(1e38), contract())
    with pytest.raises(ValueError, match="[Nn]onfinite|[Ff]inite|[Oo]verflow"):
        item.predict(torch.full((1, 2), 1e6))
    with pytest.raises(ValueError, match="[Nn]onfinite|[Ff]inite|[Oo]verflow"):
        classification(item.model(), torch.full((1, 2), 1e6), torch.tensor([0]))


def test_source_license_policy_cannot_be_silently_replaced(tmp_path, base):
    item = export_artifact(tmp_path / "attributed", model(1), contract(),
                           base_id=base.manifest["base_id"], license="CC-BY-4.0")
    with pytest.raises(CompatibilityError, match="[Ll]icense"):
        merge(base, [item], tmp_path / "incorrect-mit")


def batches(conflict=False):
    # Distinct calibration/selection feature bytes; labels describe the same tasks.
    calibration = {"skill": (torch.tensor([[1., 0.]]), torch.tensor([1])),
                   "clean": (torch.tensor([[1., 0.]]) if conflict else torch.tensor([[0., 1.]]), torch.tensor([0]))}
    selection = {"skill": (torch.tensor([[2., 0.]]), torch.tensor([1])),
                 "clean": (torch.tensor([[2., 0.]]) if conflict else torch.tensor([[0., 2.]]), torch.tensor([0]))}
    return calibration, selection


def small_budget(**kwargs):
    return BuildBudget(rank=kwargs.get("rank", 1), rounds=2, candidate_limit=3,
                       learning_rates=(0.2,), ranks=(1,))


def test_repair_satisfies_floor_within_rank_budget(tmp_path, base):
    calibration, selection = batches()
    caps = [Capability("skill", 1), Capability("clean", 0, protected=True, max_regression=0)]
    result, report = repair(base, calibration, selection, caps, tmp_path / "repair", budget=small_budget())
    assert report["status"] == "acceptable candidate found"
    assert report["fraction_floors_met"] == 1
    assert report["worst_deficit"] == 0
    assert report["rank_used_upper_bound"] == 1
    assert report["attempts"] <= 3
    delta = result.tensor("layers.0.weight") - base.tensor("layers.0.weight")
    assert torch.linalg.matrix_rank(delta) <= 1
    assert report["final"]["clean"]["accuracy"] == 1


def test_conflicting_protected_task_rejects_edit_and_rolls_back(tmp_path, base):
    calibration, selection = batches(conflict=True)
    caps = [Capability("skill", 1), Capability("clean", 0, protected=True, max_regression=0)]
    result, report = repair(base, calibration, selection, caps, tmp_path / "conflict", budget=small_budget())
    assert report["status"] == "no acceptable candidate found"
    assert report["attempts"] > 0
    assert report["rank_used_upper_bound"] == 0
    assert not any(entry["accepted"] for entry in report["trace"])
    assert any(not entry["protected_gate"] for entry in report["trace"])
    for name, value in base.state().items():
        torch.testing.assert_close(result.tensor(name), value)
    _, unprotected = repair(base, calibration, selection, caps, tmp_path / "unprotected",
                            budget=small_budget(), protect=False)
    assert unprotected["status"] == "acceptable candidate found"
    assert unprotected["final"]["clean"]["accuracy"] == 0


def test_zero_budget_reports_failure_and_no_correction(tmp_path, base):
    calibration, selection = batches()
    caps = [Capability("skill", 1), Capability("clean", 0)]
    result, report = repair(base, calibration, selection, caps, tmp_path / "zero", budget=small_budget(rank=0))
    assert report["status"] == "no acceptable candidate found"
    assert report["attempts"] == 0
    assert report["rank_used_upper_bound"] == 0
    torch.testing.assert_close(result.tensor("layers.0.weight"), base.tensor("layers.0.weight"))


def test_acceptable_ordinary_merge_uses_fast_path(tmp_path, base):
    calibration, selection = batches()
    _, report = repair(base, calibration, selection, [Capability("skill", 0), Capability("clean", 0)],
                        tmp_path / "fast", budget=small_budget())
    assert report["status"] == "acceptable candidate found"
    assert report["attempts"] == 0
    assert report["rank_used_upper_bound"] == 0


def test_identical_calibration_and_selection_batches_rejected(tmp_path, base):
    calibration, _ = batches()
    selection = {name: (x.clone(), y.clone()) for name, (x, y) in calibration.items()}
    with pytest.raises(ValueError, match="[Dd]isjoint|[Ii]dentical|[Oo]verlap"):
        repair(base, calibration, selection, [Capability("skill", 1), Capability("clean", 0)], tmp_path / "leak")


def test_selection_example_accounting_includes_final_measurement(tmp_path, base):
    calibration, selection = batches()
    selected_tensors = [batch[0] for batch in selection.values()]
    evaluated = 0
    def counted(model, x, y):
        nonlocal evaluated
        if any(x is tensor for tensor in selected_tensors):
            evaluated += len(y)
        return classification(model, x, y)
    _, report = repair(base, calibration, selection, [Capability("skill", 1), Capability("clean", 0)],
                        tmp_path / "counted", budget=small_budget(rank=0), evaluator=counted)
    assert report["validation_examples"] == evaluated
