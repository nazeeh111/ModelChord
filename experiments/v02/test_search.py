"""Independent v0.2 regressions; synthetic data only, no confirmation access."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from modelchord import Capability, MLP, export_artifact
import repair_search as rs

torch.set_num_threads(1)


@pytest.fixture
def inputs(tmp_path):
    torch.manual_seed(81)
    model = MLP(6, [6, 5], ["0", "1", "2", "3"])
    contract = dict(family="vision", representation="synthetic-review/v1",
                    preprocessing="identity", labels=model.spec["classes"], tokenizer=None)
    ordinary = export_artifact(tmp_path / "ordinary", model, contract)
    cal = {n: (torch.rand(3, 6), torch.tensor([0, 1, 2])) for n in ["clean", "other"]}
    selection = {n: (torch.rand(4, 6), torch.tensor([0, 1, 2, 3])) for n in cal}
    caps = [Capability("clean", .88, True, .02), Capability("other", .90)]
    return ordinary, cal, selection, caps


def fingerprints(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


def patch_selection_evaluator(monkeypatch, callback, data):
    """Script complete measurement results while exercising real access counters."""
    names = list(data)
    calls = 0
    cached = None

    def classification(model, x, y):
        nonlocal calls, cached
        position = calls % len(names)
        if position == 0:
            cached = callback(model, data)
        calls += 1
        return cached[names[position]]

    monkeypatch.setattr(rs, "classification", classification)


def test_factor_starts_at_zero_correction_learns_and_folds():
    torch.manual_seed(2)
    original = torch.nn.Linear(6, 5)
    layer = rs.FactorLinear(original, 2, 9)
    x = torch.randn(7, 6)
    torch.testing.assert_close(layer(x), original(x), atol=0, rtol=0)
    torch.nn.functional.cross_entropy(layer(x), torch.arange(7) % 5).backward()
    assert layer.b.grad.norm() > 0
    assert all(p.grad is None for p in layer.base.parameters())
    with torch.no_grad():
        layer.b.add_(layer.b.grad, alpha=-.3)
    assert not torch.equal(layer(x), original(x))
    torch.testing.assert_close(layer(x), layer.fold()(x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(layer.fold().bias, original.bias, atol=0, rtol=0)


def test_projection_bounds_correction_and_preserves_factorization():
    layer = rs.FactorLinear(torch.nn.Linear(6, 5), 2, 9)
    with torch.no_grad():
        layer.b.fill_(20)
    layer.project(.25)
    assert float((layer.b @ layer.a).detach().norm()) <= .250001
    assert torch.linalg.matrix_rank(layer.b @ layer.a) <= 2


def test_exact_accuracy_counts_do_not_reject_a_floor_boundary():
    logits = torch.tensor([[10., 0.]] * 22 + [[0., 10.]] * 3)
    labels = torch.zeros(25, dtype=torch.long)
    metrics = rs.classification(torch.nn.Identity(), logits, labels)
    assert metrics["correct"] == 22
    assert metrics["examples"] == 25
    assert metrics["accuracy"] == 22 / 25
    assert rs.objective({"clean": metrics}, {"clean": .88})[0] == 0.


def test_search_fixed_reference_gate_rollback_rank_counts_and_immutable_parent(inputs, tmp_path, monkeypatch):
    ordinary, cal, selection, caps = inputs
    before = fingerprints(ordinary.path)
    seen = []

    def measured(model, data):
        call = len(seen)
        seen.append(call)
        # Stage one accepts .89; stage two must still protect against original
        # .90, rejecting .87 even though it meets .89 minus .02.
        clean, other = ((.90, .10) if call == 0 else
                        (.89, .50) if call in [1, 13] else (.87, .95))
        return {n: {"accuracy": a, "cross_entropy": 1-a, "examples": len(data[n][1])}
                for n, a in [("clean", clean), ("other", other)]}

    patch_selection_evaluator(monkeypatch, measured, selection)
    artifact, report = rs.search(ordinary, cal, selection, caps, tmp_path / "out",
                                budget=rs.SearchBudget(rank=4, stages=2, steps_per_trial=1))
    assert report["final"]["clean"]["accuracy"] == .89
    assert report["status"] == "no acceptable candidate found"
    assert sum(t["selected"] for t in report["trace"]) == 1
    assert report["rank_used_upper_bound"] == 1
    assert report["gradient_steps_total_search"] == 12
    assert report["gradient_examples_total_search"] == 72
    assert report["selection_examples_total_search"] == 112
    assert fingerprints(ordinary.path) == before
    assert artifact.path.exists()


def test_reference_keeps_earlier_checkpoint_bytes_and_matches_accesses(inputs, tmp_path, monkeypatch):
    ordinary, cal, selection, caps = inputs
    before = fingerprints(ordinary.path)
    seen = []
    winning_state = {}

    def measured(model, data):
        call = len(seen)
        seen.append(call)
        if call == 1:
            winning_state.update({n: p.detach().clone() for n, p in model.state_dict().items()})
        clean, other = ((.90, .10) if call == 0 else
                        (.90, .95) if call in [1, 13] else (.87, .95))
        return {n: {"accuracy": a, "cross_entropy": 1-a, "examples": len(data[n][1])}
                for n, a in [("clean", clean), ("other", other)]}

    patch_selection_evaluator(monkeypatch, measured, selection)
    artifact, report = rs.reference_search(ordinary, cal, selection, caps, tmp_path / "reference",
                                           stages=2, budget=rs.SearchBudget(steps_per_trial=1))
    assert report["status"] == "acceptable candidate found"
    assert report["gradient_steps_total_search"] == 12
    assert report["gradient_examples_total_search"] == 72
    assert report["selection_examples_total_search"] == 112
    assert sum(t["selected"] for t in report["trace"]) == 1
    assert report["trace"][0]["selected"]
    for name, tensor in artifact.state().items():
        torch.testing.assert_close(tensor, winning_state[name], atol=0, rtol=0)
    assert fingerprints(ordinary.path) == before


def test_failed_selection_inference_still_consumes_access_budget(inputs, tmp_path, monkeypatch):
    ordinary, cal, selection, caps = inputs
    calls = 0

    def classification(model, x, y):
        nonlocal calls
        calls += 1
        if calls == 3:  # Initial two domains finish; first trial fails on its first domain.
            raise RuntimeError("deliberate selection failure")
        return {"accuracy": .1, "cross_entropy": 2., "examples": len(y)}

    monkeypatch.setattr(rs, "classification", classification)
    _, report = rs.search(ordinary, cal, selection, caps, tmp_path / "failed-trial",
                          budget=rs.SearchBudget(rank=2, stages=1, steps_per_trial=1))
    assert report["trace"][0]["failure"] == "deliberate selection failure"
    assert report["trial_count"] == 6
    assert report["selection_examples_total_search"] == 60
    assert calls == 15
    assert report["gradient_steps_total_search"] == 6


def test_fixed_rank_ablation_keeps_six_trials_and_enforces_rank_sum(inputs, tmp_path, monkeypatch):
    ordinary, cal, selection, caps = inputs
    calls = 0

    def measured(model, data):
        nonlocal calls
        call = calls
        calls += 1
        other = .1 if call == 0 else .5 if call <= 6 else .6
        return {n: {"accuracy": a, "cross_entropy": 1-a, "examples": len(data[n][1])}
                for n, a in [("clean", .90), ("other", other)]}

    patch_selection_evaluator(monkeypatch, measured, selection)
    _, report = rs.search(ordinary, cal, selection, caps, tmp_path / "fixed-rank",
                          fixed_rank=True,
                          budget=rs.SearchBudget(rank=4, stages=2, steps_per_trial=1))
    assert report["ablations"]["fixed_rank"] is True
    assert report["trial_count"] == 12
    assert report["gradient_steps_total_search"] == 12
    assert report["rank_used_upper_bound"] == 4
    assert {t["spec"]["rank"] for t in report["trace"]} == {2}
    assert {t["spec"]["initialization_slot"] for t in report["trace"]} == {0, 1}


@pytest.mark.parametrize("protect", [True, False])
def test_no_protection_ablation_removes_gate_and_effective_margin(inputs, tmp_path, monkeypatch, protect):
    ordinary, cal, selection, _ = inputs
    caps = [Capability("clean", .80, True, .02), Capability("other", .90)]
    calls = 0

    def measured(model, data):
        nonlocal calls
        call = calls
        calls += 1
        clean, other = ((.90, .10) if call == 0 or (call == 7 and protect) else (.85, .95))
        return {n: {"accuracy": a, "cross_entropy": 1-a, "examples": len(data[n][1])}
                for n, a in [("clean", clean), ("other", other)]}

    patch_selection_evaluator(monkeypatch, measured, selection)
    _, report = rs.search(ordinary, cal, selection, caps, tmp_path / "protection",
                          protect=protect,
                          budget=rs.SearchBudget(rank=2, stages=1, steps_per_trial=1))
    assert report["ablations"]["protect"] is protect
    assert report["effective_floors"]["clean"] == (.88 if protect else .8)
    assert sum(t["selected"] for t in report["trace"]) == (0 if protect else 1)
    assert report["status"] == ("no acceptable candidate found" if protect else "acceptable candidate found")
    assert report["gradient_steps_total_search"] == 6


@pytest.mark.parametrize("function", [rs.search, rs.reference_search])
def test_identical_calibration_selection_is_rejected_before_training(inputs, tmp_path, function):
    ordinary, cal, _, caps = inputs
    kwargs = dict(budget=rs.SearchBudget(steps_per_trial=1))
    if function is rs.reference_search:
        kwargs["stages"] = 1
    with pytest.raises(ValueError, match="[Ii]dentical|disjoint|overlap"):
        function(ordinary, cal, cal, caps, tmp_path / "bad-overlap", **kwargs)


@pytest.mark.parametrize("function", [rs.search, rs.reference_search])
def test_duplicate_capability_names_are_rejected(inputs, tmp_path, function):
    ordinary, cal, selection, caps = inputs
    kwargs = dict(budget=rs.SearchBudget(steps_per_trial=1))
    if function is rs.reference_search:
        kwargs["stages"] = 1
    with pytest.raises(ValueError, match="[Uu]nique|[Dd]uplicate"):
        function(ordinary, cal, selection, caps + [caps[0]], tmp_path / "bad-caps", **kwargs)


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_loss_temperature_is_rejected_before_training(inputs, tmp_path, temperature):
    ordinary, cal, selection, caps = inputs
    budget = replace(rs.SearchBudget(steps_per_trial=1), temperature=temperature)
    with pytest.raises(ValueError, match="[Tt]emperature|[Bb]udget"):
        rs.search(ordinary, cal, selection, caps, tmp_path / "bad-temperature", budget=budget)


def test_fixed_factor_reference_cannot_exceed_declared_rank(inputs, tmp_path):
    ordinary, cal, selection, caps = inputs
    with pytest.raises(ValueError, match="[Rr]ank|[Bb]udget"):
        rs.reference_search(ordinary, cal, selection, caps, tmp_path / "bad-rank",
                            mode="fixed_factor", stages=1,
                            budget=rs.SearchBudget(rank=4, stages=2, steps_per_trial=1))


def test_data_hash_binds_shape_and_dtype():
    x = torch.arange(6, dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.int64)
    assert rs.data_hashes({"a": (x.reshape(2, 3), y)}) != rs.data_hashes({"a": (x.reshape(3, 2), y)})
    assert rs.data_hashes({"a": (x, y)}) != rs.data_hashes({"a": (x.view(torch.int32), y)})
