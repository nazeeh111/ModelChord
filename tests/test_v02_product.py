"""Public v0.2 CLI/workbench contracts on tiny synthetic data only.

Numerical search behavior is covered by the frozen v0.2 tests. These tests check
request rejection, planning-floor accounting, no-search boundaries and packaging.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
import torch

from modelchord import Artifact, MLP, export_artifact
from modelchord import cli
import modelchord.server as server


@pytest.fixture
def product_project(tmp_path):
    model = MLP(6, [6, 5], ["0", "1", "2", "3"])
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    contract = {"family": "classification", "representation": "product-test/v1",
                "preprocessing": "identity", "tokenizer": None,
                "labels": model.spec["classes"]}
    ordinary = export_artifact(tmp_path / "artifacts/base", model, contract)
    specialist = export_artifact(tmp_path / "artifacts/expert", model, contract,
                                 base_id=ordinary.manifest["base_id"])
    caps = [{"name": "clean", "floor": .6}, {"name": "other", "floor": .74}]
    study = tmp_path / "study"
    study.mkdir()
    caps_path = study / "capabilities.json"
    caps_path.write_text(json.dumps(caps))
    config = {"base": {"id": "base", "path": "artifacts/base"},
              "experts": [{"id": "expert", "path": "artifacts/expert", "task": "other"}],
              "capabilities": caps}
    (study / "workbench.json").write_text(json.dumps(config))
    arrays = {}
    for index, name in enumerate(["clean", "other"]):
        arrays[f"cal_x_{name}"] = np.full((3, 6), .1 + index / 10, np.float32)
        arrays[f"cal_y_{name}"] = np.array([0, 1, 2], np.int64)
        arrays[f"x_{name}"] = np.full((4, 6), .6 + index / 10, np.float32)
        arrays[f"y_{name}"] = np.array([0, 0, 0, 1], np.int64)
    data_path = study / "workbench.npz"
    np.savez(data_path, **arrays)
    return {"path": tmp_path, "ordinary": ordinary, "specialist": specialist,
            "caps": caps_path, "data": data_path}


def cli_args(project, output):
    return ["repair-v2", str(project["ordinary"].path), "--data", str(project["data"]),
            "--capabilities", str(project["caps"]), "--output", str(output)]


@pytest.mark.parametrize("option,value", [
    ("--rank", "-1"), ("--rank", "65"),
    ("--stages", "-1"), ("--stages", "9"),
    ("--steps", "0"), ("--steps", "65"),
    ("--planning-margin", "-.001"), ("--planning-margin", ".101"),
    ("--planning-margin", "nan"), ("--planning-margin", "inf"),
])
def test_cli_v2_invalid_budget_rejected_without_creating_artifact(product_project, tmp_path, capsys, option, value):
    output = tmp_path / "rejected"
    args = cli_args(product_project, output) + ["--rank", "0", "--stages", "0", option, value]
    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["error"]
    assert not output.exists()


@pytest.mark.parametrize("rank,stages", [(0, 4), (1, 4), (8, 0)])
def test_cli_v2_no_search_budget_keeps_requested_contract(product_project, tmp_path, capsys, rank, stages):
    output = tmp_path / "no-search"
    args = cli_args(product_project, output) + ["--rank", str(rank), "--stages", str(stages),
                                              "--planning-margin", ".1", "--steps", "1"]
    assert cli.main(args) == 0
    response = json.loads(capsys.readouterr().out)
    report = response["report"]
    assert report["planning_margin"] == 0
    assert report["requested_floors"] == {"clean": .6, "other": .74}
    assert report["effective_floors"] == report["requested_floors"]
    assert report["trial_count"] == 0
    assert report["rank_used_upper_bound"] == 0
    saved = Artifact.open(response["artifact"])
    assert json.loads((saved.path / "build-report.json").read_text()) == report
    assert report["required_fraction_floors_met"] == 1.
    for name, tensor in product_project["ordinary"].state().items():
        torch.testing.assert_close(saved.tensor(name), tensor, atol=0, rtol=0)


@pytest.mark.parametrize("steps", [1, 64])
def test_cli_v2_accepts_public_step_boundaries_without_unnecessary_training(product_project, tmp_path, capsys, steps):
    # Easy requirements exercise the ordinary fast path even at the public
    # maximum rank/stage limits; no candidate trajectory needs to be trained.
    product_project["caps"].write_text(json.dumps([{"name": name, "floor": 0.} for name in ["clean", "other"]]))
    args = cli_args(product_project, tmp_path / "easy") + ["--rank", "64", "--stages", "8", "--steps", str(steps), "--planning-margin", "0"]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)["report"]
    assert report["budget"]["steps_per_trial"] == steps
    assert report["budget"]["rank"] == 64
    assert report["trial_count"] == 0


def build_request(**overrides):
    request = {"experts": ["expert"], "floors": {"clean": .6, "other": .74},
               "method": "task_arithmetic", "scale": .5, "repair": True,
               "repair_algorithm": "v2", "planning_margin": .1,
               "budget": {"rank": 6, "rounds": 4, "steps": 1}}
    request.update(overrides)
    return request


@pytest.mark.parametrize("key,dtype", [("y_clean", np.float32), ("cal_y_other", np.float64), ("x_clean", np.float64)])
def test_cli_v2_preserves_invalid_data_types_for_rejection(product_project, tmp_path, capsys, key, dtype):
    with np.load(product_project["data"], allow_pickle=False) as archive:
        arrays = {n: archive[n].copy() for n in archive.files}
    arrays[key] = arrays[key].astype(dtype)
    if "y_" in key:
        arrays[key][0] = .75
    np.savez(product_project["data"], **arrays)
    output = tmp_path / "invalid-types"
    assert cli.main(cli_args(product_project, output) + ["--rank", "0", "--stages", "0"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]
    assert not output.exists()


@pytest.fixture
def search_spy(monkeypatch):
    calls = []

    def fake_search(ordinary, calibration, selection, capabilities, output, *, budget):
        # Fit behavior is deliberately replaced, while real safe export and the
        # surrounding product's normalization/packaging continue to execute.
        metrics = {name: server.classification(ordinary.model(), *batch) for name, batch in selection.items()}
        floors = {cap.name: cap.floor for cap in capabilities}
        met = sum(metrics[name]["accuracy"] >= floor for name, floor in floors.items())
        report = {"algorithm": "product-interface-spy", "effective_floors": floors,
                  "trial_count": 6, "rank_used_upper_bound": 0,
                  "fraction_floors_met": met / len(floors),
                  "status": "acceptable candidate found" if met == len(floors) else "no acceptable candidate found"}
        calls.append({"budget": budget, "capabilities": capabilities,
                      "calibration": calibration, "selection": selection})
        artifact = export_artifact(output, ordinary.model(), ordinary.manifest["contract"],
                                   base_id=ordinary.manifest["base_id"], provenance={"interface_test": True})
        return artifact, report

    monkeypatch.setattr(server, "search", fake_search)
    return calls


def test_workbench_v2_keeps_requested_floors_distinct_from_planning_targets(product_project, search_spy):
    workbench = server.Workbench(product_project["path"])
    result = workbench.build(build_request())
    report = result["report"]
    assert report["requested_floors"] == {"clean": .6, "other": .74}
    assert report["effective_selection_floors"] == pytest.approx({"clean": .7, "other": .84})
    assert report["planning_margin"] == .1
    assert report["required_fraction_floors_met"] == 1.
    assert report["fraction_floors_met"] == .5
    assert report["status"] == "no acceptable candidate found"
    assert report["attempts"] == 6
    budget = search_spy[0]["budget"]
    assert (budget.rank, budget.stages, budget.steps_per_trial) == (6, 3, 1)
    assert set(search_spy[0]["calibration"]) == {"clean", "other"}
    assert set(search_spy[0]["selection"]) == {"clean", "other"}
    # The downloadable report must retain both contracts, rather than allowing
    # a failed planning target to be mistaken for a failed requested floor.
    with zipfile.ZipFile(workbench.archives[result["id"]]) as archive:
        packed = json.loads(archive.read("build-report.json"))
        assert packed["requested_floors"] == report["requested_floors"]
        assert packed["effective_selection_floors"] == report["effective_selection_floors"]
        assert packed["required_fraction_floors_met"] == 1.


def test_workbench_caps_planning_targets_at_one(product_project, search_spy):
    workbench = server.Workbench(product_project["path"])
    result = workbench.build(build_request(floors={"clean": .98, "other": .4}, planning_margin=.03))
    assert result["report"]["effective_selection_floors"] == pytest.approx({"clean": 1., "other": .43})
    assert {cap.name: cap.floor for cap in search_spy[0]["capabilities"]} == pytest.approx({"clean": 1., "other": .43})


@pytest.mark.parametrize("rank,rounds", [(0, 4), (1, 4), (8, 0)])
def test_workbench_real_zero_search_budget_has_no_planning_buffer(product_project, rank, rounds):
    workbench = server.Workbench(product_project["path"])
    result = workbench.build(build_request(budget={"rank": rank, "rounds": rounds, "steps": 1}))
    report = result["report"]
    assert report["planning_margin"] == 0.
    assert report["attempts"] == 0
    assert report["effective_selection_floors"] == report["requested_floors"]
    assert report["required_fraction_floors_met"] == 1.
    for name, tensor in product_project["ordinary"].state().items():
        torch.testing.assert_close(Artifact.open(workbench.paths[result["id"]]).tensor(name), tensor, atol=0, rtol=0)


def test_workbench_ordinary_build_has_no_planning_buffer_or_repair_dispatch(product_project, monkeypatch):
    def unexpected_search(*args, **kwargs):
        pytest.fail("Ordinary merge must not dispatch repair")
    monkeypatch.setattr(server, "search", unexpected_search)
    result = server.Workbench(product_project["path"]).build(build_request(repair=False))
    assert result["report"]["planning_margin"] == 0.
    assert result["report"]["requested_floors"] == {"clean": .6, "other": .74}
    assert result["report"]["attempts"] == 0
    assert result["report"]["required_fraction_floors_met"] == 1.
    assert "repair" not in result["label"]


@pytest.mark.parametrize("override", [
    {"repair_algorithm": "v3"},
    {"repair": "yes"},
    {"planning_margin": -.01}, {"planning_margin": .11},
    {"planning_margin": float("nan")}, {"planning_margin": float("inf")},
    {"planning_margin": True},
    {"budget": {"unknown": 1}},
    {"budget": {"rank": True}}, {"budget": {"rank": 2.5}},
    {"budget": {"rank": -1}}, {"budget": {"rank": 65}},
    {"budget": {"rounds": 9}},
    {"budget": {"steps": 0}}, {"budget": {"steps": 65}},
    {"budget": {"steps": True}},
    {"experts": ["missing"]}, {"experts": ["expert", "expert"]},
    {"floors": {"clean": .5}},
    {"floors": {"clean": float("nan"), "other": .5}},
    {"floors": {"clean": True, "other": .5}},
])
def test_workbench_rejects_invalid_product_request_before_writing_outputs(product_project, monkeypatch, override):
    def unexpected_merge(*args, **kwargs):
        pytest.fail("Invalid requests must be rejected before model construction")
    monkeypatch.setattr(server, "merge", unexpected_merge)
    workbench = server.Workbench(product_project["path"])
    with pytest.raises(ValueError):
        workbench.build(build_request(**override))
    assert not (product_project["path"] / "exports").exists()
