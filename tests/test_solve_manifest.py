"""The solve manifest and the provenance a training run records: what a rebuild needs, and which deletions are safe."""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import solve_manifest as sm  # noqa: E402
from cfr.provenance import code_version, file_digest, provenance  # noqa: E402


def solve(directory, name, **args):
    stem = os.path.join(directory, name)
    with open(stem + ".pkl", "wb") as handle:
        handle.write(os.urandom(64))
    with open(stem + ".flat.npz", "wb") as handle:
        handle.write(os.urandom(32))
    with open(stem + ".json", "w") as handle:
        json.dump({"args": args, "results": {}, "information_sets_reached": 1, "seconds": 5}, handle)
    return stem


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A results/cfr with a root solve, a leaf warm-started from it, an orphan, and a live ladder linking the leaf."""
    cfr = tmp_path / "results" / "cfr"
    x = cfr / "experiments"
    x.mkdir(parents=True)
    root = solve(str(x), "root", seed=0)
    leaf = solve(str(x), "leaf", warm_start=root + ".pkl", seed=0)
    solve(str(x), "orphan", seed=1)
    live = cfr / "ladder_live"
    live.mkdir()
    os.symlink(leaf + ".pkl", live / "cap2_70bb.pkl")
    monkeypatch.setattr(sm, "ROOT", str(tmp_path))
    monkeypatch.setattr(sm, "CFR", str(cfr))
    monkeypatch.setattr(sm, "inferred_commit", lambda started: {"commit": "abc", "inferred": True})
    out = tmp_path / "MANIFEST.json"
    return {"tmp": tmp_path, "x": x, "out": out, "root": root, "leaf": leaf}


def run(tree, *extra, capsys=None):
    sys.argv = ["solve_manifest.py", "--out", str(tree["out"]), *extra]
    sm.main()
    return json.load(open(tree["out"]))["solves"] if not extra or extra[0] != "--deletable" else None


def test_every_solve_with_its_files_sources_users_and_code(tree):
    solves = run(tree)
    key = lambda name: os.path.relpath(str(tree["x"] / name), str(tree["tmp"]))
    assert set(solves) == {key("root"), key("leaf"), key("orphan")}
    leaf, root = solves[key("leaf")], solves[key("root")]
    assert set(leaf["files"]) == {".pkl", ".flat.npz", ".json"}
    assert leaf["files"][".pkl"]["blake2b"] == file_digest(tree["leaf"] + ".pkl")
    assert leaf["sources"]["warm_start"] == {"stem": key("root"), "present": True}
    assert root["built_on_by"] == [key("leaf")]
    assert leaf["live"] and not root["live"]
    assert leaf["code"]["inferred"] and not leaf["recorded"]


def test_only_the_leaves_nothing_live_or_built_on_uses_are_deletable(tree, capsys):
    run(tree)
    run(tree, "--deletable")
    out = capsys.readouterr().out
    assert "orphan" in out and "1 leaves" in out
    assert "experiments/root" not in out and "experiments/leaf" not in out   # built on, and live


def test_a_rerun_hashes_only_what_changed(tree, monkeypatch):
    run(tree)
    hashed = []
    monkeypatch.setattr(sm, "file_digest", lambda path: hashed.append(path) or "x")
    run(tree)
    assert hashed == []
    with open(tree["root"] + ".pkl", "ab") as handle:
        handle.write(b"retrained")
    os.utime(tree["root"] + ".pkl", (1, 1))
    run(tree)
    assert hashed == [tree["root"] + ".pkl"]


def test_a_recorded_commit_is_used_and_not_inferred(tree):
    with open(tree["leaf"] + ".json") as handle:
        summary = json.load(handle)
    summary["provenance"] = {"code": {"commit": "deadbeef", "dirty": False}, "command": ["train"]}
    with open(tree["leaf"] + ".json", "w") as handle:
        json.dump(summary, handle)
    leaf = run(tree)[os.path.relpath(tree["leaf"], str(tree["tmp"]))]
    assert leaf["recorded"] and leaf["code"]["commit"] == "deadbeef" and leaf["command"] == ["train"]


def test_provenance_names_this_checkout_s_commit_and_digests_each_source(tmp_path):
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert code_version(ROOT)["commit"] == head
    source = tmp_path / "w.pkl"
    source.write_bytes(b"warm")
    record = provenance({"warm_start": str(source), "abstraction_from": str(tmp_path / "gone.pkl")}, ROOT)
    assert record["sources"]["warm_start"]["blake2b"] == file_digest(str(source))
    assert record["sources"]["abstraction_from"]["missing"]
    assert record["code"]["commit"] == head and record["started"]


def test_dirty_files_are_named_whole_and_data_does_not_make_a_run_dirty(tmp_path):
    repo = str(tmp_path)
    git = lambda *a: subprocess.run(["git", "-C", repo, *a], capture_output=True, check=True)
    git("init", "-q"); git("config", "user.email", "t@t"); git("config", "user.name", "t")
    os.makedirs(os.path.join(repo, "cfr")); os.makedirs(os.path.join(repo, "results"))
    for f in ("cfr/solver.py", "results/ledger.md"):
        open(os.path.join(repo, f), "w").write("a")
    git("add", "."); git("commit", "-qm", "x")
    open(os.path.join(repo, "results/ledger.md"), "w").write("b")
    assert code_version(repo)["dirty"] is False and code_version(repo)["data_changed"] == ["results/ledger.md"]
    open(os.path.join(repo, "cfr/solver.py"), "w").write("b")
    assert code_version(repo)["dirty_files"] == ["cfr/solver.py"]
