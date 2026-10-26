"""Dual-source preparation ownership; preserve delivered-v1 markers verbatim."""
import hashlib
import json
from pathlib import Path
import time
from control_common import PACKAGE, STUDY_SHA, protocol, read, dump, sha, regular, require, safe_artifact, disjoint

LEGACY_MANIFEST = "acd3b17a4e0b8697dda464221c1e6ff1d2bc2f1ae1112848b8cae16f94136168"


def owner_core(source, target, base, expected):
    return dict(schema="compact80_stepmatched_prepare_v2", source=str(source), study=str(target),
                base_package_source=str(base), base_package_hashes=protocol()["base_package_hashes"],
                control_protocol_sha256=sha(PACKAGE / "protocol.json"),
                package_manifest_sha256=expected, gpu="0", formal_jobs=3)


def legacy_owner(target, source):
    wanted = dict(schema="compact80_stepmatched_prepare_v1", source=str(source), study=str(target),
                  package_manifest_sha256=LEGACY_MANIFEST, gpu="0", formal_jobs=3)
    path = regular(target / "control_owner.json")
    expected_bytes = (json.dumps(wanted, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    require(read(path) == wanted and path.read_bytes() == expected_bytes,
            "Legacy owner differs from canonical delivered-v1 record; preserve and report")
    return sha(path)


def validate_partial(target, expected_files, own_files):
    allowed = dict(expected_files)
    allowed.update(own_files)
    parents = set()
    for rel in allowed:
        parents.update(p.as_posix() for p in Path(rel).parents if p != Path("."))
    inventory = []
    for path in sorted(target.rglob("*")):
        rel = path.relative_to(target).as_posix()
        require(not path.is_symlink(), "Partial contains a symlink")
        if path.is_dir():
            require(rel in parents, "Unexpected partial directory (training/views/console are not adoptable): " + rel)
            continue
        require(rel in allowed, "Unexpected partial file; no adoption: " + rel)
        digest = sha(regular(path))
        require(digest == allowed[rel], "Partial bytes differ; no overwrite: " + rel)
        inventory.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest))
    return inventory


def validate_owner(target, core):
    owner_path = regular(target / "control_owner_v2.json")
    owner = read(owner_path)
    require(set(owner) == {"core", "legacy_adoption"} and owner["core"] == core, "V2 owner identity differs")
    adoption = owner["legacy_adoption"]
    if adoption is None:
        require(not (target / "control_owner.json").exists(), "Unexpected legacy owner in fresh v2 study")
    else:
        old_sha = legacy_owner(target, Path(core["source"]))
        require(adoption["schema"] == "compact80_v1_partial_adoption_v1"
                and adoption["old_package_manifest_sha256"] == LEGACY_MANIFEST
                and adoption["new_package_manifest_sha256"] == core["package_manifest_sha256"]
                and adoption["original_owner_sha256"] == old_sha
                and adoption["original_owner_preserved"] is True
                and adoption["training_started"] is False,
                "Legacy-to-v2 adoption binding differs")
        for row in adoption["original_partial_files"]:
            path = safe_artifact(target, row["path"])
            require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Adopted original partial changed")
    return owner


def establish_owner(target, core, expected_files, resume_partial, adopt_v1_partial):
    if not target.exists():
        target.mkdir(exist_ok=False)
        owner = dict(core=core, legacy_adoption=None)
        dump(target / "control_owner_v2.json", owner)
        return owner
    if (target / "control_owner_v2.json").exists():
        owner = validate_owner(target, core)
        if (target / "control_prepared.json").exists():
            return owner
        require(resume_partial, "V2 partial preserved; explicitly --resume-partial after inspection")
        own = {"control_owner_v2.json": sha(target / "control_owner_v2.json")}
        if owner["legacy_adoption"] is not None:
            own["control_owner.json"] = owner["legacy_adoption"]["original_owner_sha256"]
        if (target / "prepare_evidence.json").exists():
            evidence = read(regular(target / "prepare_evidence.json"))
            require(evidence["package_manifest_sha256"] == core["package_manifest_sha256"]
                    and evidence["control_owner_v2_sha256"] == sha(target / "control_owner_v2.json")
                    and evidence["no_training_started"] is True, "Partial prepare evidence differs")
            own["prepare_evidence.json"] = sha(target / "prepare_evidence.json")
        validate_partial(target, expected_files, own)
        return owner
    require(resume_partial and adopt_v1_partial, "Legacy partial requires --resume-partial --adopt-v1-partial")
    require(not (target / "control_prepared.json").exists(), "Completed v1 study is not a no-training partial")
    old_sha = legacy_owner(target, Path(core["source"]))
    inventory = validate_partial(target, expected_files, {"control_owner.json": old_sha})
    owner = dict(core=core, legacy_adoption=dict(schema="compact80_v1_partial_adoption_v1",
        old_package_manifest_sha256=LEGACY_MANIFEST, new_package_manifest_sha256=core["package_manifest_sha256"],
        original_owner_sha256=old_sha, original_owner_preserved=True, training_started=False,
        observed_unix_time=time.time(), original_partial_files=inventory,
        note="New preparation-time compatibility receipt; does not relabel any historical CUDA execution"))
    dump(target / "control_owner_v2.json", owner)
    require(sha(target / "control_owner.json") == old_sha, "Legacy owner was altered")
    return owner


def prepared_body(target, owner):
    adoption = owner["legacy_adoption"]
    return dict(owner["core"], control_owner_v2_sha256=sha(target / "control_owner_v2.json"),
                legacy_owner_sha256=adoption["original_owner_sha256"] if adoption else None)


def verify_prepared_ownership(target, expected):
    record = read(regular(target / "control_prepared.json"))
    require(record["source"] == protocol()["source_study"], "Prepared source is not original study")
    source, base = Path(record["source"]), Path(record["base_package_source"])
    require(source.is_absolute() and base.is_absolute(), "Recorded source/package paths must be absolute")
    for protected in (source, base, PACKAGE):
        disjoint(target, protected)
    core = owner_core(source, target, base, expected)
    owner = validate_owner(target, core)
    require(record == prepared_body(target, owner), "Preparation completion does not bind v2 ownership")
    for rel, digest in core["base_package_hashes"].items():
        require(sha(safe_artifact(target / "run_package", rel)) == digest, "Prepared execution package changed")
    return record
