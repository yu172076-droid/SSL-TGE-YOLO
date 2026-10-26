"""Whitelist-only compact80 control: prepare -> 2-epoch smoke -> 3 formal runs."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace
import resource_base as resources
from control_common import *
from prepare_layout import owner_core, establish_owner, prepared_body, verify_prepared_ownership


def frozen(root):
    require(sha(regular(root / "study.json")) == STUDY_SHA and sha(regular(root / "manifest.json")) == DATA_SHA,
            "Frozen study identity differs")
    info = read(root / "study.json")
    require(len(info["frozen_hashes"]) == 3238, "Frozen inventory differs")
    for rel, digest in info["frozen_hashes"].items():
        require(sha(safe_artifact(root, rel)) == digest, "Frozen bytes changed: " + rel)
    return info


def checked_copy(source, dest, digest, image=False):
    regular(source)
    require(sha(source) == digest, "Source changed before copy")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        require(sha(regular(dest)) == digest, "Existing partial differs; no overwrite")
        return
    if image:
        try:
            os.link(str(source), str(dest))
        except OSError:
            shutil.copy2(source, dest)
    else:
        shutil.copy2(source, dest)
    require(sha(dest) == digest and sha(source) == digest, "Copy/source stability check failed")


def check_baselines(root, source=None):
    for row in protocol()["reference_files"]:
        path = safe_artifact(root / "baseline", row["path"])
        require(sha(path) == row["sha256"], "Copied reference evidence changed")
    if source is not None:
        for row in protocol()["reference_weights"]:
            require(sha(safe_artifact(source, row["path"])) == row["sha256"], "Original baseline weight changed")


def prepare(args):
    require(args.source is not None, "--source required for preparation")
    require(args.base_package is not None, "--base-package required; original study does not contain run_package")
    source, target, base = args.source.absolute(), args.study.absolute(), args.base_package.absolute()
    require(str(source) == protocol()["source_study"], "Use the declared original study only")
    require(source.resolve() == source and target.resolve() == target and base.resolve() == base, "Study/package paths must be real")
    for other in (source, base, PACKAGE, Path(read(source / "study.json")["source_project"])):
        disjoint(target, other)
    resources.free_check(target.parent)
    info = frozen(source)
    # Resolve and verify ALL actual source locations before creating/adopting a target.
    inputs = []
    for rel, digest in dict(info["frozen_hashes"], **{"study.json": STUDY_SHA}).items():
        inputs.append((safe_artifact(source, rel), rel, digest, rel.startswith("images/")))
    for rel, digest in protocol()["base_package_hashes"].items():
        inputs.append((safe_artifact(base, rel), "run_package/" + rel, digest, False))
    for row in protocol()["reference_files"]:
        inputs.append((safe_artifact(source, row["path"]), "baseline/" + row["path"], row["sha256"], False))
    expected_files = {rel: digest for origin, rel, digest, image in inputs}
    require(len(expected_files) == len(inputs), "Preparation destinations collide")
    for origin, rel, digest, image in inputs:
        require(sha(origin) == digest, "Preparation input differs before target write: " + str(origin))
    for row in protocol()["reference_weights"]:
        require(sha(safe_artifact(source, row["path"])) == row["sha256"], "Original baseline weight differs")
    core = owner_core(source, target, base, args.expected_package_sha256)
    owner = establish_owner(target, core, expected_files, args.resume_partial, args.adopt_v1_partial)
    if (target / "control_prepared.json").is_file():
        verify_prepared_ownership(target, args.expected_package_sha256)
        frozen(target)
        check_baselines(target, source)
        print("PREPARED_REUSE", target)
        return
    for origin, rel, digest, image in inputs:
        checked_copy(origin, target / rel, digest, image=image)
    frozen(source)
    frozen(target)
    check_baselines(target, source)
    for origin, rel, digest, image in inputs:
        require(sha(regular(origin)) == digest and sha(safe_artifact(target, rel)) == digest, "Final input/copy mismatch")
    evidence = target / "prepare_evidence.json"
    if not evidence.exists():
        dump(evidence, dict(completed_unix_time=time.time(), no_training_started=True,
             source_project_not_written=True, frozen_files_verified=len(info["frozen_hashes"]),
             copied_files_verified=len(inputs), source_and_copy_SHA_verified=True,
             base_package_source=str(base), base_package_hashes=protocol()["base_package_hashes"],
             control_owner_v2_sha256=sha(target / "control_owner_v2.json"),
             legacy_owner_sha256=prepared_body(target, owner)["legacy_owner_sha256"],
             hardlinks_only_for_frozen_images=True, package_manifest_sha256=args.expected_package_sha256))
    # Completion is committed last; interrupted copies retain their original owners.
    dump(target / "control_prepared.json", prepared_body(target, owner))
    verify_prepared_ownership(target, args.expected_package_sha256)
    print("PREPARE_COMPLETE", target)


def make_study(root, args, binding):
    base_dir = root / "run_package"
    sys.path.insert(0, str(base_dir))
    legacy = load("pinned_stepmatched_study", base_dir / "run_internal_study.py")
    options = SimpleNamespace(device="0", workers=4, retry_failed=args.retry_failed, v8_python=None)

    class ControlStudy(legacy.Study):
        def task(self, identifier, config):
            config = dict(config, study=str(root), physical_gpu="0", device="0", gpu_binding=binding,
                package_manifest_sha256=args.expected_package_sha256,
                control_protocol_sha256=sha(PACKAGE / "protocol.json"),
                worker_sha256=sha(PACKAGE / "control_worker.py"), runner_sha256=sha(__file__),
                legacy_worker_sha256=sha(base_dir / "study_worker.py"),
                study_protocol_sha256=sha(root / "study.json"))
            folder = root / "jobs" / identifier
            record_path = folder / "completed.json"
            if record_path.is_file():
                saved = read(record_path)
                wanted = dict(config, gpu_binding=saved["input"]["gpu_binding"])
                require(wanted == saved["input"] and saved["input"]["gpu_binding"]["uuid"] == binding["uuid"],
                        "Completed task protocol/GPU differs")
                for rel, digest in saved["output_hashes"].items():
                    require(sha(safe_artifact(folder, rel)) == digest, "Completed task output changed")
                work = Path(saved["work"])
                require(work.parent == folder and work.resolve() == work, "Completed work escaped job")
                return work
            resources.free_check(root)
            resources.gpu_check("0", idle=True)
            folder.mkdir(parents=True, exist_ok=True)
            attempts = sorted(folder.glob("attempt_*"))
            require(not attempts or args.retry_failed, "Partial attempt preserved; explicit retry starts fresh, never epoch-resumes")
            work = folder / "attempt_{:02d}".format(len(attempts) + 1)
            work.mkdir(exist_ok=False)
            full = dict(config, work=str(work))
            dump(work / "worker_config.json", full)
            env = os.environ.copy()
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
                env.pop(key, None)
            env.update(CUDA_VISIBLE_DEVICES="0", CUDA_DEVICE_ORDER="PCI_BUS_ID", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", WANDB_MODE="disabled", YOLO_AUTOINSTALL="false",
                       MPLCONFIGDIR=str(work / "plot_config"), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
            command = [sys.executable, "-B", str(PACKAGE / "control_worker.py"), "--config", str(work / "worker_config.json"),
                       "--expected-package-sha256", args.expected_package_sha256]
            dump(work / "started.json", dict(command=command, unix_time=time.time()))
            print("START", identifier, work / "output.log", flush=True)
            require(resources.run_child(command, work, env, root) == 0, "Worker failed; queue stopped")
            required = ["worker_config.json", "started.json", "worker_execution.json", "exit_status.json", "parent_resource_checks.json"]
            required += (["train/weights/init.pt", "train/weights/best.pt", "train/weights/last.pt", "train/results.txt",
                          "train/opt.yaml", "train/hyp.yaml", "training_settings.json", "seed_applied.json",
                          "worker_retention_audit.json", "optimizer_steps.jsonl", "optimizer_step_summary.json"]
                         if config["mode"] == "train" else ["predictions.json", "inference_metadata.json"])
            hashes = {p.relative_to(folder).as_posix(): sha(p) for p in [regular(work / rel) for rel in required]}
            dump(record_path, dict(input=config, work=str(work), output_hashes=hashes))
            print("DONE", identifier, flush=True)
            return work

        def evaluate(self, identifier, backend, weight, split, imgsz):
            super().evaluate(identifier, backend, weight, split, imgsz)
            job = identifier + "_eval_" + split + "_640"
            completed = root / "jobs" / job / "completed.json"
            record = read(completed)
            metrics = Path(record["work"]) / "coco_metrics.json"
            body = read(metrics)
            old = body.get("control_wrapper_manifest_sha256")
            require(old is None or old == args.expected_package_sha256, "Metric wrapper identity changed")
            body["control_wrapper_manifest_sha256"] = args.expected_package_sha256
            if identifier.startswith("smoke_"):
                body["main_checkpoint_rule"] = "Nonformal 2-epoch smoke last; excluded from formal statistics"
            dump(metrics, body)
            record["output_hashes"][metrics.relative_to(completed.parent).as_posix()] = sha(metrics)
            dump(completed, record)

    return ControlStudy(root, options)


def export(root, returns, phase, outer_log, outer_exit):
    from control_audit import audit_training
    frozen(root)
    check_baselines(root)
    for item in plan(phase):
        audit_training(root, item)
    require(outer_log is not None and outer_exit is not None, "Completed phase outer log/exit required")
    require(outer_log.parent == root / "console" and outer_exit.parent == root / "console", "Outer evidence escaped console")
    require(read(regular(outer_exit)) == {"python_exit": 0, "tee_exit": 0}, "Actual shell exits not both zero")
    marker = "STEPMATCHED_{}_COMPLETE".format(phase.upper())
    require(marker in regular(outer_log).read_text(encoding="utf-8"), "Missing actual phase completion marker")
    out = returns / ("compact80_stepmatched_{}_{}".format(phase, time.time_ns()))
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    frozen_keys = set(read(root / "study.json")["frozen_hashes"])
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.suffix in (".pt", ".cache") or "__pycache__" in path.parts or path.name.endswith(".lock"):
            continue
        if rel in frozen_keys and "/" in rel and not rel.startswith(("source/", "configs/")):
            continue
        if rel.startswith("views/") and path.name not in ("view.json", "data.yaml"):
            continue
        digest = sha(regular(path))
        checked_copy(path, out / rel, digest)
        entries.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest))
    for path in PACKAGE.iterdir():
        if path.is_file():
            rel = "control_package/" + path.name
            digest = sha(path)
            checked_copy(path, out / rel, digest)
            entries.append(dict(path=rel, bytes=path.stat().st_size, sha256=digest))
    dump(out / "export_manifest.json", dict(phase=phase, files=entries, file_count=len(entries),
         raw_images_and_checkpoints_omitted=True, actual_outer_exit_verified=True,
         package_manifest_sha256=sha(PACKAGE / "delivery_manifest.json")))
    (out / "export_manifest.sha256").write_text(sha(out / "export_manifest.json") + "  export_manifest.json\n", encoding="ascii")
    print("CONTROL_RETURN", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "smoke", "formal", "export"))
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--base-package", type=Path)
    parser.add_argument("--expected-package-sha256", required=True)
    parser.add_argument("--return-root", type=Path)
    parser.add_argument("--export-phase", choices=("smoke", "formal"))
    parser.add_argument("--outer-log", type=Path)
    parser.add_argument("--outer-exit", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--adopt-v1-partial", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    verify_package(args.expected_package_sha256)
    print(__import__("json").dumps(dict(phase=args.phase, gpu="0", formal_training_jobs=3,
          queue=plan(args.phase) if args.phase in ("smoke", "formal") else [], execute=args.execute), ensure_ascii=False), flush=True)
    if not args.execute:
        print("PLAN_ONLY_NO_GPU_OR_WRITES")
        return
    require(sys.platform == "linux", "Execution requires laboratory Linux")
    root = args.study.absolute()
    require(root.resolve() == root, "Study symlink refused")
    with resources.Lock(root.parent / ".compact80_stepmatched_locks" / "gpu0.lock"):
        if args.phase == "prepare":
            prepare(args)
            return
        meta = verify_prepared_ownership(root, args.expected_package_sha256)
        source = Path(meta["source"])
        frozen(root)
        check_baselines(root, source)
        for protected in (root, source, PACKAGE, Path(read(root / "study.json")["source_project"])):
            if args.return_root is not None:
                require(args.return_root.resolve() == args.return_root.absolute(), "Return root symlink refused")
                disjoint(args.return_root, protected)
        if args.phase == "export":
            require(args.return_root is not None and args.export_phase is not None, "Export location/phase required")
            export(root, args.return_root.absolute(), args.export_phase, args.outer_log, args.outer_exit)
            return
        from control_audit import audit_training
        if args.phase == "formal":
            for item in plan("smoke"):
                audit_training(root, item)
            require((root / "smoke_gate.json").is_file() and read(root / "smoke_gate.json")["package_sha256"] == args.expected_package_sha256,
                    "New instrumented smoke gate missing")
            verify_formal_release(root, args.expected_package_sha256)
        resources.free_check(root)
        binding = resources.gpu_check("0", idle=True)
        study = make_study(root, args, binding)
        for position, item in enumerate(plan(args.phase), 1):
            ids = expected_ids(root, item["seed"])
            data = study.make_view(item["job"], ids)
            config = {k: item[k] for k in ("architecture", "budget", "seed", "method", "epochs")}
            config.update(mode="train", backend="v7", data=str(data), data_view_sha256=sha(data.parent / "view.json"),
                          batch=8, workers=4, device="0", dataset_manifest_sha256=DATA_SHA, n_training_samples=len(ids))
            work = study.task(item["job"], config)
            for split in ("val", "test"):
                study.evaluate(item["job"], "v7", work / "train/weights/last.pt", split, 640)
            audit = audit_training(root, item)
            dump(root / (args.phase + "_progress.json"), dict(verified_jobs=position, planned_jobs=len(plan(args.phase)),
                 last_job=item["job"], audit=audit, package_sha256=args.expected_package_sha256))
        frozen(root)
        frozen(source)
        check_baselines(root, source)
        if args.phase == "smoke":
            dump(root / "smoke_gate.json", dict(package_sha256=args.expected_package_sha256, instrumentation_passed=True,
                 nonformal_epochs=2, source_schedule_reconstruction_only_for_historical_SSL=True))
        print("STEPMATCHED_{}_COMPLETE".format(args.phase.upper()), flush=True)


if __name__ == "__main__":
    def interrupted(*args):
        raise KeyboardInterrupt("Signal received; only own worker group is stopped")
    signal.signal(signal.SIGTERM, interrupted)
    main()
