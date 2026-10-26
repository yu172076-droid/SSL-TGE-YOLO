"""Audit one control training plus its fixed val/test evaluations."""
import math
from pathlib import Path
import re
from control_common import *


def task_record(root, job):
    folder = root / "jobs" / job
    record = read(regular(folder / "completed.json"))
    work = Path(record["work"])
    require(work.parent == folder and work.resolve() == work, "Completed task escaped its job")
    for rel, digest in record["output_hashes"].items():
        require(sha(safe_artifact(folder, rel)) == digest, "Completed output changed: " + job)
    config = read(regular(work / "worker_config.json"))
    require(config == dict(record["input"], work=str(work)), "Worker configuration differs from completion")
    validate_config(config, work / "worker_config.json", record["input"]["package_manifest_sha256"])
    require(read(work / "exit_status.json")["returncode"] == 0, "Parent-observed worker exit nonzero")
    execution = read(work / "worker_execution.json")
    require(execution["actual_exitcode"] == 0 and execution["status"] == "complete"
            and execution["config_sha256"] == sha(work / "worker_config.json")
            and execution["package_sha256"] == config["package_manifest_sha256"]
            and execution["environment_verified_before_torch"] is True, "Worker execution binding failed")
    device = execution["torch_device_binding"]
    require(device["physical_gpu"] == "0" and device["visible_devices"] == 1
            and device["uuid_verification"] in ("matched_parent_uuid", "unavailable_environment_binding_only"),
            "CUDA binding evidence invalid")
    if device["uuid_verification"] == "matched_parent_uuid":
        from retention_base import verify_runtime_uuid
        verify_runtime_uuid(device["uuid"], config["gpu_binding"]["uuid"])
    else:
        require(device["uuid"] is None, "Unavailable UUID scope must have no invented runtime UUID")
    parent = read(work / "parent_resource_checks.json")
    require(parent["actual_child_exitcode"] == 0 and parent["error"] is None
            and parent["samples"] and all(r["free_bytes"] >= 6 * 1024 ** 3 for r in parent["samples"]),
            "Resource/parent exit evidence failed")
    return record, config, work


def mean_valid(array):
    values = array[array >= 0]
    return float(values.mean()) if values.size else None


def audit_eval(root, job, split, checkpoint, expected_package):
    import numpy as np
    record, config, work = task_record(root, job + "_eval_" + split + "_640")
    meta = read(work / "inference_metadata.json")
    metrics = read(work / "coco_metrics.json")
    require(meta["checkpoint_sha256"] == checkpoint == config["weight_sha256"] == metrics["checkpoint_sha256"],
            "Evaluation/checkpoint binding differs")
    require(meta["settings"] == config and meta["precision"] == "FP32" and meta["image_count"] == 149,
            "Inference metadata differs")
    require(metrics["input"] == dict(gt_sha256=sha(root / ("gt_" + split + ".json")),
            predictions_sha256=sha(work / "predictions.json"),
            worker_sha256=protocol()["base_package_hashes"]["study_worker.py"],
            evaluator_runner_sha256=protocol()["base_package_hashes"]["run_internal_study.py"]),
            "COCO input/source binding differs")
    require(metrics["control_wrapper_manifest_sha256"] == expected_package and metrics["split"] == split
            and metrics["imgsz"] == 640 and metrics["independent_generalization_test"] is False, "COCO protocol differs")
    with np.load(work / "coco_precision_recall.npz", allow_pickle=False) as z:
        p, r = z["precision"], z["recall"]
        require(p.shape == (10, 101, 1, 4, 3) and r.shape == (10, 1, 4, 3), "COCO array shape differs")
        values = dict(AP50_95=mean_valid(p[:, :, :, 0, 2]), AP50=mean_valid(p[0, :, :, 0, 2]),
                      AP75=mean_valid(p[5, :, :, 0, 2]), AP_small=mean_valid(p[:, :, :, 1, 2]),
                      AP_medium=mean_valid(p[:, :, :, 2, 2]), AP_large=mean_valid(p[:, :, :, 3, 2]),
                      AR1=mean_valid(r[:, :, 0, 0]), AR10=mean_valid(r[:, :, 0, 1]), AR100=mean_valid(r[:, :, 0, 2]),
                      AR_small=mean_valid(r[:, :, 1, 2]), AR_medium=mean_valid(r[:, :, 2, 2]), AR_large=mean_valid(r[:, :, 3, 2]))
    for key, value in values.items():
        actual = metrics["metrics"][key]
        require((value is None and actual is None) or (value is not None and actual is not None and abs(value - actual) <= 1e-12),
                "Saved COCO metric does not match returned array: " + key)
    return values


def audit_training(root, item):
    job = item["job"]
    record, config, work = task_record(root, job)
    for key in ("architecture", "budget", "seed", "method", "epochs"):
        require(config[key] == item[key], "Training whitelist mismatch")
    view = root / "views" / job
    body = read(view / "view.json")
    ids = expected_ids(root, item["seed"])
    require(body["input"] == dict(ids=ids, label_text=None) and body["n_training_samples"] == len(ids)
            and sha(view / "view.json") == config["data_view_sha256"], "Manual repetition view differs")
    index = {row["id"]: row for row in read(root / "manifest.json")}
    for relative, digest in body["hashes"].items():
        require(sha(safe_artifact(view, relative)) == digest, "Training view file changed")
    for position, image_id in enumerate(ids):
        row = index[image_id]
        name = str(position).zfill(6) + "_" + row["staged_name"]
        require(body["hashes"]["images/train/" + name] == row["image_sha256"]
                and body["hashes"]["labels/train/" + Path(name).stem + ".txt"] == row["label_sha256"],
                "Training loader received something other than the 80 manual images/labels")
    settings = read(work / "training_settings.json")
    opts = settings["native_options"]
    require(settings["precision"] == "FP32" and settings["amp"] is False and settings["training_seed"] == item["seed"]
            and settings["hyp"]["warmup_epochs"] == 3 and opts["weights"] == ""
            and opts["epochs"] == item["epochs"] and opts["batch_size"] == opts["total_batch_size"] == 8
            and opts["img_size"] == [640, 640] and opts["workers"] == 4 and opts["adam"] is False
            and opts["multi_scale"] is False and opts["resume"] is False and opts["rect"] is False,
            "Native training protocol differs")
    seeds = read(work / "seed_applied.json")
    require(len(seeds) == 1 and seeds[0]["applied_training_seed"] == item["seed"], "Training seed hook failed")
    lines = (work / "train/results.txt").read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    require(len(lines) == item["epochs"] and all(line.split()[0] == "{}/{}".format(i, item["epochs"] - 1)
             for i, line in enumerate(lines)), "Epoch record incomplete")
    summary = read(work / "optimizer_step_summary.json")
    nb, nw, reference = schedule(len(ids), item["epochs"])
    require(summary["success"] and summary["hook_restored"] and summary["actual_successful_SGD_step_calls"] == len(reference)
            and summary["trace_sha256"] == sha(work / "optimizer_steps.jsonl")
            and summary["historical_SSL_counter_observed"] is False, "Update counter evidence failed")
    trace = [__import__("json").loads(line) for line in (work / "optimizer_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    require(len(trace) == len(reference), "Optimizer trace truncated")
    for number, (row, expected) in enumerate(zip(trace, reference), 1):
        c = row["context"]
        require(row["update"] == number and row["optimizer_step_completed"] is True
                and tuple(c[k] for k in ("epoch", "i", "ni", "accumulate")) == expected
                and c["nb"] == nb and c["nw"] == nw and c["nbs"] == 64 and c["total_batch_size"] == 8
                and c["actual_dataset_samples"] == len(ids), "Optimizer trace differs from pinned reference schedule")
    retention = read(work / "worker_retention_audit.json")
    require(retention["success"] and retention["hook_restored"] and retention["checkpoint_deletion_performed"] is False,
            "Save hook not restored or failed")
    weights, caches = work / "train/weights", {view / "labels/train.cache", view / "labels/val.cache"}
    saved, suppressed = [], []
    for event in retention["events"]:
        path = Path(event["path"])
        require(event["success"] is True and (path.parent == weights or path in caches), "Save outside weights/current view caches")
        if event["action"] == "suppress":
            require(path.parent == weights and re.fullmatch(r"(?:epoch|best)_\d{3}\.pt", path.name)
                    and event["checkpoint_epoch"] == event["epoch"] and not path.exists(), "Invalid numbered suppression")
            suppressed.append(event)
        else:
            require(event["action"] == "save" and event["free_bytes_before_save"] >= 6 * 1024 ** 3, "Real save below reserve/failed")
            if path.parent == weights:
                require(path.name in ("init.pt", "last.pt", "best.pt"), "Undeclared retained checkpoint")
                saved.append(event)
    require(sorted(e["checkpoint_epoch"] for e in saved if Path(e["path"]).name == "last.pt")
            == [-1] + list(range(item["epochs"])), "Missing per-epoch last or final stripping save")
    require(sum(Path(e["path"]).name == "init.pt" for e in saved) == 1
            and sum(Path(e["path"]).name == "best.pt" and e["checkpoint_epoch"] == -1 for e in saved) == 1,
            "Init/best final save missing")
    expected_numbered = [ep for ep in range(item["epochs"]) if ep == 0 or (ep + 1) % 25 == 0 or ep >= item["epochs"] - 5]
    require(sorted(e["epoch"] for e in suppressed if Path(e["path"]).name.startswith("epoch_")) == expected_numbered,
            "Numbered epoch interception incomplete")
    expected_best = sorted(e["checkpoint_epoch"] for e in saved if Path(e["path"]).name == "best.pt"
                           and e["checkpoint_epoch"] >= 200)
    require(sorted(e["epoch"] for e in suppressed if Path(e["path"]).name.startswith("best_")) == expected_best,
            "Numbered best interception incomplete")
    require({p.name for p in weights.iterdir()} == {"init.pt", "last.pt", "best.pt"}, "Unexpected weights retained")
    checkpoint = sha(weights / "last.pt")
    evaluation = {split: audit_eval(root, job, split, checkpoint, config["package_manifest_sha256"]) for split in ("val", "test")}
    result = dict(job=job, epochs=item["epochs"], seed=item["seed"], only_manual_images=80,
                  repeated_samples=len(ids), nb=nb, actual_optimizer_updates=len(reference),
                  reference_SSL_updates_reconstructed_not_retrospectively_observed=True,
                  last_sha256=checkpoint, evaluation=evaluation, passed=True)
    dump(root / (job + "_audit.json"), result)
    return result
