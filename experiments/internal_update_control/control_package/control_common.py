"""Fixed compact/80 control protocol and deterministic update schedule."""
import hashlib
import json
import math
from pathlib import Path
from resource_base import read, dump, sha, regular, require, load

PACKAGE = Path(__file__).resolve().parent
SEEDS = (101, 202, 303)
STUDY_SHA = "2e5b071a355e57e7b36bc4b17268c245eddb31b90dcc41b013450bea5aabfcd0"
DATA_SHA = "25ad1cac4bc842090b4044317654d3ac136f6d6ddc50c7b6ec61534394e7dc4d"
TRAIN_SHA = "7078fcbc2085a75991282de2701a33d9796be1b5a1af1366139f069507c69f5e"


def protocol():
    return read(PACKAGE / "protocol.json")


def verify_package(expected):
    manifest = PACKAGE / "delivery_manifest.json"
    require(sha(regular(manifest)) == expected, "Package manifest SHA differs")
    require((PACKAGE / "delivery_manifest.sha256").read_text("ascii").split()[0] == expected,
            "Package sidecar differs")
    rows = read(manifest)["files"]
    names = [row["path"] for row in rows]
    require(len(names) == len(set(names)), "Duplicate package payload")
    require({p.name for p in PACKAGE.iterdir()} == set(names) | {"delivery_manifest.json", "delivery_manifest.sha256"},
            "Unlisted package item/module refused")
    for row in rows:
        path = safe_artifact(PACKAGE, row["path"])
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Package payload changed")


def safe_artifact(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and "\\" not in relative, "Unsafe relative path")
    return regular(root / rel)


def disjoint(a, b):
    a, b = Path(a).absolute(), Path(b).absolute()
    require(a != b and a not in b.parents and b not in a.parents, "Protected paths overlap")


def bank_round(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(2 * remainder > denominator or (2 * remainder == denominator and quotient % 2 == 1))


def schedule(samples, epochs):
    """Reconstruct pinned train.py's *scheduled* SGD calls, not observed old calls."""
    require(type(samples) is int and type(epochs) is int and samples > 0 and epochs > 0, "Positive integer samples/epochs required")
    nb = (samples + 7) // 8
    nw = max(3 * nb, 1000)
    rows = []
    for ni in range(nb * epochs):
        accumulate = max(1, bank_round(nw + 7 * ni, nw)) if ni <= nw else 8
        if ni % accumulate == 0:
            rows.append((ni // nb, ni % nb, ni, accumulate))
    return nb, nw, rows


def plan(phase):
    require(phase in ("smoke", "formal"), "Unknown train phase")
    seeds, epochs = ((101,), 2) if phase == "smoke" else (SEEDS, 300)
    return [dict(architecture="compact", budget=80, seed=s, method="stepmatched", epochs=epochs,
                 job=("smoke_" if phase == "smoke" else "") + "compact_b80_s{}_stepmatched".format(s))
            for s in seeds]


def expected_ids(root, seed):
    ids = read(root / "label_budgets.json")["80"]["labeled_ids"]
    require(len(ids) == len(set(ids)) == 80, "Manual label subset changed")
    target = protocol()["targets"][str(seed)]["samples"]
    return (ids * ((target + 79) // 80))[:target]


def verify_formal_release(root, expected):
    paths = [root / "jobs" / item["job"] / "completed.json" for item in plan("smoke")]
    for item in plan("smoke"):
        paths.extend(root / "jobs" / (item["job"] + "_eval_" + split + "_640") / "completed.json"
                     for split in ("val", "test"))
    wanted = dict(schema="compact80_stepmatched_formal_release_v1", study=str(root), gpu="0",
                  seeds=[101, 202, 303], package_sha256=expected,
                  accepted_smoke_completed_sha256={p.relative_to(root).as_posix(): sha(regular(p)) for p in paths})
    require(read(regular(root / "formal_release.json")) == wanted, "Formal release missing or does not bind this actual smoke")


def validate_config(config, path, expected):
    root, work = [Path(config[k]).absolute() for k in ("study", "work")]
    require(root.resolve() == root and work.resolve() == work and root / "jobs" in work.parents,
            "Worker path escaped its private study")
    require(path == work / "worker_config.json" and work.name.startswith("attempt_"), "Config/work differs")
    require(config["package_manifest_sha256"] == expected and config["control_protocol_sha256"] == sha(PACKAGE / "protocol.json"),
            "Control identity differs")
    require(config["device"] == "0" and config["physical_gpu"] == "0", "GPU0 only")
    require(config["backend"] == "v7", "Only frozen v7 backend allowed")
    require(config["worker_sha256"] == sha(PACKAGE / "control_worker.py")
            and config["runner_sha256"] == sha(PACKAGE / "run_control.py")
            and config["legacy_worker_sha256"] == protocol()["base_package_hashes"]["study_worker.py"]
            and config["dataset_manifest_sha256"] == DATA_SHA and config["study_protocol_sha256"] == STUDY_SHA,
            "Worker/runner/data source bindings differ")
    binding = config["gpu_binding"]
    require(binding["index"] == "0" and binding["uuid"].startswith("GPU-")
            and "V100" in binding["name"] and binding["pci_bus_id"], "Physical GPU binding differs")
    require(sha(root / "study.json") == STUDY_SHA and sha(root / "manifest.json") == DATA_SHA, "Frozen identity differs")
    from prepare_layout import verify_prepared_ownership
    meta = verify_prepared_ownership(root, expected)
    require(meta["study"] == str(root) and meta["package_manifest_sha256"] == expected, "Prepared ownership differs")
    for rel, digest in protocol()["base_package_hashes"].items():
        require(sha(safe_artifact(root / "run_package", rel)) == digest, "Legacy package changed")
    require(sha(regular(root / "source/train.py")) == TRAIN_SHA, "Native train source changed")
    allowed = {item["job"]: item for phase in ("smoke", "formal") for item in plan(phase)}
    job = work.parent.name
    if config["mode"] == "train":
        require(job in allowed, "Unlisted training job")
        for key in ("architecture", "budget", "seed", "method", "epochs"):
            require(config[key] == allowed[job][key], "Train setting differs: " + key)
        require(config["batch"] == 8 and config["workers"] == 4, "Batch/workers differ")
        require(config["n_training_samples"] == len(expected_ids(root, config["seed"])), "Sample target differs")
        require(Path(config["data"]) == root / "views" / job / "data.yaml", "Training data escaped view")
        view_root = Path(config["data"]).parent
        view = read(regular(view_root / "view.json"))
        ids = expected_ids(root, config["seed"])
        require(sha(view_root / "view.json") == config["data_view_sha256"]
                and view["input"] == dict(ids=ids, label_text=None) and view["n_training_samples"] == len(ids),
                "Pretraining manual-only view identity differs")
        for rel, digest in view["hashes"].items():
            require(sha(safe_artifact(view_root, rel)) == digest, "Pretraining view bytes differ")
        index = {r["id"]: r for r in read(root / "manifest.json")}
        for position, image_id in enumerate(ids):
            row = index[image_id]
            name = str(position).zfill(6) + "_" + row["staged_name"]
            require(view["hashes"]["images/train/" + name] == row["image_sha256"]
                    and view["hashes"]["labels/train/" + Path(name).stem + ".txt"] == row["label_sha256"],
                    "Pretraining view is not the fixed 80 manual images/GT")
    else:
        require(config["mode"] == "infer" and config["backend"] == "v7", "V7 inference only")
        parts = job.rsplit("_eval_", 1)
        require(len(parts) == 2 and parts[0] in allowed and parts[1] in ("val_640", "test_640"), "Unlisted evaluation")
        split = parts[1].split("_")[0]
        ids = [r["id"] for r in read(root / "manifest.json") if r["split"] == split]
        require(config["ids"] == ids and config["imgsz"] == 640 and config["nms_iou"] == .65, "Evaluation protocol differs")
        prior = read(root / "jobs" / parts[0] / "completed.json")
        require(Path(config["weight"]) == Path(prior["work"]) / "train/weights/last.pt"
                and config["weight_sha256"] == sha(regular(Path(config["weight"]))), "Evaluation last weight differs")
    return root, work


class StepTrace:
    def __init__(self, config, work):
        self.config, self.work = config, work
        self.nb, self.nw, self.expected = schedule(config["n_training_samples"], config["epochs"])
        self.count, self.restored = 0, False
        self.file = (work / "optimizer_steps.jsonl").open("x", encoding="utf-8")

    def check(self, context):
        require(self.count < len(self.expected), "Unexpected extra optimizer update")
        actual = tuple(int(context[k]) for k in ("epoch", "i", "ni", "accumulate"))
        require(actual == self.expected[self.count], "Actual optimizer update sequence differs")
        require(int(context["nb"]) == self.nb and int(context["nw"]) == self.nw
                and int(context["nbs"]) == 64 and int(context["total_batch_size"]) == 8,
                "Actual accumulation/loader configuration differs")
        require(context["actual_dataset_samples"] == self.config["n_training_samples"], "Loader discarded/added training samples")

    def completed(self, context, lrs):
        self.check(context)
        require(all(math.isfinite(x) for x in lrs), "Nonfinite learning rate")
        self.file.write(json.dumps(dict(update=self.count + 1, context=context, learning_rates=lrs,
                                       optimizer_step_completed=True), allow_nan=False) + "\n")
        self.file.flush()
        self.count += 1

    def close(self, success):
        self.file.close()
        dump(self.work / "optimizer_step_summary.json", dict(actual_successful_SGD_step_calls=self.count,
            expected_calls_from_pinned_source=len(self.expected), nb=self.nb, nw=self.nw,
            epochs=self.config["epochs"], samples=self.config["n_training_samples"], hook_restored=self.restored,
            success=success and self.count == len(self.expected), trace_sha256=sha(self.work / "optimizer_steps.jsonl"),
            historical_SSL_counter_observed=False, reference="Pinned source schedule reconstruction; old stripped last.pt has updates=None"))
        if success:
            require(self.count == len(self.expected), "Optimizer update count incomplete")
