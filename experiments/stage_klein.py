#!/usr/bin/env python3
"""Stage the klein backend of the release model service on 127.0.0.1:20017 and drive
it over HTTP like the worker: text-to-image comparison prompts, the 24 edit/reference quality cases (payloads
from build_stage_cases.py), FHD, a determinism repeat and contract checks. Same card hand-off contract as
experiments/z-image-turbo/run_device_check.py. Writes stage/result.json and one PNG per case."""
import base64
import fcntl
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image, ImageChops, ImageStat

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
HOME = Path.home()
NAME = "klein-stage"
RELEASE = json.loads((PROJECT / "deploy/model-release-klein.json").read_text())
BASE = "http://127.0.0.1:20017"
OUT = ROOT / "stage"
COPY_THRESHOLD = 6.0  # the reference-copy threshold the caller applies
result = {"status": "starting", "cases": [], "baseline_restored": False}


def save():
    (OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


def run(*args, check=True, timeout=1000):
    return subprocess.run(args, check=check, text=True, capture_output=True, cwd=PROJECT, timeout=timeout)


def http(method, path, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_ready(url, bound=900):
    deadline = time.monotonic() + bound
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                body = json.loads(r.read())
            if body.get("status") == "ok":
                return body
            if body.get("status") == "error":
                raise RuntimeError(str(body))
        except OSError:
            pass
        time.sleep(3)
    raise TimeoutError("Model readiness exceeded bound")


def copy_score(output, canvas, padding):  # same as app.references.reference_copy_score
    def mad(a, b):
        a = a.convert("RGB").resize((128, 128), Image.BILINEAR)
        b = b.convert("RGB").resize((128, 128), Image.BILINEAR)
        return sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
    w, h = canvas.size
    content = canvas.crop((padding["left"], padding["top"], w - padding["right"], h - padding["bottom"]))
    return min(mad(output, canvas), mad(output, content))


def interrupted(signum, frame):
    raise InterruptedError(f"Stage interrupted by signal {signum}")


def generate(case, name=None):
    body = {"prompt": case["effective_prompt"], "seed": case["seed"], "width": case["width"],
            "height": case["height"], "num_steps": 4, "return_rgba": False}
    if case.get("image_b64"):
        body["images"] = [case["image_b64"]]
    started = time.monotonic()
    code, resp = http("POST", "/predict", body)
    if code != 200:
        raise RuntimeError(f"{case['id']}: HTTP {code} {resp}")
    raw = base64.b64decode(resp.pop("image"), validate=True)
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        assert im.size == (case["width"], case["height"]), im.size
        std = max(ImageStat.Stat(im.convert("RGB")).stddev)
        digest = hashlib.sha256(im.tobytes()).hexdigest()
        rec = {"name": name or case["id"], "task_mode": case["task_mode"], "width": case["width"],
               "height": case["height"], "wall_s": time.monotonic() - started, "stddev": std, "pixels_sha256": digest,
               "timing_ms": resp.get("timing_ms"), "reference_count": resp.get("reference_count"),
               "generated_size": resp.get("generated_size")}
        if case["task_mode"] == "reference":
            with Image.open(OUT / case["canvas"]) as canvas:
                rec["copy_score"] = round(copy_score(im, canvas, case["padding"]), 2)
                rec["would_be_rejected"] = rec["copy_score"] < COPY_THRESHOLD
    (OUT / f"{rec['name']}.png").write_bytes(raw)
    assert std > 3, "blank output"
    return rec


def main():
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, interrupted)
    cases = json.loads((OUT / "cases.json").read_text())
    lock = (PROJECT / ".runtime/control.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = {svc: bool(run("docker", "compose", "ps", "--status", "running", "-q", svc).stdout.strip())
              for svc in ("model", "worker")}
    result["before"] = before
    save()
    try:
        if run("docker", "image", "inspect", RELEASE["tag"], "--format", "{{.Id}}").stdout.strip() != RELEASE["image_id"]:
            raise RuntimeError("Release tag does not point to the pinned image")
        run("docker", "compose", "stop", "-t", "900", "worker")
        run("docker", "compose", "stop", "-t", "60", "model")
        ids = run("docker", "ps", "-q").stdout.split()
        if ids:
            for item in json.loads(run("docker", "inspect", *ids).stdout):
                host = item["HostConfig"]
                if host.get("Privileged") or any("tenstorrent" in json.dumps(x) for x in (host.get("Devices") or [])):
                    raise RuntimeError("Another accelerator workload is active: " + item["Name"])
        run("docker", "rm", "-f", NAME, check=False)
        cache = PROJECT / "data/model-cache"
        cmd = ["docker", "create", "--name", NAME, "--ipc", "host", "--device", "/dev/tenstorrent",
               "-p", "127.0.0.1:20017:20000", "--log-opt", "max-size=30m", "--log-opt", "max-file=3"]
        for src, dst, ro in [(HOME / ".cache/huggingface", "/hf", True), (cache, "/cache", False),
                             (Path("/dev/hugepages-1G"), "/dev/hugepages-1G", False), (PROJECT / "deploy", "/service", True)]:
            cmd += ["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")]
        env = {"HF_HOME": "/hf", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TT_METAL_VISIBLE_DEVICES": "0",
               "MESH_DEVICE": "P100", "PYTHONUNBUFFERED": "1", "TT_METAL_OPERATION_TIMEOUT_SECONDS": "90",
               "PHOTO_TT_BACKEND": "klein", "PHOTO_MAX_DIMENSION": "1920", "PHOTO_DIMENSION_STEP": "8",
               "PHOTO_MAX_PIXELS": "2088960", "PHOTO_MODEL_DEFAULT_SIZE": "1024",
               "KLEIN_SNAPSHOT": "/hf/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/" + RELEASE["weights_revision"]}
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        cmd += [RELEASE["tag"], "python", "-m", "uvicorn", "tt_service:app", "--app-dir", "/service",
                "--host", "0.0.0.0", "--port", "20000", "--workers", "1", "--lifespan", "on"]
        run(*cmd)
        t0 = time.monotonic()
        run("docker", "start", NAME)
        result["initial_health"] = wait_ready(BASE)
        result["startup_wall_s"] = time.monotonic() - t0
        result["info"] = http("GET", "/info")[1]
        save()
        for case in cases:
            rec = generate(case)
            result["cases"].append(rec)
            save()
            print("PASS", rec["name"], round(rec["wall_s"], 2), rec.get("copy_score", ""), flush=True)
        first = cases[0]
        again = generate(first, first["id"] + "-repeat")
        again["deterministic_repeat"] = again["pixels_sha256"] == result["cases"][0]["pixels_sha256"]
        result["cases"].append(again)
        for name, w, h in (("t2i_fhd", 1920, 1080), ("t2i_fhd_warm", 1920, 1080), ("t2i_fhd_portrait", 1080, 1920)):
            result["cases"].append(generate({**first, "width": w, "height": h, "id": name}))
            save()
        img = next(c for c in cases if c.get("image_b64"))
        code, _ = http("POST", "/predict", {"prompt": "x", "seed": 1, "width": 512, "height": 512, "num_steps": 4,
                                            "images": [img["image_b64"], img["image_b64"]]})
        result["two_references_status"] = code
        code, _ = http("POST", "/predict", {"prompt": "x", "seed": 1, "width": 512, "height": 512, "num_steps": 9})
        result["wrong_steps_status"] = code
        result["health_after_rejections"] = http("GET", "/health")[1].get("status")
        assert result["two_references_status"] == 422 and result["wrong_steps_status"] == 422
        assert result["health_after_rejections"] == "ok" and again["deterministic_repeat"]
        result["status"] = "pass"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        run("docker", "stop", "-t", "60", NAME, check=False, timeout=100)
        logs = run("docker", "logs", NAME, check=False)
        (OUT / "model.log").write_text(logs.stdout + logs.stderr)
        run("docker", "rm", "-f", NAME, check=False)
        try:
            if before["model"]:
                run("docker", "compose", "start", "model")
                result["baseline_health"] = wait_ready("http://127.0.0.1:20014")
            if before["worker"]:
                run("docker", "compose", "start", "worker")
            result["baseline_restored"] = True
        finally:
            save()
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
