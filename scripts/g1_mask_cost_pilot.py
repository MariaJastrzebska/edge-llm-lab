#!/usr/bin/env python3
"""G1 pilot — the structured-generation energy tax on edge (see docs/research/STATUS.md).

Measures what nobody has measured: what grammar/schema-constrained decoding
actually costs in **milliseconds and Joules** on an Apple-Silicon edge SoC,
where the CPU-side mask construction cannot hide behind a long GPU forward pass
because CPU and GPU share one memory bus and the model is small.

Three conditions over the same 20 medical tool-call prompts:

  a. ``unconstrained``  — plain decode, no constraint
  b. ``gbnf``           — GBNF grammar generated from the medical JSON schema by
                          llama.cpp's own ``json_schema_to_grammar.py`` (the
                          grammar text is committed with the results)
  c. ``json_schema``    — the same schema handed to llama-server's native
                          ``json_schema`` parameter (its internal C++ converter)

(b) and (c) are not redundant: both derive from one schema, so any gap between
them isolates *converter* differences rather than the cost of constraining.

Per run we capture llama.cpp's own timings (prompt_ms / predicted_ms / per-token
ms), wall clock, energy integrated over the generation window via
:class:`edge_llm_lab.telemetry.PowermetricsRecorder`, and output validity
(parses as JSON? validates against the schema?) so Joules-per-VALID-output (G2)
is computable rather than inferred.

Measurement hygiene, because this is meant to survive review:
  * ``cache_prompt: false`` — otherwise prefill cost collapses after run 1 and
    prompt_ms is not comparable across conditions.
  * A warm-up request per condition is issued and discarded.
  * Condition order is rotated per (repeat, prompt) so thermal drift over a
    multi-hour run cannot alias onto one condition.
  * An idle power baseline is measured before and after, so energy can be
    reported gross AND baseline-subtracted (powermetrics reports whole-SoC
    power, not per-process).
  * Nothing is interpolated or back-filled. A failed request is written out as
    a failed row.

Usage:  python3 scripts/g1_mask_cost_pilot.py [--repeats 5] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "examples" / "desktop"))

LLAMA_DIR = REPO_ROOT / "examples" / "desktop" / "llama.cpp"
LLAMA_SERVER = LLAMA_DIR / "build" / "bin" / "llama-server"
SCHEMA_TO_GRAMMAR = LLAMA_DIR / "examples" / "json_schema_to_grammar.py"
SYSTEM_PROMPT_PATH = REPO_ROOT / "examples" / "desktop" / "prompts" / "constant_data_en.txt"
PROMPT_SET_PATH = REPO_ROOT / "experiments" / "prompts" / "g1-medical-toolcall-v1.json"

PORT = 8127
BASE_URL = f"http://127.0.0.1:{PORT}"
TOOL_NAME = "send_medical_data_en"

# Sampler settings — identical across conditions, recorded in the manifest.
# 768 gives a well-formed COT tool call room to finish (a complete one measured
# ~450-500 tokens); runs that still hit the cap are recorded as stop_type=limit
# rather than being retried, because truncation-under-constraint is itself a
# result.
N_PREDICT = 768
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 40
SEEDS = [1001, 1002, 1003, 1004, 1005]

CONDITIONS = ["unconstrained", "gbnf", "json_schema"]


# --------------------------------------------------------------------------
# schema + grammar
# --------------------------------------------------------------------------

def build_envelope_schema() -> dict:
    """The full tool-call envelope the system prompt actually demands.

    Constraining only the ``arguments`` object would be an easier grammar than
    the app really needs; the envelope (tool_calls array → name/arguments/id)
    is what constant_data_en.txt asks for, and its deeper rule stack is exactly
    the regime llama.cpp's GBNF docs warn about.
    """
    from pydantic_models.pydantic_models_en import ConstantDataAnalysisCOT

    args = ConstantDataAnalysisCOT.model_json_schema()
    defs = args.pop("$defs", {})
    return {
        "$defs": defs,
        "title": "MedicalToolCallEnvelope",
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_calls"],
        "properties": {
            "tool_calls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "arguments", "id"],
                    "properties": {
                        "name": {"const": TOOL_NAME},
                        "arguments": args,
                        "id": {"type": "string"},
                    },
                },
            }
        },
    }


def schema_to_gbnf(schema: dict, workdir: Path) -> str:
    """Convert via llama.cpp's own converter so the grammar is reproducible."""
    schema_file = workdir / "_schema_for_grammar.json"
    schema_file.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCHEMA_TO_GRAMMAR), str(schema_file)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"json_schema_to_grammar.py failed: {proc.stderr[:800]}")
    schema_file.unlink(missing_ok=True)
    return proc.stdout


# --------------------------------------------------------------------------
# model resolution
# --------------------------------------------------------------------------

def resolve_ollama_gguf(model: str = "granite3.1-dense", tag: str = "2b") -> dict:
    """Resolve the GGUF blob Ollama stores for a model, without copying it."""
    manifest = (
        Path.home() / ".ollama" / "models" / "manifests"
        / "registry.ollama.ai" / "library" / model / tag
    )
    if not manifest.exists():
        raise FileNotFoundError(f"Ollama manifest not found: {manifest}")
    data = json.loads(manifest.read_text())
    layer = next(
        l for l in data["layers"]
        if l["mediaType"] == "application/vnd.ollama.image.model"
    )
    digest = layer["digest"].replace(":", "-")
    blob = Path.home() / ".ollama" / "models" / "blobs" / digest
    if not blob.exists():
        raise FileNotFoundError(f"Ollama blob not found: {blob}")
    return {
        "model": f"{model}:{tag}",
        "manifest": str(manifest),
        "gguf_path": str(blob),
        "gguf_digest": layer["digest"],
        "gguf_size_bytes": layer["size"],
    }


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------

class LlamaServer:
    def __init__(self, gguf_path: str, log_path: Path, ctx: int = 8192, ngl: int = 99):
        self.gguf_path = gguf_path
        self.log_path = log_path
        self.ctx = ctx
        self.ngl = ngl
        self.proc: Optional[subprocess.Popen] = None
        self.args: list[str] = []

    def start(self, timeout_s: int = 300) -> None:
        import requests

        self.args = [
            str(LLAMA_SERVER),
            "-m", self.gguf_path,
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "-c", str(self.ctx),
            "-ngl", str(self.ngl),
            "--parallel", "1",
            "--no-webui",
        ]
        log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            self.args, stdout=log, stderr=subprocess.STDOUT, cwd=str(LLAMA_DIR)
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited rc={self.proc.returncode}; see {self.log_path}"
                )
            try:
                r = requests.get(f"{BASE_URL}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"llama-server not healthy in {timeout_s}s; see {self.log_path}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()


def apply_chat_template(system: str, user: str) -> str:
    """Render the model's own chat template server-side.

    Using /apply-template + /completion (rather than the OAI chat endpoint)
    keeps the prompt string byte-identical across all three conditions, so the
    only thing that varies is the constraint.
    """
    import requests

    r = requests.post(
        f"{BASE_URL}/apply-template",
        json={"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["prompt"]


# --------------------------------------------------------------------------
# validity
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_first_object(text: str) -> Optional[str]:
    """Balanced-brace scan for the first top-level JSON object."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def assess_validity(text: str, schema: dict) -> dict:
    """Strict and lenient JSON parsing plus schema validation.

    Both are reported. Lenient parsing (fence-stripping / object extraction) is
    a favour we do the unconstrained condition, and hiding it behind a single
    'valid' column would flatter it silently.
    """
    import jsonschema

    out = {
        "parses_json_strict": False,
        "parses_json_lenient": False,
        "schema_valid": False,
        "validation_error": None,
    }

    obj = None
    try:
        obj = json.loads(text)
        out["parses_json_strict"] = True
        out["parses_json_lenient"] = True
    except Exception:
        candidate = None
        m = _FENCE_RE.search(text)
        if m:
            candidate = m.group(1)
        if candidate is None:
            candidate = _extract_first_object(text)
        if candidate is not None:
            try:
                obj = json.loads(candidate)
                out["parses_json_lenient"] = True
            except Exception:
                obj = None

    if obj is not None:
        try:
            jsonschema.validate(instance=obj, schema=schema)
            out["schema_valid"] = True
        except Exception as e:
            out["validation_error"] = str(e).split("\n")[0][:300]
    return out


# --------------------------------------------------------------------------
# one measured request
# --------------------------------------------------------------------------

def run_one(prompt_text: str, condition: str, seed: int, grammar: str,
            schema: dict, recorder) -> dict:
    import requests

    payload: dict[str, Any] = {
        "prompt": prompt_text,
        "n_predict": N_PREDICT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": seed,
        "cache_prompt": False,
        "stream": False,
    }
    if condition == "gbnf":
        payload["grammar"] = grammar
    elif condition == "json_schema":
        payload["json_schema"] = schema

    rec: dict[str, Any] = {"condition": condition, "seed": seed}

    t0 = time.monotonic()
    try:
        r = requests.post(f"{BASE_URL}/completion", json=payload, timeout=900)
        r.raise_for_status()
        data = r.json()
        t1 = time.monotonic()
        rec["status"] = "ok"
    except Exception as e:
        t1 = time.monotonic()
        rec["status"] = "failed"
        rec["error"] = str(e)[:500]
        rec["wall_ms"] = round((t1 - t0) * 1000, 2)
        if recorder is not None:
            rec.update(recorder.energy_for_window(t0, t1).as_dict())
        return rec

    rec["wall_ms"] = round((t1 - t0) * 1000, 2)
    if recorder is not None:
        rec.update(recorder.energy_for_window(t0, t1).as_dict())

    timings = data.get("timings") or {}
    rec["prompt_ms"] = timings.get("prompt_ms")
    rec["prompt_n"] = timings.get("prompt_n")
    rec["prompt_per_token_ms"] = timings.get("prompt_per_token_ms")
    rec["predicted_ms"] = timings.get("predicted_ms")
    rec["predicted_n"] = timings.get("predicted_n")
    rec["predicted_per_token_ms"] = timings.get("predicted_per_token_ms")
    rec["predicted_per_second"] = timings.get("predicted_per_second")
    rec["stopped_limit"] = data.get("stopped_limit")
    rec["stopped_eos"] = data.get("stopped_eos")
    rec["stop_type"] = data.get("stop_type")

    content = data.get("content", "")
    rec["output_chars"] = len(content)
    rec["output"] = content
    rec.update(assess_validity(content, schema))
    return rec


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def sh(cmd: list[str], cwd: Path) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, text=True, cwd=cwd).strip()
    except Exception:
        return None


def main() -> int:
    global N_PREDICT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-predict", type=int, default=N_PREDICT)
    ap.add_argument("--limit-prompts", type=int, default=None,
                    help="use only the first N prompts (smoke testing)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--out", default=os.environ.get("RUN_DIR"))
    ap.add_argument("--settle-s", type=float, default=1.5,
                    help="idle gap between requests so energy windows don't bleed")
    args = ap.parse_args()

    N_PREDICT = args.n_predict
    conditions = [c for c in args.conditions.split(",") if c]

    out_dir = Path(args.out) if args.out else (
        REPO_ROOT / "experiments" / "results"
        / f"{_dt.datetime.now():%Y%m%d-%H%M%S}-g1-mask-cost-pilot"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[g1] output → {out_dir}")

    if not LLAMA_SERVER.exists():
        print(f"[g1] FATAL: llama-server not built at {LLAMA_SERVER}", file=sys.stderr)
        return 2

    # ---- assets -------------------------------------------------------
    prompt_set = json.loads(PROMPT_SET_PATH.read_text(encoding="utf-8"))
    prompts = prompt_set["prompts"]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    schema = build_envelope_schema()
    grammar = schema_to_gbnf(schema, out_dir)
    (out_dir / "envelope_schema.json").write_text(
        json.dumps(schema, indent=2), encoding="utf-8")
    (out_dir / "envelope.gbnf").write_text(grammar, encoding="utf-8")
    print(f"[g1] grammar: {len(grammar.splitlines())} rules")

    model_info = resolve_ollama_gguf()
    print(f"[g1] model: {model_info['gguf_path']}")

    # ---- manifest -----------------------------------------------------
    manifest = {
        "experiment": "g1-mask-cost-pilot",
        "hypothesis": "G1 — structured-generation energy tax on edge (docs/research/STATUS.md)",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "repo_git_sha": sh(["git", "rev-parse", "HEAD"], REPO_ROOT),
        "repo_git_dirty": bool(sh(["git", "status", "--porcelain"], REPO_ROOT)),
        "llama_cpp_git_sha": sh(["git", "rev-parse", "HEAD"], LLAMA_DIR),
        "llama_server_version": sh([str(LLAMA_SERVER), "--version"], LLAMA_DIR),
        "model": model_info,
        "prompt_set_id": prompt_set["prompt_set_id"],
        "prompt_set_path": str(PROMPT_SET_PATH.relative_to(REPO_ROOT)),
        "system_prompt_path": str(SYSTEM_PROMPT_PATH.relative_to(REPO_ROOT)),
        "n_prompts": len(prompts),
        "conditions": conditions,
        "repeats": args.repeats,
        "seeds": SEEDS[: args.repeats],
        "sampler": {
            "n_predict": N_PREDICT, "temperature": TEMPERATURE,
            "top_p": TOP_P, "top_k": TOP_K, "cache_prompt": False,
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "energy_backend": "PowermetricsRecorder(cpu_power, 100ms, integrated per window)",
        "energy_caveat": (
            "powermetrics reports whole-SoC package power, not per-process. "
            "Values are gross; baseline_idle_mw is recorded for subtraction."
        ),
    }

    # ---- instruments --------------------------------------------------
    from edge_llm_lab.telemetry import PowermetricsRecorder

    recorder = PowermetricsRecorder(interval_ms=100)
    if not recorder.start():
        print(f"[g1] WARNING: energy telemetry unavailable — {recorder.start_error}")
        manifest["energy_status"] = f"unavailable: {recorder.start_error}"
        recorder = None
    else:
        manifest["energy_status"] = "ok"

    server = LlamaServer(model_info["gguf_path"], out_dir / "llama-server.log")
    records: list[dict] = []
    try:
        print("[g1] starting llama-server …")
        server.start()
        manifest["llama_server_args"] = server.args
        print("[g1] server healthy")

        if recorder is not None:
            manifest["baseline_idle_mw_before"] = recorder.measure_baseline(8.0)
            print(f"[g1] idle baseline: {manifest['baseline_idle_mw_before']:.0f} mW")

        rendered = {p["id"]: apply_chat_template(system_prompt, p["user"]) for p in prompts}
        manifest["rendered_prompt_chars"] = {k: len(v) for k, v in rendered.items()}

        # warm-up per condition (discarded)
        print("[g1] warm-up …")
        for cond in conditions:
            run_one(rendered[prompts[0]["id"]], cond, SEEDS[0], grammar, schema, None)

        total = args.repeats * len(prompts) * len(conditions)
        done = 0
        t_start = time.monotonic()
        for repeat in range(1, args.repeats + 1):
            seed = SEEDS[(repeat - 1) % len(SEEDS)]
            for pi, p in enumerate(prompts):
                # rotate condition order so thermal drift cannot alias onto one condition
                shift = (repeat + pi) % len(conditions)
                order = conditions[shift:] + conditions[:shift]
                for cond in order:
                    time.sleep(args.settle_s)
                    rec = run_one(rendered[p["id"]], cond, seed, grammar, schema, recorder)
                    rec.update({
                        "run_id": f"r{repeat:02d}-{p['id']}-{cond}",
                        "repeat": repeat,
                        "prompt_id": p["id"],
                        "prompt_category": p.get("category"),
                    })
                    records.append(rec)
                    done += 1
                    elapsed = time.monotonic() - t_start
                    eta = (elapsed / done) * (total - done)
                    print(
                        f"[g1] {done}/{total} {rec['run_id']}: {rec['status']} "
                        f"pred={rec.get('predicted_ms')}ms n={rec.get('predicted_n')} "
                        f"valid={rec.get('schema_valid')} J={rec.get('combined_j')} "
                        f"eta={eta/60:.0f}m",
                        flush=True,
                    )
                    # incremental write — a crash at hour 3 must not lose hours 1-2
                    (out_dir / "raw.json").write_text(
                        json.dumps({"manifest": manifest, "runs": records},
                                   indent=2, default=str),
                        encoding="utf-8",
                    )
        manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        print("[g1] interrupted — writing what we have")
    except Exception as e:
        manifest["status"] = "error"
        manifest["error"] = f"{type(e).__name__}: {e}"
        print(f"[g1] ERROR: {e}", file=sys.stderr)
    finally:
        if recorder is not None:
            try:
                manifest["baseline_idle_mw_after"] = recorder.measure_baseline(8.0)
            except Exception:
                pass
        server.stop()
        if recorder is not None:
            manifest["energy_samples_collected"] = len(recorder.samples)
            recorder.stop()

    # ---- outputs ------------------------------------------------------
    manifest["n_runs"] = len(records)
    (out_dir / "raw.json").write_text(
        json.dumps({"manifest": manifest, "runs": records}, indent=2, default=str),
        encoding="utf-8",
    )
    write_tidy_csv(out_dir / "tidy.csv", records, manifest)
    print(f"[g1] wrote {len(records)} runs → {out_dir}")
    return 0 if manifest.get("status") == "completed" else 1


TIDY_COLUMNS = [
    "run_id", "condition", "repeat", "seed", "prompt_id", "prompt_category",
    "status", "prompt_ms", "prompt_n", "prompt_per_token_ms",
    "predicted_ms", "predicted_n", "predicted_per_token_ms", "predicted_per_second",
    "wall_ms", "stop_type", "stopped_limit", "stopped_eos", "output_chars",
    "parses_json_strict", "parses_json_lenient", "schema_valid", "validation_error",
    "cpu_j", "gpu_j", "ane_j", "combined_j",
    "mean_cpu_mw", "mean_gpu_mw", "mean_combined_mw",
    "energy_window_s", "energy_n_samples", "energy_coverage",
    "baseline_idle_mw_before", "combined_j_baseline_subtracted",
    "error",
]


def write_tidy_csv(path: Path, records: list[dict], manifest: dict) -> None:
    """One row per run, long format — the shape analysis expects (RUNBOOK)."""
    baseline = manifest.get("baseline_idle_mw_before")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TIDY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            row = dict(r)
            row["baseline_idle_mw_before"] = baseline
            # Baseline subtraction only where every input is real — never guessed.
            if (baseline is not None and r.get("combined_j") is not None
                    and r.get("energy_window_s")):
                row["combined_j_baseline_subtracted"] = round(
                    r["combined_j"] - (baseline / 1000.0) * r["energy_window_s"], 5
                )
            w.writerow(row)


if __name__ == "__main__":
    sys.exit(main())
