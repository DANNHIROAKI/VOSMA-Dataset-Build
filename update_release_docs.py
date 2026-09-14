#!/usr/bin/env python3
"""Update v0.2.0 public READMEs only after the completed DocLayNet audits.

Run from the real-datasets workspace containing builder/ and data/, or pass
--root. This script changes documentation only; it never invents dataset counts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


RELEASE = "v0.2.0"
EXPECTED_PAGES = 4999
DATASETS = ("doclaynet", "mot20", "coco_sama")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def count(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid completed count: {label}")
    return value


def completed_reports(root):
    reports = {}
    for name in DATASETS:
        directory = root / "data" / name
        info, validation = read_json(directory / "dataset.json"), read_json(directory / "validation.json")
        if validation.get("passed") is not True:
            raise ValueError(f"Dataset validation has not passed: {name}")
        for key in ("r1_count", "r2_count", "group_count"):
            count(info[key], f"{name}.{key}")
        for key in ("within_group_overlap_count", "groups_with_positive_mass", "groups_with_zero_mass"):
            count(validation[key], f"{name}.{key}")
        if validation["groups_with_positive_mass"] + validation["groups_with_zero_mass"] != info["group_count"]:
            raise ValueError(f"Inconsistent validated group counts: {name}")
        reports[name] = (info, validation)

    info, validation = reports["doclaynet"]
    audit = read_json(root / "data" / "doclaynet" / "inference-audit.json")
    if audit.get("passed") is not True:
        raise ValueError("The independent DocLayNet inference audit has not passed")
    relation_audit = read_json(root / "data" / "doclaynet" / "independent-audit.json")
    if relation_audit.get("passed") is not True:
        raise ValueError("The independent full DocLayNet relation audit has not passed")
    if info.get("splits") != ["test"] or info["group_count"] != EXPECTED_PAGES:
        raise ValueError("This release requires every official DocLayNet test page")
    pages, inference = info["page_counts"], info["inference_counts"]
    if (pages.get("selected") != EXPECTED_PAGES
            or pages.get("successful_inference") != EXPECTED_PAGES
            or pages.get("missing") != 0 or pages.get("failed") != 0
            or inference.get("successful_pages") != EXPECTED_PAGES):
        raise ValueError("The selected DocLayNet inference run is incomplete")
    for key in ("selected_pages", "successful_pages", "verified_images"):
        if audit.get(key) != EXPECTED_PAGES:
            raise ValueError(f"The independent image/page audit is incomplete: {key}")
    statistics = info["filter_statistics"]
    for side in ("r1", "r2"):
        raw = count(statistics[f"{side}.raw"], f"{side}.raw")
        exclusions = sum(count(value, key) for key, value in statistics.items()
                         if key.startswith(f"{side}.excluded."))
        if raw != info[f"{side}_count"] + exclusions:
            raise ValueError(f"DocLayNet source accounting does not balance: {side}")
    if audit.get("prediction_count") != statistics["r2.raw"]:
        raise ValueError("Audited prediction count differs from the adapter input")
    if inference.get("exported_predictions") != statistics["r2.raw"]:
        raise ValueError("Exported prediction accounting does not balance")
    excluded = sum(count(inference[key], key) for key in
                   ("threshold_excluded", "background_excluded", "nonfinite_score_excluded"))
    if (inference.get("top_k_candidates") != EXPECTED_PAGES * 100
            or inference["top_k_candidates"] != excluded + inference["exported_predictions"]):
        raise ValueError("Fixed top-100 inference accounting does not balance")
    provenance = info["prediction_provenance"]
    for report_key, info_key in (("pages", "pages_manifest"), ("predictions", "predictions"), ("protocol", "protocol")):
        if audit["input_sha256"][report_key] != provenance[info_key]["sha256"]:
            raise ValueError(f"Inference audit is for different frozen inputs: {report_key}")
    protocol = info["model_protocol"]
    inference_protocol = protocol["inference"]
    if (inference_protocol.get("device") != "cpu" or inference_protocol.get("dtype") != "float32"
            or inference_protocol.get("threads_per_worker") != 4):
        raise ValueError("The completed run differs from the release CPU float32 protocol")
    model = protocol["model"]
    if (model.get("model_id") != "Aryn/deformable-detr-DocLayNet"
            or model.get("license") != "Apache-2.0"
            or model.get("training_overlap_status") != "unknown"):
        raise ValueError("Unexpected source model identity, license, or training-overlap statement")
    if not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", "")):
        raise ValueError("The model is not pinned to a full immutable revision")
    return reports, audit


def section(text, title):
    match = re.search(r"^## " + re.escape(title) + r"\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if not match:
        raise ValueError(f"Required README section is missing: {title}")
    return match


def replace_section(text, title, body):
    match = section(text, title)
    return text[:match.start()] + f"## {title}\n\n{body.strip()}\n\n" + text[match.end():]


def protected_contracts(text):
    body = section(text, "Source selection and grouping").group(1)
    result = {}
    for name in ("MOT20", "COCO/Sama-COCO"):
        match = re.search(r"\*\*" + re.escape(name) + r"\.\*\*.*?(?=\n\n|\Z)", body, re.S)
        if not match:
            raise ValueError(f"Missing protected source contract: {name}")
        result[name] = match.group(0)
    return result


def available_table(reports):
    roles = {
        "doclaynet": ("Official test-page ground truth", "Fixed Aryn model predictions"),
        "mot20": ("Valid pedestrian ground truth", "Official detections"),
        "coco_sama": ("COCO 2017 non-crowd instances", "Sama-COCO non-crowd instances"),
    }
    lines = ["| Dataset directory | R1 | R2 | Retained R1 records | Retained R2 records | Groups |",
             "|---|---|---|---:|---:|---:|"]
    for name in DATASETS:
        info = reports[name][0]
        lines.append(f"| `{name}` | {roles[name][0]} | {roles[name][1]} | {info['r1_count']:,} | {info['r2_count']:,} | {info['group_count']:,} |")
    return "\n".join(lines)


def overlap_table(reports):
    lines = ["| Completed dataset | Positive-area record pairs | Groups with positive mass | Groups with zero mass |",
             "|---|---:|---:|---:|"]
    for name, label in (("doclaynet", "DocLayNet"), ("mot20", "MOT20"), ("coco_sama", "COCO/Sama-COCO")):
        validation = reports[name][1]
        lines.append(f"| {label} | {validation['within_group_overlap_count']:,} | {validation['groups_with_positive_mass']:,} | {validation['groups_with_zero_mass']:,} |")
    return "\n".join(lines)


def doc_contract(info):
    model = info["model_protocol"]["model"]
    model_url = f"https://huggingface.co/{model['model_id']}/tree/{model['revision']}"
    return f"""**DocLayNet.** R1 contains all original precedence-0 ground-truth boxes on all {info['group_count']:,} official test pages from the [DocLayNet 1.0.0 Core archive](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip), using `COCO/test.json`. R2 contains the exported predictions of the fixed [Aryn/deformable-detr-DocLayNet model]({model_url}). A group is `(split, source_image_id)`. The physical key `(doc_category, collection, doc_name, page_no)` is unique across selected pages; official PNG filename, dimensions, source identity, and per-image bytes are checked. All selected pages remain groups, including successful pages with zero predictions. Missing, failed, or duplicate inference pages stop the build. All classes remain in their page groups, so cross-class intersections are included. Predictions are not matched to GT or selected using GT.

The released model run uses CPU float32, batch size 1, and four PyTorch threads per worker. It began with eight workers and resumed with sixteen workers under the same frozen numerical protocol; completed page results were reused only after protocol and image-hash checks. The reproduction command uses sixteen workers. The fixed processor produces `[1,3,800,800]` model inputs and restores predicted `xyxy` boxes to each original PNG's dimensions. Native postprocessing takes the top 100 scores over the flattened 200-query × 12-class sigmoid output. The mutually exclusive exclusions are nonfinite scores, finite scores at most 0.7, and remaining class-0 (`N/A`) candidates, in that order. All remaining records are exported. There is no NMS, clipping, deduplication, GT-based threshold tuning, or class-pair filtering. Original top-100 rank, query index, label, score, and PNG-coordinate endpoints are retained in metadata. Geometry exclusions are applied subsequently by the common dataset conversion.

The checkpoint revision is `{model['revision']}` and its model-card license is Apache-2.0. The model's training overlap with these DocLayNet test pages is **unknown**. These relations measure spatial-join sampling on GT and frozen predictions; they do not establish an unseen-test-set detection result. Model files, page selection, protocol, predictions, and image hashes are recorded in provenance; `doclaynet/inference-audit.json` reports the independent all-page PNG, provenance, and prediction-accounting audit. See `doclaynet/README.md` for the completed counts and hash links."""


def reproduce_section():
    return """From the builder repository root, with Python 3.12 and a C++17 compiler available:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-inference.txt
.venv/bin/python fetch_sources.py --raw-root raw --sources doclaynet mot20 coco2017 sama_train sama_val
.venv/bin/python fetch_doclaynet_model.py --output-root runtime/doclaynet
.venv/bin/python fetch_doclaynet_images.py --raw-root raw --output-root runtime/doclaynet --splits test --workers 16
.venv/bin/python infer_doclaynet.py --pages runtime/doclaynet/pages.json --images runtime/doclaynet/images --model runtime/doclaynet/model --model-manifest runtime/doclaynet/model-manifest.json --output-root runtime/doclaynet/predictions --workers 16 --threads 4
.venv/bin/python build.py --raw-root raw --output-root data --datasets doclaynet mot20 coco_sama --doc-pages runtime/doclaynet/pages.json --doc-predictions runtime/doclaynet/predictions/predictions.jsonl.gz --doc-protocol runtime/doclaynet/predictions/protocol.json
.venv/bin/python -m unittest discover -s tests
.venv/bin/python validate.py --data-root data --work-root runtime/validation --datasets doclaynet mot20 coco_sama
.venv/bin/python audit_doclaynet_inference.py --pages runtime/doclaynet/pages.json --images-manifest runtime/doclaynet/images-manifest.json --predictions runtime/doclaynet/predictions/predictions.jsonl.gz --protocol runtime/doclaynet/predictions/protocol.json --images runtime/doclaynet/images --output data/doclaynet/inference-audit.json
.venv/bin/python audit_doclaynet_relations.py --raw-root raw --data-root data --pages runtime/doclaynet/pages.json --predictions runtime/doclaynet/predictions/predictions.jsonl.gz --output data/doclaynet/independent-audit.json
.venv/bin/python independent_coco_audit.py --raw-root raw --data-root data --output data/coco_sama/independent-audit.json
```

Retain the independent image/protocol/prediction audit as `doclaynet/inference-audit.json` and the full relation audit as `doclaynet/independent-audit.json`. Both image fetching and inference began with eight workers and resumed with sixteen; inference kept four PyTorch threads per worker. Construction concurrency is recorded separately from the frozen numerical protocol. The original construction's `run-history.json` records the complete run; the inference summary's elapsed time covers only its resumed phase.

The annotation fetcher uses HTTP byte ranges and ZIP/ZIP64 member extraction. It enforces `sources.lock.json`, checks archive ETag/size and the exact selected member set, verifies ZIP CRC32 when reusing files, and pins each extracted member's SHA-256. The DocLayNet image fetcher additionally obtains only the selected official test PNGs needed for model inference. These images are build inputs and are not redistributed in the processed dataset. Model files are pinned by revision, size, and SHA-256; inference verifies the model manifest before loading.

The adapter checks each result's exact protocol SHA-256, the protocol's page-manifest SHA-256, fixed postprocessing, page completeness, dimensions, and the original category map. The independent inference audit reads all selected PNG bytes and checks image hashes, CRCs, dimensions, prediction identity, and provenance. The separate relation audit checks full record membership against the selected original GT and frozen predictions, including geometry filtering, metadata, and group assignment. The validator checks output records, IDs, group slices, unit weights, common translations, actual slab containment, and group separation, and counts positive-area overlaps without saving join pairs. Geometric spot checks are diagnostics, not full-pair population estimates. The independent inference audit checks the protocol's recorded model manifest; it does not itself reread checkpoint weights."""


def updated_body(text, reports):
    protected = protected_contracts(text)
    info = reports["doclaynet"][0]
    intro = (f"Construction pipeline for three real-data rectangle relations for area-weighted and IoU-weighted spatial join sampling. "
             f"**Release {RELEASE} contains completed DocLayNet, MOT20, and COCO/Sama-COCO datasets.** "
             "Each retained source annotation or exported prediction becomes one rectangle with unit record weight. "
             "This release contains integer arrays and provenance metadata; images, segmentation masks, algorithm indexes, and materialized join pairs are not included.")
    title = re.search(r"^# VOSMA Real Rectangle Datasets[ \t]*\r?\n", text, re.M)
    table = re.search(r"^\| Dataset directory \|.*?(?=\n\n|\Z)", text, re.M | re.S)
    if not title or not table or table.start() < title.end():
        raise ValueError("Cannot identify the root README introduction/table")
    text = text[:title.end()] + "\n" + intro + "\n\n" + available_table(reports) + text[table.end():]

    status = section(text, "Release status").group(1)
    # Preserve the existing MOT/COCO-specific validation discussion verbatim.
    preserved = []
    for paragraph in status.strip().split("\n\n"):
        if paragraph.startswith("The pair counts are validation statistics."):
            preserved.append(paragraph)
        elif paragraph.startswith("Builder: "):
            preserved.append(paragraph)
    status = (f"All {EXPECTED_PAGES:,} selected official DocLayNet test pages completed fixed-model inference and the independent image/provenance audit. "
              "The three datasets passed record-level geometry and group-isolation validation.\n\n"
              + overlap_table(reports) + "\n\n" + "\n\n".join(preserved))
    text = replace_section(text, "Release status", status)
    source = section(text, "Source selection and grouping").group(1)
    start_mot = source.find("**MOT20.**")
    if start_mot < 0:
        raise ValueError("Cannot preserve the MOT/COCO source contracts")
    text = replace_section(text, "Source selection and grouping", doc_contract(info) + "\n\n" + source[start_mot:].strip())
    old_geometry = "The implementation parses JSON decimals as `Decimal` and performs endpoint arithmetic as exact rational arithmetic. It does not pass through float32."
    new_geometry = ("Raw annotation decimals are parsed as `Decimal`, and source endpoint arithmetic is exact. "
                    "The DocLayNet model itself intentionally runs in float32. Its restored PNG-coordinate `xyxy` outputs are serialized with Python float round-trip precision, "
                    "read back as exact decimals, and converted to `xywh` with exact endpoint differences before Q1000 quantization. "
                    "Dataset conversion performs no additional float32 coercion after the model output.")
    if old_geometry in text:
        text = text.replace(old_geometry, new_geometry, 1)
    elif new_geometry not in text:
        raise ValueError("Unrecognized core geometry paragraph")
    text = text.replace("Coordinates are not clipped or resized.", "Coordinates are not clipped or resized during dataset conversion.")
    text = text.replace("Equal rectangles remain distinct records when they originate from distinct source annotations.",
                        "Equal rectangles remain distinct records when they originate from distinct source annotation or prediction records.")
    files = section(text, "Files and schema").group(1)
    paragraphs = files.split("\n\n", 1)
    text = replace_section(text, "Files and schema", "Each of the three completed dataset directories contains the following files.\n\n" + paragraphs[1].strip())
    text = text.replace('revision="v0.1.0"', f'revision="{RELEASE}"')
    text = text.replace('dataset = "mot20"  # also available: coco_sama; doclaynet is pending',
                        'dataset = "doclaynet"  # also available: mot20, coco_sama')
    text = text.replace(
        "Keep download, source conversion, and file-loading costs separate from algorithm measurements, and report the release revision and requested sample count with results.",
        "Keep download, model inference, source conversion, and file-loading costs separate from algorithm measurements. "
        "Algorithm preprocessing remains part of measured algorithm cost. Report the release revision and requested sample count with results.",
    )
    text = replace_section(text, "Rebuild from original annotations", reproduce_section())
    model = info["model_protocol"]["model"]
    license_row = (f"| Aryn source model | [Apache-2.0 model card](https://huggingface.co/{model['model_id']}/blob/{model['revision']}/README.md) "
                   "| Model weights are fetched as build inputs; their pinned revision and license remain recorded in provenance. |")
    if "| Aryn source model |" not in text:
        anchor = "| MOT20-derived data |"
        position = text.find(anchor)
        if position < 0:
            raise ValueError("Cannot identify dataset license table")
        text = text[:position] + license_row + "\n" + text[position:]
    text = text.replace("Repeated annotation, detection versus GT, and annotation-version comparisons have different provenance",
                        "Model predictions versus GT, detection versus GT, and annotation-version comparisons have different provenance")
    text = text.replace("Install `requirements-hub.txt` for the optional Hub loader.", "Use `requirements-hub.txt` in a separate download-only environment. The inference environment already includes its compatible pinned Hub client; do not upgrade it to the loader-only pin.")
    text = re.sub(r"<!-- INTERNAL RELEASE CHECKLIST:.*?-->\s*", "", text, flags=re.S)
    if re.search(r"\bpending\b|precedence[- ]?1|precedence[- ]?2|\[DOC_\w+\]|v0\.1\.0", text, re.I):
        raise ValueError("Stale DocLayNet release prose remains; inspect before publication")
    if protected_contracts(text) != protected:
        raise ValueError("A protected MOT/COCO source contract changed")
    return text.rstrip() + "\n"


def split_frontmatter(text):
    match = re.match(r"\A---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    if not match:
        raise ValueError("The dataset README is missing its existing YAML front matter")
    return match.group(1), text[match.end():].lstrip("\n")


def doc_hf_configs(yaml):
    config = re.search(r"^configs:\s*\n", yaml, re.M)
    if config:
        next_key = re.search(r"^[A-Za-z_][A-Za-z0-9_-]*:", yaml[config.end():], re.M)
        end = config.end() + next_key.start() if next_key else len(yaml)
        existing = yaml[config.end():end]
        starts = list(re.finditer(r"^([ \t]*)-\s+config_name:\s*([^\n]+)\n?", existing, re.M))
        if existing.strip() and not starts:
            raise ValueError("Unsupported existing HF configs format")
        indent = len(starts[0].group(1)) if starts else 2
        blocks = []
        for index, start in enumerate(starts):
            stop = starts[index + 1].start() if index + 1 < len(starts) else len(existing)
            name = start.group(2).strip().strip("\"'")
            if name not in ("doclaynet_records", "doclaynet_groups"):
                blocks.append(existing[start.start():stop].rstrip())
    else:
        tags = re.search(r"^tags:", yaml, re.M)
        insertion = tags.start() if tags else len(yaml)
        before = yaml[:insertion]
        if before and not before.endswith("\n"):
            before += "\n"
        yaml = before + "configs:\n" + yaml[insertion:]
        return doc_hf_configs(yaml)
    prefix = " " * indent
    def entry(name, files):
        lines = [prefix + f"- config_name: {name}", prefix + "  data_files:"]
        for split, path in files:
            lines += [prefix + f"    - split: {split}", prefix + f"      path: {path}"]
        return "\n".join(lines)
    blocks += [entry("doclaynet_records", [("r1", "doclaynet/R1_metadata.parquet"), ("r2", "doclaynet/R2_metadata.parquet")]),
               entry("doclaynet_groups", [("groups", "doclaynet/groups.parquet")])]
    result = yaml[:config.end()] + "\n".join(blocks) + "\n" + yaml[end:]
    for name in ("doclaynet_records", "doclaynet_groups"):
        if len(re.findall(r"config_name:\s*" + name + r"\s*(?:\n|$)", result)) != 1:
            raise ValueError(f"HF config was not added uniquely: {name}")
    return result.rstrip()


def doc_readme(info, validation, audit):
    stats, inference = info["filter_statistics"], info["inference_counts"]
    model = info["model_protocol"]["model"]
    lines = [f"# DocLayNet GT and Fixed Predictions — {RELEASE}", "",
             f"This completed release pairs all **{info['group_count']:,} official DocLayNet test pages** with predictions from the pinned Aryn model. "
             "The two relations contain unit-weight rectangles and preserve every successful page as a group.", "",
             "| Quantity | Count |", "|---|---:|"]
    metrics = [
        ("Selected / successfully inferred / independently image-verified pages", info["group_count"]),
        ("R1 source GT records on selected pages", stats["r1.raw"]),
        ("R1 retained records", info["r1_count"]),
        ("R2 exported predictions before geometry filtering", stats["r2.raw"]),
        ("R2 retained records", info["r2_count"]),
        ("Positive-area record pairs", validation["within_group_overlap_count"]),
        ("Groups with positive mass", validation["groups_with_positive_mass"]),
        ("Groups with zero mass", validation["groups_with_zero_mass"]),
        ("Successful pages with zero exported predictions", inference["zero_output_successful_pages"]),
    ]
    lines += [f"| {name} | {number:,} |" for name, number in metrics]
    states = info['group_states']
    empty_sides = sum(states.get(k, 0) for k in ('r1_only', 'r2_only', 'both_empty'))
    nonempty_zero = validation['groups_with_zero_mass'] - empty_sides
    lines += ["", f"The {validation['groups_with_zero_mass']:,} zero-mass groups comprise {states.get('r1_only',0):,} R1-only groups, {states.get('r2_only',0):,} R2-only groups, {states.get('both_empty',0):,} groups with both sides empty, and {nonempty_zero:,} groups whose nonempty sides have no positive-area intersection. Zero mass and zero exported predictions are different conditions."]
    lines += ["", "No materialized join-pair list is included. Overlap counts use strictly positive intersection width and height.",
              "", "## Selection and fixed inference", "", doc_contract(info).replace(
                  "See `doclaynet/README.md` for the completed counts and hash links.",
                  "The completed counts and frozen input hashes appear below."), "",
              "## Candidate and geometry accounting", "", "| Fixed model stage | Count |", "|---|---:|"]
    for key in ("top_k_candidates", "nonfinite_score_excluded", "threshold_excluded", "background_excluded", "exported_predictions"):
        lines.append(f"| `{key}` | {inference[key]:,} |")
    lines += ["", "The three model exclusions are mutually exclusive. They precede common geometry filtering and are separate from `r2.excluded.*`.",
              "", "| Geometry exclusion reason | R1 | R2 |", "|---|---:|---:|"]
    reasons = sorted({key.split(".excluded.", 1)[1] for key in stats
                      if key.startswith(("r1.excluded.", "r2.excluded."))})
    if reasons:
        for reason in reasons:
            lines.append(f"| `{reason}` | {stats.get('r1.excluded.' + reason, 0):,} | {stats.get('r2.excluded.' + reason, 0):,} |")
    else:
        lines.append("| No geometry exclusions | 0 | 0 |")
    lines += ["", "Raw GT decimals and serialized model PNG-coordinate `xyxy` values are parsed exactly. "
              "Prediction widths and heights are formed with exact decimal endpoint differences. "
              "Q1000 then rounds endpoints to nearest integers with ties to even; no additional float32 coercion occurs after model inference. "
              "Negative and out-of-bounds coordinates remain. Both relations receive the same translation per page and occupy disjoint page slabs. "
              "Classes and scores remain metadata; all record weights are 1.",
              "", "## Frozen provenance and verification", "",
              f"- Model: [{model['model_id']}](https://huggingface.co/{model['model_id']}/tree/{model['revision']}), revision `{model['revision']}`.",
              "- Model license: [Apache-2.0](Model-Apache-2.0.txt); DocLayNet-derived data license: CDLA-Permissive-1.0.",
              "- Model training overlap with the selected test pages: **unknown**.",
              f"- Independent inference/image audit: passed; {audit['verified_images']:,} PNGs verified from their actual bytes.",
              "- `dataset.json` records the full unchanged model protocol and source selection; `validation.json` records geometry and group checks; `inference-audit.json` records the independent image and prediction audit; `independent-audit.json` checks full retained-record membership, geometry, metadata, and grouping against original GT and frozen predictions.",
              "- `postprocess-check.json` independently checks the saved single-page raw tensors against the exported predictions. Its replay inputs are included as `sources/pilot.npz` and `sources/pilot.json`; run `check_prediction_postprocess.py` with width and height 1025. This pilot check is separate from the full image and full relation audits.",
              "- The independent inference audit checks the model manifest recorded in the protocol. Checkpoint file verification belongs to the fixed model download and inference loader; the audit does not reload model weights.",
              "", "| Frozen input | SHA-256 |", "|---|---|"]
    for key in ("pages_manifest", "predictions", "protocol"):
        source = info["prediction_provenance"][key]
        lines.append(f"| {source['file']} | `{source['sha256']}` |")
    lines += ["", "The protocol names the exact page-manifest SHA-256, and every prediction result names the exact protocol SHA-256. "
              "The independent audit checks these links against the frozen files and selected image manifest. "
              "Original model rank, query index, label, score, `xyxy`, PNG hash, and source identity remain in rectangle or page metadata.",
              "", "## Files and reuse", "",
              "Use `R1.npy`, `R2.npy`, and `groups.parquet` from the same immutable dataset commit. "
              "Record `(relation, record_id)` to identify generated records. HF configs `doclaynet_records` and `doclaynet_groups` expose the Parquet metadata; the arrays contain the benchmark input coordinates. "
              "Images and model weights are not included. The [root README](../README.md) documents schema, loading, reproduction commands, and source-specific licenses.", ""]
    return "\n".join(lines)


def atomic_write(path, content):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update(root, dry_run=False):
    reports, audit = completed_reports(root)
    builder = root / "builder" / "README.md"
    data = root / "data" / "README.md"
    doc = root / "data" / "doclaynet" / "README.md"
    yaml, data_body = split_frontmatter(data.read_text(encoding="utf-8"))
    contents = {
        builder: updated_body(builder.read_text(encoding="utf-8"), reports),
        data: "---\n" + doc_hf_configs(yaml) + "\n---\n\n" + updated_body(data_body, reports),
        doc: doc_readme(*reports["doclaynet"], audit),
    }
    for path, content in contents.items():
        if not dry_run:
            atomic_write(path, content)
        print(("CHECKED " if dry_run else "UPDATED ") + str(path))
    return contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    update(args.root.resolve(), args.dry_run)


if __name__ == "__main__":
    main()
