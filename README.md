# VOSMA Real Rectangle Datasets

Construction pipeline for four real-data rectangle relations for area-weighted and IoU-weighted spatial join sampling. **Release v0.3.0 adds the Kenya Microsoft/Google building-rectangle dataset. The completed DocLayNet, MOT20, and COCO/Sama-COCO files from v0.2.0 are preserved unchanged.** Each retained source annotation or exported prediction becomes one rectangle with unit record weight. This release contains integer arrays and provenance metadata; images, segmentation masks, algorithm indexes, and materialized join pairs are not included.

| Dataset directory | R1 | R2 | Retained R1 records | Retained R2 records | Groups |
|---|---|---|---:|---:|---:|
| `doclaynet` | Official test-page ground truth | Fixed Aryn model predictions | 66,531 | 50,084 | 4,999 |
| `mot20` | Valid pedestrian ground truth | Official detections | 1,134,614 | 661,143 | 8,931 |
| `coco_sama` | COCO 2017 non-crowd instances | Sama-COCO non-crowd instances | 886,282 | 1,068,028 | 386,072 |
| `kenya_buildings` | Microsoft Kenya footprint rectangles | Google Open Buildings V3 footprint rectangles | 12,798,990 | 25,560,434 | 1 |

Counts refer to this processed release. Empty sides and groups remain represented according to the group universes below. The per-dataset `dataset.json` and `validation.json` provide filtering counts and validation results.

## Release status

All 4,999 selected official DocLayNet test pages completed fixed-model inference and the independent image/provenance audit. The original three datasets passed record-level geometry and group-isolation validation. Kenya passed checks of every published record and aligned metadata, sampled original-polygon reprojection checks, and independent fixed-window diagnostics. The pair-count table below describes the original three datasets; Kenya national pairs were not enumerated.

| Completed dataset | Positive-area record pairs | Groups with positive mass | Groups with zero mass |
|---|---:|---:|---:|
| DocLayNet | 52,670 | 4,876 | 123 |
| MOT20 | 3,176,148 | 8,931 | 0 |
| COCO/Sama-COCO | 1,508,091 | 322,274 | 63,798 |

The pair counts are validation statistics. No pair list is distributed. COCO has two zero-height non-crowd boxes; Sama has eight zero-width/height non-crowd boxes. These ten source records are itemized in `coco_sama/independent-audit.json`. All 119 train and 5 validation Sama shards passed an independent manifest, identity, grouping, and row-count audit. Out-of-bounds diagnostics use quantized local coordinates and the source's pixel origin.

Builder: [DANNHIROAKI/VOSMA-Dataset-Build](https://github.com/DANNHIROAKI/VOSMA-Dataset-Build). Built arrays: [DannHiroaki/VOSMA-Dataset](https://huggingface.co/datasets/DannHiroaki/VOSMA-Dataset). See `BUILD_INFO.json` for the construction commit and environment.

## Source selection and grouping

**DocLayNet.** R1 contains all original precedence-0 ground-truth boxes on all 4,999 official test pages from the [DocLayNet 1.0.0 Core archive](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip), using `COCO/test.json`. R2 contains the exported predictions of the fixed [Aryn/deformable-detr-DocLayNet model](https://huggingface.co/Aryn/deformable-detr-DocLayNet/tree/d5503a90ae08dd43565de6984a5dd7924cad2400). A group is `(split, source_image_id)`. The physical key `(doc_category, collection, doc_name, page_no)` is unique across selected pages; official PNG filename, dimensions, source identity, and per-image bytes are checked. All selected pages remain groups, including successful pages with zero predictions. Missing, failed, or duplicate inference pages stop the build. All classes remain in their page groups, so cross-class intersections are included. Predictions are not matched to GT or selected using GT.

The released model run uses CPU float32, batch size 1, and four PyTorch threads per worker. It began with eight workers and resumed with sixteen workers under the same frozen numerical protocol; completed page results were reused only after protocol and image-hash checks. The reproduction command uses sixteen workers. The fixed processor produces `[1,3,800,800]` model inputs and restores predicted `xyxy` boxes to each original PNG's dimensions. Native postprocessing takes the top 100 scores over the flattened 200-query × 12-class sigmoid output. The mutually exclusive exclusions are nonfinite scores, finite scores at most 0.7, and remaining class-0 (`N/A`) candidates, in that order. All remaining records are exported. There is no NMS, clipping, deduplication, GT-based threshold tuning, or class-pair filtering. Original top-100 rank, query index, label, score, and PNG-coordinate endpoints are retained in metadata. Geometry exclusions are applied subsequently by the common dataset conversion.

The checkpoint revision is `d5503a90ae08dd43565de6984a5dd7924cad2400` and its model-card license is Apache-2.0. The model's training overlap with these DocLayNet test pages is **unknown**. These relations measure spatial-join sampling on GT and frozen predictions; they do not establish an unseen-test-set detection result. Model files, page selection, protocol, predictions, and image hashes are recorded in provenance; `doclaynet/inference-audit.json` reports the independent all-page PNG, provenance, and prediction-accounting audit. See `doclaynet/README.md` for the completed counts and hash links.

**MOT20.** The source is [MOT20Labels.zip](https://motchallenge.net/data/MOT20Labels.zip). Only training sequences `MOT20-01`, `MOT20-02`, `MOT20-03`, and `MOT20-05` are used. A group is `(sequence, frame)`; all 8,931 frames are retained. R1 keeps GT rows with `valid=1` and `class=1`, without a visibility threshold. R2 keeps official detections with a finite score and valid geometry, without a score threshold. Detection scores, track IDs, and visibility are metadata; they never become sampling weights. The original MOT coordinate convention is preserved before quantization, with no one-pixel origin adjustment. [Official benchmark and downloads](https://motchallenge.net/data/MOT20/).

**COCO/Sama-COCO.** R1 uses the [COCO 2017 train/validation instance annotations](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip); R2 uses the official [Sama train](https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-train.zip) and [Sama validation](https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip) annotations. Images are paired by source file identity within the original split, with dimension checks. Category names establish the mapping between the 80 classes. A group is `(split, image, category)` from the union of category mentions on either side before filtering. Both sides require `iscrowd=0`. Annotation-free images remain in `images.parquet`; unmentioned image-category combinations are not synthesized as groups. Sama reannotation began with COCO annotations, so this is a comparison of annotation versions, not a claim of independent blind annotation. [Sama's description of its labeling process](https://www.sama.com/sama-coco-dataset).

**Kenya buildings.** R1 is the original Microsoft Kenya national building-footprint release; R2 is Google Open Buildings V3. A frozen full-resolution geoBoundaries Kenya ADM0 defines the common study region. Full source polygons are projected to one fixed WGS84 LAEA coordinate system, selected by projected polygon-centroid coverage, converted to axis-aligned bounds, quantized at 1000 units per metre, and translated by one shared integer offset. The entire country is one group. No source confidence cutoff, geometry repair, deduplication, object matching or tile isolation is added. The millimetre grid is a numerical representation, not a claim about imagery accuracy. See the [construction guide](https://github.com/DANNHIROAKI/VOSMA-Dataset-Build/blob/v0.3.0/KENYA_BUILD.md) and [published dataset documentation](https://huggingface.co/datasets/DannHiroaki/VOSMA-Dataset/blob/v0.3.0/kenya_buildings/README.md).

## Geometry contract

For the first three image-based datasets, for each source `bbox=[x,y,w,h]`, construct the endpoints exactly from the original decimal values, then compute:

```text
Q(z) = nearest integer to 1000*z, with exact ties rounded to even
(x0,y0,x1,y1) = (Q(x), Q(y), Q(x+w), Q(y+h))
```

Raw annotation decimals are parsed as `Decimal`, and source endpoint arithmetic is exact. The DocLayNet model itself intentionally runs in float32. Its restored PNG-coordinate `xyxy` outputs are serialized with Python float round-trip precision, read back as exact decimals, and converted to `xywh` with exact endpoint differences before Q1000 quantization. Dataset conversion performs no additional float32 coercion after the model output. Rectangles are half-open: `[x0,x1) × [y0,y1)`. A join pair requires strictly positive intersection width and height; boundary contact alone is not an intersection.

Malformed or non-finite boxes, nonpositive source widths/heights, and rectangles that become degenerate after quantization are excluded and counted. Finite negative coordinates and boxes outside the declared image bounds are retained. Coordinates are not clipped or resized during dataset conversion. Representability failures stop the build rather than silently wrapping integers.

For these image-based datasets, after quantization each group receives one shared integer translation for both R1 and R2. Its minimum retained y-coordinate becomes 1. Groups occupy disjoint x intervals with a one-unit gap, using the union of retained rectangle extents on both sides. The translation and local bounds are stored in `groups.parquet`. Translation preserves the quantized within-group geometry, areas, and IoUs while preventing intersections between different groups. Quantization itself may change the original continuous geometry slightly.

Every output `weight` is 1. Area sampling uses the quantized intersection area; IoU sampling uses intersection area divided by union area. No object matching, IoU threshold, NMS, deduplication, bbox merging, or annotation-count balancing is performed. Equal numbers of samples per group do not implement global mass-weighted sampling. Algorithm preprocessing remains part of measured algorithm cost. Int64 coordinate storage does not guarantee that area or cumulative-mass intermediates fit int64. Equal rectangles remain distinct records when they originate from distinct source annotation or prediction records.

## Files and schema

The four dataset directories share the following array and metadata schema. Kenya source provenance uses `SOURCE_LOCK.json`, `country_boundary.geojson`, and `sources/`; its validation reports local overlap diagnostics rather than an exhaustive national pair count.

| File | Contents |
|---|---|
| `R1.npy`, `R2.npy` | NumPy arrays with shape `(N,7)`, little-endian signed int64, no pickle |
| `R1_metadata.parquet`, `R2_metadata.parquet` | One row per retained rectangle, aligned with array row order |
| `groups.parquet` | Group identity, coordinate-frame dimensions, common translation, bounds, contiguous array slices, empty-side state, and available overlap statistics |
| `dataset.json` | Schema version, relation definitions, counts, filtering statistics, and numerical contract |
| `validation.json` | Validation results and clearly scoped geometric spot checks |
| `exclusions.jsonl.gz` | Rejected source records and exclusion reasons |
| `SHA256SUMS` | Checksums for the processed files |

Dataset-specific pairing reports, category mappings, and `images.parquet` are included where applicable. For the first three datasets, `sources/*.json` records source URLs, versions, retrieval metadata, archive members, and SHA-256 checksums of the extracted original annotation files. For Kenya, `SOURCE_LOCK.json` pins the complete original building archives and boundary file; `sources/` preserves small upstream notices and tile metadata. Source fingerprints identify the exact input bytes used for the release.

The seven array columns are:

```text
record_id, group_id, x0, y0, x1, y1, weight
```

`record_id` is a zero-based row number unique within its relation; `(relation, record_id)` is the full generated record identity. It is not the source annotation ID. `group_id` is shared across the two relations. Coordinates are quantized and translated; `weight` is always 1.

For Kenya, the source record position is a 1-based original JSONL record/line for Microsoft and a 1-based data-row index excluding the header for Google; physical line numbers are also preserved.

Metadata includes source file, record position, annotation/image/category IDs, the original bbox decimal values (projected metre bounds for Kenya), quantized local endpoints, and source-specific attributes. In `groups.parquet`, `r1_start/r1_count` and `r2_start/r2_count` identify contiguous row slices. `state` is one of `both_nonempty`, `r1_only`, `r2_only`, or `both_empty`. A group with two nonempty sides can still have zero positive-area join pairs.

## Load a frozen release

Use an immutable Hugging Face commit SHA rather than a moving branch. Install `numpy`, `pyarrow`, and `huggingface_hub`, then:

```python
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, HfApi

revision = HfApi().dataset_info("DannHiroaki/VOSMA-Dataset", revision="v0.3.0").sha
print("Record this immutable revision with your experiment:", revision)
dataset = "doclaynet"  # also available: mot20, coco_sama, kenya_buildings
root = Path(snapshot_download(
    repo_id="DannHiroaki/VOSMA-Dataset",
    repo_type="dataset",
    revision=revision,
    allow_patterns=[f"{dataset}/*"],
)) / dataset

r1 = np.load(root / "R1.npy", mmap_mode="r", allow_pickle=False)
r2 = np.load(root / "R2.npy", mmap_mode="r", allow_pickle=False)
groups = pq.read_table(root / "groups.parquet")
info = json.loads((root / "dataset.json").read_text())
assert info["schema_version"] == "vosma-real-rectangles-v1"
assert r1.shape[1] == r2.shape[1] == 7

# Read one group's original row slices, including an empty side if present.
g = groups.slice(0, 1).to_pylist()[0]
a = r1[g["r1_start"]:g["r1_start"] + g["r1_count"]]
b = r2[g["r2_start"]:g["r2_start"] + g["r2_count"]]
print(r1.shape, r2.shape, groups.num_rows, a.shape, b.shape)
```

The arrays can be used directly as benchmark inputs. Keep download, model inference, source conversion, and file-loading costs separate from algorithm measurements. Algorithm preprocessing remains part of measured algorithm cost. Report the release revision and requested sample count with results.

## Rebuild from original annotations

The commands below reproduce the first three datasets. For the fourth dataset, follow [KENYA_BUILD.md](https://github.com/DANNHIROAKI/VOSMA-Dataset-Build/blob/v0.3.0/KENYA_BUILD.md), using the committed `kenya-sources.lock.json`.

From the builder repository root, with Python 3.12 and a C++17 compiler available:

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

The adapter checks each result's exact protocol SHA-256, the protocol's page-manifest SHA-256, fixed postprocessing, page completeness, dimensions, and the original category map. The independent inference audit reads all selected PNG bytes and checks image hashes, CRCs, dimensions, prediction identity, and provenance. The separate relation audit checks full record membership against the selected original GT and frozen predictions, including geometry filtering, metadata, and group assignment. The validator checks output records, IDs, group slices, unit weights, common translations, actual slab containment, and group separation, and counts positive-area overlaps without saving join pairs. Geometric spot checks are diagnostics, not full-pair population estimates. The independent inference audit checks the protocol's recorded model manifest; it does not itself reread checkpoint weights.

## Licenses

Licenses apply to each dataset portion separately; this collection does not replace them with one blanket license.

| Portion | Source license | Conditions carried with this release |
|---|---|---|
| DocLayNet-derived data | [CDLA-Permissive-1.0](https://github.com/DS4SD/DocLayNet/blob/main/LICENSE) | Preserve source attribution and license access; identify the modifications described above. |
| Aryn source model | [Apache-2.0 model card](https://huggingface.co/Aryn/deformable-detr-DocLayNet/blob/d5503a90ae08dd43565de6984a5dd7924cad2400/README.md) | Model weights are fetched as build inputs; their pinned revision and license remain recorded in provenance. |
| MOT20-derived data | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | Attribution, noncommercial use, and ShareAlike conditions apply. |
| COCO/Sama-COCO-derived annotation data | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Preserve attribution and license access, and indicate modifications. |
| Kenya Microsoft/Google-derived rectangle data | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) | Microsoft specifies ODbL; the Google dual-license ODbL option is selected. Source attribution and the complete license accompany the fourth dataset. |
| Kenya boundary | Public Domain upstream boundary, with geoBoundaries/RCMRD attribution | Fixed source, version, and processing are recorded with the fourth dataset. |

Credit the original DocLayNet, MOT20, COCO, Sama-COCO, Microsoft, Google, and geoBoundaries/RCMRD creators when using their portions. This derivative changes annotation selection, numerical representation, and group placement as documented above. Source image licenses are separate; source images are not redistributed here. See the [COCO terms](https://cocodataset.org/#termsofuse) and [Sama dataset information](https://www.sama.com/sama-coco-dataset) for the original resources.

## Cite the sources

- Pfitzmann et al. (2022), [DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis](https://arxiv.org/abs/2206.01062).
- Dendorfer et al. (2020), [MOT20: A benchmark for multi object tracking in crowded scenes](https://arxiv.org/abs/2003.09003).
- Lin et al. (2014), [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312).
- Zimmermann et al. (2023), [Benchmarking a Benchmark: How Reliable is MS-COCO?](https://arxiv.org/abs/2311.02709).

Also identify this dataset repository and the exact release commit SHA in experiment reports. Model predictions versus GT, detection versus GT, annotation-version comparisons, and cross-source building detections have different provenance and should be reported separately.

## Software license

The construction software is MIT-licensed. The dataset-specific licenses remain applicable to source and derived data. See `LICENSE` and `licenses/`. Use `requirements-hub.txt` in a separate download-only environment. The inference environment already includes its compatible pinned Hub client; do not upgrade it to the loader-only pin.
