# VOSMA Real Rectangle Datasets

Construction pipeline for three real-data rectangle relations for area-weighted and IoU-weighted spatial join sampling. **Release v0.1.0 contains two completed datasets: MOT20 and COCO/Sama-COCO. DocLayNet is pending because the official source snapshot lacks the required second annotation layer.** Each retained source annotation becomes one rectangle with unit record weight. This release contains integer arrays and provenance metadata; images, segmentation masks, algorithm indexes, and materialized join pairs are not included.

| Dataset directory | R1 | R2 | Retained R1 records | Retained R2 records | Groups |
|---|---|---|---:|---:|---:|
| `doclaynet` | Original annotation precedence 0 | Original annotation precedence 1 | Pending | Pending | Pending |
| `mot20` | Valid pedestrian ground truth | Official detections | 1,134,614 | 661,143 | 8,931 |
| `coco_sama` | COCO 2017 non-crowd instances | Sama-COCO non-crowd instances | 886,282 | 1,068,028 | 386,072 |

Counts refer to this processed release. Empty sides and groups remain represented according to the group universes below. The per-dataset `dataset.json` and `validation.json` provide filtering counts and validation results.

## Release status

The official DocLayNet Core snapshot has 80,863 physical pages and 1,107,470 annotation records, **all with precedence 0**. No precedence-1 or precedence-2 records occur in any split. It therefore cannot produce the intended paired annotation dataset. `doclaynet/source-audit.json` contains the complete counts, and `doclaynet/README.md` explains the evidence. This release does not duplicate the first layer, synthesize annotations, or substitute PDF text cells. The DocLayNet builder fails explicitly when the required layers are absent.

| Completed dataset | Positive-area record pairs | Groups with positive mass | Groups with zero mass |
|---|---:|---:|---:|
| MOT20 | 3,176,148 | 8,931 | 0 |
| COCO/Sama-COCO | 1,508,091 | 322,274 | 63,798 |

The pair counts are validation statistics. No pair list is distributed. COCO has two zero-height non-crowd boxes; Sama has eight zero-width/height non-crowd boxes. These ten source records are itemized in `coco_sama/independent-audit.json`. All 119 train and 5 validation Sama shards passed an independent manifest, identity, grouping, and row-count audit. Out-of-bounds diagnostics use quantized local coordinates and the source's pixel origin.

Builder: [DANNHIROAKI/VOSMA-Dataset-Build](https://github.com/DANNHIROAKI/VOSMA-Dataset-Build). Built arrays: [DannHiroaki/VOSMA-Dataset](https://huggingface.co/datasets/DannHiroaki/VOSMA-Dataset). See `BUILD_INFO.json` for the construction commit and environment.

## Source selection and grouping

**DocLayNet (pending).** The intended source is the original [DocLayNet 1.0.0 Core archive](https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip), using `COCO/train.json`, `COCO/val.json`, and `COCO/test.json`. The physical page key is `(doc_category, collection, doc_name, page_no)`. Image file identity and declared dimensions are checked before pairing; unresolved identity or layer conflicts stop the build. Each annotation's own `precedence` selects its side: 0 for R1 and 1 for R2. Precedence 2 is counted but excluded from the primary relations. Image-level precedence is preserved as metadata and does not replace annotation-level precedence. Pages originally containing both selected layers remain groups even if geometry filtering empties a side. Categories are retained in metadata, and category disagreements remain in the page-level comparison. See the [official schema](https://github.com/DS4SD/DocLayNet#coco-annotations).

**MOT20.** The source is [MOT20Labels.zip](https://motchallenge.net/data/MOT20Labels.zip). Only training sequences `MOT20-01`, `MOT20-02`, `MOT20-03`, and `MOT20-05` are used. A group is `(sequence, frame)`; all 8,931 frames are retained. R1 keeps GT rows with `valid=1` and `class=1`, without a visibility threshold. R2 keeps official detections with a finite score and valid geometry, without a score threshold. Detection scores, track IDs, and visibility are metadata; they never become sampling weights. The original MOT coordinate convention is preserved before quantization, with no one-pixel origin adjustment. [Official benchmark and downloads](https://motchallenge.net/data/MOT20/).

**COCO/Sama-COCO.** R1 uses the [COCO 2017 train/validation instance annotations](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip); R2 uses the official [Sama train](https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-train.zip) and [Sama validation](https://sama-documentation-assets.s3.amazonaws.com/sama-coco/sama-coco-val.zip) annotations. Images are paired by source file identity within the original split, with dimension checks. Category names establish the mapping between the 80 classes. A group is `(split, image, category)` from the union of category mentions on either side before filtering. Both sides require `iscrowd=0`. Annotation-free images remain in `images.parquet`; unmentioned image-category combinations are not synthesized as groups. Sama reannotation began with COCO annotations, so this is a comparison of annotation versions, not a claim of independent blind annotation. [Sama's description of its labeling process](https://www.sama.com/sama-coco-dataset).

## Geometry contract

For each source `bbox=[x,y,w,h]`, construct the endpoints exactly from the original decimal values, then compute:

```text
Q(z) = nearest integer to 1000*z, with exact ties rounded to even
(x0,y0,x1,y1) = (Q(x), Q(y), Q(x+w), Q(y+h))
```

The implementation parses JSON decimals as `Decimal` and performs endpoint arithmetic as exact rational arithmetic. It does not pass through float32. Rectangles are half-open: `[x0,x1) × [y0,y1)`. A join pair requires strictly positive intersection width and height; boundary contact alone is not an intersection.

Malformed or non-finite boxes, nonpositive source widths/heights, and rectangles that become degenerate after quantization are excluded and counted. Finite negative coordinates and boxes outside the declared image bounds are retained. Coordinates are not clipped or resized. Representability failures stop the build rather than silently wrapping integers.

After quantization, each group receives one shared integer translation for both R1 and R2. Its minimum retained y-coordinate becomes 1. Groups occupy disjoint x intervals with a one-unit gap, using the union of retained rectangle extents on both sides. The translation and local bounds are stored in `groups.parquet`. Translation preserves the quantized within-group geometry, areas, and IoUs while preventing intersections between different groups. Quantization itself may change the original continuous geometry slightly.

Every output `weight` is 1. Area sampling uses the quantized intersection area; IoU sampling uses intersection area divided by union area. No object matching, IoU threshold, NMS, deduplication, bbox merging, or annotation-count balancing is performed. Equal numbers of samples per group do not implement global mass-weighted sampling. Algorithm preprocessing remains part of measured algorithm cost. Int64 coordinate storage does not guarantee that area or cumulative-mass intermediates fit int64. Equal rectangles remain distinct records when they originate from distinct source annotations.

## Files and schema

Each completed dataset directory contains the following files. The pending `doclaynet/` directory contains status, source audits, licenses, and provenance only; it has no runtime arrays.

| File | Contents |
|---|---|
| `R1.npy`, `R2.npy` | NumPy arrays with shape `(N,7)`, little-endian signed int64, no pickle |
| `R1_metadata.parquet`, `R2_metadata.parquet` | One row per retained rectangle, aligned with array row order |
| `groups.parquet` | Group identity, image dimensions, common translation, bounds, contiguous array slices, empty-side state, and validated overlap counts |
| `dataset.json` | Schema version, relation definitions, counts, filtering statistics, and numerical contract |
| `validation.json` | Validation results and clearly scoped geometric spot checks |
| `exclusions.jsonl.gz` | Rejected source records and exclusion reasons |
| `SHA256SUMS` | Checksums for the processed files |

Dataset-specific pairing reports, category mappings, and `images.parquet` are included where applicable. Each dataset's `sources/*.json` records source URLs, versions, retrieval metadata, archive members, and SHA-256 checksums of the extracted original annotation files. Source fingerprints identify the exact input bytes used for the release.

The seven array columns are:

```text
record_id, group_id, x0, y0, x1, y1, weight
```

`record_id` is a zero-based row number unique within its relation; `(relation, record_id)` is the full generated record identity. It is not the source annotation ID. `group_id` is shared across the two relations. Coordinates are quantized and translated; `weight` is always 1.

Metadata includes source file, record position, annotation/image/category IDs, the original bbox decimal values, quantized local endpoints, and source-specific attributes. In `groups.parquet`, `r1_start/r1_count` and `r2_start/r2_count` identify contiguous row slices. `state` is one of `both_nonempty`, `r1_only`, `r2_only`, or `both_empty`. A group with two nonempty sides can still have zero positive-area join pairs.

## Load a frozen release

Use an immutable Hugging Face commit SHA rather than a moving branch. Install `numpy`, `pyarrow`, and `huggingface_hub`, then:

```python
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download, HfApi

revision = HfApi().dataset_info("DannHiroaki/VOSMA-Dataset", revision="v0.1.0").sha
print("Record this immutable revision with your experiment:", revision)
dataset = "mot20"  # also available: coco_sama; doclaynet is pending
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

The arrays can be used directly as benchmark inputs. Keep download, source conversion, and file-loading costs separate from algorithm measurements, and report the release revision and requested sample count with results.

## Rebuild from original annotations

From the builder repository root, with Python 3.12 and a C++17 compiler available:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python fetch_sources.py --raw-root work/raw --sources mot20 coco2017 sama_train sama_val
.venv/bin/python build.py --raw-root work/raw --output-root outputs/data --datasets mot20 coco_sama
.venv/bin/python -m unittest discover -s tests
.venv/bin/python validate.py --data-root outputs/data --work-root work/validation --datasets mot20 coco_sama
.venv/bin/python independent_coco_audit.py --raw-root work/raw --data-root outputs/data --output outputs/data/coco_sama/independent-audit.json
```

The fetcher uses HTTP byte ranges and ZIP/ZIP64 member extraction to retrieve annotation files without downloading the image collections. It enforces `sources.lock.json`, checks archive ETag/size and the exact selected member set, verifies ZIP CRC32 when reusing files, and pins each extracted member's SHA-256; the builder verifies source member hashes before conversion. The validator checks every output record, relation IDs, group slices, unit weights, translations, and separation between groups. It also counts positive-area overlaps without saving join pairs. Spot-check distributions are diagnostics, not estimates of the full pair distribution.

## Licenses

Licenses apply to each dataset portion separately; this collection does not replace them with one blanket license.

| Portion | Source license | Conditions carried with this release |
|---|---|---|
| DocLayNet-derived data | [CDLA-Permissive-1.0](https://github.com/DS4SD/DocLayNet/blob/main/LICENSE) | Preserve source attribution and license access; identify the modifications described above. |
| MOT20-derived data | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | Attribution, noncommercial use, and ShareAlike conditions apply. |
| COCO/Sama-COCO-derived annotation data | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Preserve attribution and license access, and indicate modifications. |

Credit the original DocLayNet, MOT20, COCO, and Sama-COCO creators when using their portions. This derivative changes annotation selection, numerical representation, and group placement as documented above. Source image licenses are separate; source images are not redistributed here. See the [COCO terms](https://cocodataset.org/#termsofuse) and [Sama dataset information](https://www.sama.com/sama-coco-dataset) for the original resources.

## Cite the sources

- Pfitzmann et al. (2022), [DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis](https://arxiv.org/abs/2206.01062).
- Dendorfer et al. (2020), [MOT20: A benchmark for multi object tracking in crowded scenes](https://arxiv.org/abs/2003.09003).
- Lin et al. (2014), [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312).
- Zimmermann et al. (2023), [Benchmarking a Benchmark: How Reliable is MS-COCO?](https://arxiv.org/abs/2311.02709).

Also identify this dataset repository and the exact release commit SHA in experiment reports. Repeated annotation, detection versus GT, and annotation-version comparisons have different provenance and should be reported separately.

## Software license

The construction software is MIT-licensed. The dataset-specific licenses remain applicable to source and derived data. See `LICENSE` and `licenses/`. Install `requirements-hub.txt` for the optional Hub loader.
