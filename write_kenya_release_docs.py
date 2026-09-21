#!/usr/bin/env python3
"""Document one validated Kenya release; never touch other dataset directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

BUILDER = "https://github.com/DANNHIROAKI/VOSMA-Dataset-Build"
MICROSOFT = "https://github.com/microsoft/KenyaNigeriaBuildingFootprints"
GOOGLE = "https://sites.research.google/open-buildings/"
BOUNDARY_COMMIT = "9469f09592ced973a3448cf66b6100b741b64c0d"
BOUNDARY_SHA256 = "89248c71d10e86d76a2ca336aa2444aac9901651c9cdb974a83f29fbb72c55fa"
BOUNDARY_BASE = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/"
                 + BOUNDARY_COMMIT + "/releaseData/gbOpen/KEN/ADM0/")
LICENSE_URL = "https://opendatacommons.org/licenses/odbl/odbl-10.txt"
LICENSE_SHA256 = "607680718977f6f6c9607972afd98f208573f19251315ed1362a8589b51beaf5"
SNAPSHOTS = {
    "microsoft-readme.md": "https://raw.githubusercontent.com/microsoft/"
        "KenyaNigeriaBuildingFootprints/13adff8da80c4980f00f586459dcef4e7ba438c8/README.md",
    "kenya-boundary-metadata.json": BOUNDARY_BASE + "geoBoundaries-KEN-ADM0-metaData.json",
    "kenya-boundary-metadata.txt": BOUNDARY_BASE + "geoBoundaries-KEN-ADM0-metaData.txt",
    "google-tiles.geojson": "https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson",
    "google-extra-tiles.json": "https://storage.googleapis.com/storage/v1/b/open-buildings-data/o",
    "google-187-prefix-list.json": "https://storage.googleapis.com/storage/v1/b/open-buildings-data/o?prefix=v3%2Fpolygons_s2_level_4_gzip%2F187",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf8")
    temporary.replace(path)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def coordinate_bounds(geojson):
    positions = []

    def visit(value):
        if isinstance(value, dict):
            for key in ("features", "geometry", "coordinates"):
                if key in value:
                    visit(value[key])
        elif isinstance(value, list):
            if len(value) >= 2 and all(isinstance(x, (int, float)) for x in value[:2]):
                positions.append(value[:2])
            else:
                for item in value:
                    visit(item)

    visit(geojson)
    return [min(p[0] for p in positions), min(p[1] for p in positions),
            max(p[0] for p in positions), max(p[1] for p in positions)]


def validate_local_overlay(root, info):
    import pyproj

    report = read_json(root / "qa_local_overlay.json")
    if report.get("passed") is not True or report.get("dataset") != "kenya_buildings":
        raise ValueError("Successful fixed-region original-polygon QA is required")
    if report.get("source_lock_sha256") != info["source_lock_sha256"] or report.get("coordinate_offset") != info["coordinate_offset"]:
        raise ValueError("Local QA does not match this dataset's source lock and offset")
    for side in ("r1", "r2"):
        if report.get(f"{side}_count") != info[f"{side}_count"]:
            raise ValueError("Local QA does not match this dataset's record counts")
    if report.get("window_size_m") != 200 or not 1 <= report.get("maximum_records_per_side_per_window", 0) <= 12:
        raise ValueError("Unexpected local overlay window size or selection limit")
    if report.get("does_not_filter_published_data") is not True or report.get("pair_lists_saved") is not False:
        raise ValueError("Unexpected local overlay selection scope")
    fixed = [("Nairobi", [36.8219, -1.2921]), ("Kisumu", [34.7617, -0.0917]),
             ("Turkana", [35.6000, 3.1000])]
    projection = info["projection"]
    if not any(option in projection for option in ("+datum=", "+ellps=", "+R=")):
        if info.get("projection_datum") != "WGS84":
            raise ValueError("Expected an explicit WGS84 projection datum")
        projection += " +datum=WGS84"
    if report.get("projection") != projection:
        raise ValueError("Local overlay projection differs from this dataset")
    transformer = pyproj.Transformer.from_crs("EPSG:4326", projection, always_xy=True)
    regions = report.get("regions", [])
    if len(regions) != len(fixed):
        raise ValueError("Local overlay must preserve all three fixed windows")
    checked = 0
    for region, (name, center) in zip(regions, fixed):
        if region.get("name") != name or region.get("center_wgs84_lon_lat") != center:
            raise ValueError("Local overlay changed a predeclared window center")
        cx, cy = transformer.transform(*center)
        for actual, expected in ((region.get("center_projected_m", []), [cx, cy]),
                                 (region.get("window_projected_m", []), [cx - 100, cy - 100, cx + 100, cy + 100])):
            if len(actual) != len(expected) or any(not math.isfinite(value) or abs(value - target) > 1e-6
                                                  for value, target in zip(actual, expected)):
                raise ValueError("Local overlay projected center or window was moved")
        for side in ("r1", "r2"):
            sample = region[side]
            ids = sample["record_ids"]
            if sample["selected_count"] != len(ids) or len(ids) != len(sample["source_records"]) or len(ids) > report["maximum_records_per_side_per_window"]:
                raise ValueError("Inconsistent local overlay sample counts")
            if ids != sorted(set(ids)) or any(not 0 <= rid < info[f"{side}_count"] for rid in ids):
                raise ValueError("Local overlay IDs must be distinct, ascending, and in range")
            if [record["record_id"] for record in sample["source_records"]] != ids:
                raise ValueError("Local overlay source identities disagree with selected IDs")
            checked += len(ids)
    if report.get("original_polygons_checked") != checked:
        raise ValueError("Local overlay polygon accounting mismatch")
    if report.get("overlay") != "qa_local_overlay.png" or sha256(root / "qa_local_overlay.png") != report.get("overlay_sha256"):
        raise ValueError("Local overlay PNG changed or is not the validated file")
    return report


def generate(dataset_dir, source_research, builder_commit):
    root, research = Path(dataset_dir), Path(source_research)
    if not re.fullmatch(r"[0-9a-f]{40}", builder_commit):
        raise ValueError("builder-commit must be a full lowercase Git commit SHA")
    info, report, lock = (read_json(root / name) for name in
                          ("dataset.json", "validation.json", "SOURCE_LOCK.json"))
    if info["dataset"] != "kenya_buildings" or report.get("dataset") != info["dataset"]:
        raise ValueError("This script only documents the Kenya dataset")
    if report.get("passed") is not True or report.get("every_record_checked") is not True:
        raise ValueError("A successful whole-population validation is required")
    local_qa = validate_local_overlay(root, info)
    if info["group_count"] != 1 or info["quantization_scale"] != 1000:
        raise ValueError("Unexpected group or quantization contract")
    if report["coordinate_offset"] != info["coordinate_offset"]:
        raise ValueError("Validation and dataset offsets differ")
    for side in ("r1", "r2"):
        if report["whole_population"][side]["records"] != info[f"{side}_count"]:
            raise ValueError("Validation and dataset record counts differ")
    if sha256(root / "SOURCE_LOCK.json") != info["source_lock_sha256"]:
        raise ValueError("Source lock changed after construction")
    if sha256(root / "country_boundary.geojson") != lock["boundary"]["sha256"]:
        raise ValueError("Published boundary differs from source lock")
    if lock["boundary"]["sha256"] != BOUNDARY_SHA256:
        raise ValueError("Boundary differs from this release's frozen source")
    old_manifest = dict(line.split("  ", 1)[::-1]
                        for line in (root / "SHA256SUMS").read_text().splitlines())
    verified = {}
    for name in ("R1.npy", "R2.npy", "R1_metadata.parquet", "R2_metadata.parquet",
                 "dataset.json", "SOURCE_LOCK.json", "country_boundary.geojson",
                 "groups.parquet", "exclusions.jsonl.gz", "qa_samples.geojson"):
        if name not in old_manifest or not (root / name).is_file():
            raise ValueError(f"Missing checksummed dataset artifact: {name}")
        verified[name] = sha256(root / name)
        if verified[name] != old_manifest[name]:
            raise ValueError(f"Dataset artifact changed since construction: {name}")
    if not (root / "qa_overlay.png").is_file():
        raise ValueError("Missing validated polygon overlay")
    if sha256(research / "odbl-1.0.txt") != LICENSE_SHA256:
        raise ValueError("Expected the unmodified official ODbL license text")
    for name in SNAPSHOTS:
        if not (research / name).is_file() or (research / name).is_symlink():
            raise ValueError(f"Missing regular provenance snapshot: {name}")
    diagnostics = report["local_diagnostics"]
    if diagnostics["national_pairs_enumerated"] or not diagnostics["does_not_filter_published_data"]:
        raise ValueError("Unexpected local diagnostic scope")

    stats = info["filter_statistics"]
    scale = table(["Relation", "Source", "Read records", "Retained rectangles", "Excluded records"], [
        [side.upper(), label, f"{stats[f'{side}.input_records']:,}", f"{info[f'{side}_count']:,}",
         f"{stats[f'{side}.input_records'] - info[f'{side}_count']:,}"]
        for side, label in (("r1", "Microsoft Kenya"), ("r2", "Google Open Buildings V3"))])
    exclusions = table(["Reason", "R1", "R2"], [
        [reason, f"{stats.get('r1.excluded.' + reason, 0):,}", f"{stats.get('r2.excluded.' + reason, 0):,}"]
        for reason in sorted({key.split(".excluded.", 1)[1] for key in stats if ".excluded." in key})])
    windows = table(["Diagnostic window", "R1 records", "R2 records", "Positive rectangle pairs", "R1 without overlap", "R2 without overlap"], [
        [region["name"], *(f"{region[key]:,}" for key in
          ("r1_count", "r2_count", "positive_pairs", "r1_no_overlap", "r2_no_overlap"))]
        for region in diagnostics["regions"]])
    bbox = coordinate_bounds(read_json(root / "country_boundary.geojson"))
    extent = ", ".join(f"{value:.9f}" for value in bbox)
    shift = info["coordinate_offset"]
    tokens = [item.get("s2_token", Path(item["path"]).name.split("_")[0])
              for item in lock["sources"] if item["side"] == "r2"]
    readme = f"""---
license: odbl
language:
- en
---
# Kenya cross-source building rectangles

This fourth VOSMA real dataset compares Microsoft Kenya building footprints (R1)
with Google Open Buildings V3 (R2). Both sides are machine-generated detections;
neither side is human ground truth. The intended workload samples rectangle
pairs with positive intersection area, weighted by rectangle area or rectangle
IoU. Rectangle overlap is not proof of a matching real-world building, nor the
same quantity as original-polygon overlap.

## Published scale

{scale}

There is **one national group** (`group_id=0`). Download tiles do not form
separate scenes. The full national relative geometry is preserved. Actual source
and output checksums are in [SOURCE_LOCK.json](SOURCE_LOCK.json) and
[SHA256SUMS](SHA256SUMS); full construction statistics are in
[dataset.json](dataset.json).

## Frozen sources and geographic scope

- [Microsoft KenyaNigeriaBuildingFootprints]({MICROSOFT}): the fixed legacy
  Kenya release, with imagery from 2020–2021. The upstream advertised count is
  14,748,685 before this benchmark's validation and country inclusion rule.
- [Google Open Buildings V3]({GOOGLE}): inference during May 2023; the original
  polygon CSVs, not point-only records. The upstream release already excludes
  confidence below 0.65. This build applies no further confidence cutoff and
  uses unit record weights. Google tiles read: `{', '.join(tokens)}`.
- [geoBoundaries Kenya ADM0]({BOUNDARY_BASE}geoBoundaries-KEN-ADM0.geojson):
  full-resolution boundary frozen at commit `{BOUNDARY_COMMIT}`, boundary ID
  `KEN-ADM0-21065076`, representing 2020, built 2023-12-12. Longitude/latitude
  bounds (west, south, east, north): `[{extent}]`.

Google download planning uses a conservative spherical S2 level-4 covering of
the country's bounding box together with official tile geometry intersections.
Candidate `187` is absent from the official tile index and bucket, and lies
outside the actual country polygon; the available selected tiles cover the
entire frozen boundary. The additional bbox tile `19b` is read conservatively.
See the source lock and small [source records](sources/SOURCE_REFERENCES.json).
The supplied border, including its islands, defines this benchmark's study area.

## Construction and representation

1. Validate the complete original Polygon or MultiPolygon; preserve multipart
   records and holes. Invalid records are traced in `exclusions.jsonl.gz`.
2. Project every original coordinate and the fixed national boundary from
   WGS84 with `always_xy=True` into `{info['projection']}`, explicitly using
   the WGS84 datum. No simplification or edge densification is added.
3. Keep a record exactly when the projected country boundary **covers the
   centroid of the full projected original geometry**. Google CSV centroid
   columns are not the inclusion predicate. Retain complete cross-border
   footprints; do not clip them to the country.
4. Take each projected geometry's axis-aligned bounds, multiply endpoints by
   1000, and round to the nearest integer with ties to even. One coordinate
   unit is **one millimetre of quantization, not one millimetre of source accuracy**.
5. Apply the same integer translation to both complete relations:
   `dx={shift['dx']}`, `dy={shift['dy']}`. It preserves all relative positions,
   rectangle intersection areas and IoUs of the quantized representation.

There is no deduplication, confidence weighting, cross-source matching,
overlap-based selection, or algorithm-specific preprocessing. No sampler index
or pair list is included. Original duplicate records remain distinct records.
Any later size subsets must inherit the full dataset's projection, quantization
and exact common offset; they must not be repositioned independently.

`R1.npy` and `R2.npy` are little-endian int64 arrays with columns
`[record_id, group_id, x0, y0, x1, y1, weight]`; IDs start from zero independently
per side, weights equal one, and rectangles are half-open. Each aligned
`R*_metadata.parquet` preserves original source file/record identity, projected
centroid, unshifted quantized endpoints and source properties. Original record
indexes are one-based. `groups.parquet` contains the single national group.

## Exclusions

{exclusions}

Every exclusion is attributable to its original source record. No exclusion
decision uses the other relation or a benchmark result.

## Verification

Independent validation **passed**. It scanned every published record and its
aligned metadata, checked IDs, unit weights, positive bounds, source identity,
Q1000 quantization, the common shift, national centroid coverage and accounting.
Raw-polygon reprojection was independently recomputed for
{report['polygon_qa']['checked_original_polygons']:,} retained QA records.
See [validation.json](validation.json) and [qa_overlay.png](qa_overlay.png).

The additional [fixed-region overlay](qa_local_overlay.png) shows both sources'
original footprints and their final published rectangles together, at their
true relative positions, in three predeclared **200 m by 200 m windows** centered
on Nairobi, Kisumu and Turkana. Each side independently contributes at most 12
records per window, selected by the lowest published record IDs whose projected
polygon centroids fall inside. Overlap, IoU and visual appearance never affect
selection. Empty windows remain empty; centers are never relocated.
All {local_qa['original_polygons_checked']:,} selected original source records were re-read,
reprojected and checked against their aligned metadata and final NPY rectangles.
The [local overlay report](qa_local_overlay.json) records fixed centers, source
identities, array IDs and scan bounds. This is deterministic local visual QA,
not a random or representative national quality estimate. Only the plot axes
crop the view; the dataset's full geometries and rectangles remain unchanged.

The following diagnostics use only three predeclared **10 km by 10 km windows**.
Each side is selected independently by projected polygon centroid; its entire
rectangle is then used without clipping. Pairs with omitted outside-window
records are not counted. These are exact positive-pair counts for those local
subsets, **not a national join count K and not an estimate of national density
or weight distributions**. They never filter the published data.

{windows}

`validation.json` additionally records local area/IoU statistics. Local minima,
maxima and means use all local positive pairs; quantiles use all pairs up to
50,000 or a deterministic uniform-priority sample thereafter. No national pair
enumeration or pair-list artifact was produced. These diagnostics provide no
claim about which sampling algorithm will run faster.

## Reproduction

Builder commit: [`{builder_commit}`]({BUILDER}/tree/{builder_commit}).
Download this dataset directory as `kenya_buildings` next to the cloned builder,
then run from the builder checkout:

```sh
git checkout {builder_commit}
python -m pip install -r requirements.txt -r requirements-kenya.txt
python fetch_kenya_sources.py --source-lock ../kenya_buildings/SOURCE_LOCK.json --raw-root ./work/raw --log-root ./work/download-logs
python build_kenya.py --source-lock ../kenya_buildings/SOURCE_LOCK.json --raw-root ./work/raw --output-root ./work/release --work-root ./work/checkpoints
python validate_kenya.py --dataset-dir ./work/release/kenya_buildings
python qa_kenya_local_overlay.py --dataset-dir ./work/release/kenya_buildings --raw-root ./work/raw
```

The downloader verifies the frozen files rather than selecting newer upstream
versions. Construction is resumable; source, builder, geometry-library or
protocol changes invalidate checkpoints. Dependency versions used for this
release are recorded in `dataset.json`. File hashes can be checked inside this
published directory with `sha256sum -c SHA256SUMS`.

## License and attribution

The two-source derived rectangle database is provided under **ODbL 1.0**.
Microsoft's data uses ODbL; Google's dual-licensed data is used here under its
ODbL option. The boundary metadata identifies its upstream boundary as Public
Domain. See [LICENSE.txt](LICENSE.txt) for the unmodified full ODbL text and
[ATTRIBUTION.md](ATTRIBUTION.md) for source notices and boundary acknowledgement.
"""
    attribution = f"""# Attribution and license notices

This derived Kenya rectangle database contains information from
[Microsoft KenyaNigeriaBuildingFootprints]({MICROSOFT}) and
[Google Research Open Buildings V3]({GOOGLE}), made available here under the
[Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
Microsoft supplies its source under ODbL. Google offers CC BY 4.0 or ODbL 1.0;
this release explicitly selects the ODbL option. Source copyright and database
rights notices are retained in the supplied provenance and source links.

The modifications are country inclusion by projected original-polygon centroid,
validity exclusions, a fixed WGS84 LAEA projection, projected axis-aligned
bounding boxes, Q1000 endpoint quantization, a common integer translation and
row-aligned provenance metadata. Original building polygons are not matched,
repaired, simplified, confidence-filtered or deduplicated.

Administrative boundaries courtesy of [geoBoundaries.org](https://www.geoboundaries.org/);
source **RCMRD GeoPortal**, Kenya ADM0, representative year 2020, frozen commit
`{BOUNDARY_COMMIT}`. Its included upstream metadata specifies **Public Domain**.
Boundary source acknowledgement: Runfola, D. et al. (2020),
*geoBoundaries: A global database of political administrative boundaries*,
PLoS ONE 15(4): e0231866, https://doi.org/10.1371/journal.pone.0231866.
The fixed boundary is an operational study area and does not adjudicate borders.

The unmodified official ODbL 1.0 text in `LICENSE.txt` was downloaded from
{LICENSE_URL}; SHA-256 `{LICENSE_SHA256}`.
The ODbL database license does not change the builder repository's code license.
"""
    sources = root / "sources"
    if sources.is_symlink():
        raise ValueError("sources directory must not be a symlink")
    sources.mkdir(exist_ok=True)
    references = {"snapshots": []}
    for name, url in SNAPSHOTS.items():
        destination = sources / name
        if destination.is_symlink():
            raise ValueError(f"Refusing a symlink destination: {name}")
        shutil.copyfile(research / name, destination)
        references["snapshots"].append({"file": name, "source_url": url, "sha256": sha256(destination)})
    references["notes"] = ["google-extra-tiles.json combines HTTP/GCS response metadata and derived S2 vertices.",
                           "SOURCE_LOCK.json is authoritative for complete source URLs, immutable object generations and SHA-256 values.",
                           "No large upstream building archives are duplicated in sources/."]
    write_text(sources / "SOURCE_REFERENCES.json", json.dumps(references, indent=2) + "\n")
    write_text(root / "README.md", readme)
    write_text(root / "ATTRIBUTION.md", attribution)
    shutil.copyfile(research / "odbl-1.0.txt", root / "LICENSE.txt")
    manifests = []
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Release cannot contain symlinks: {file.name}")
        if not file.is_file() or file.name == "SHA256SUMS":
            continue
        relative = file.relative_to(root).as_posix()
        if file.name.endswith((".tmp", ".part")) or "\n" in relative or "\r" in relative:
            raise ValueError(f"Unfinished or invalid release file: {relative!r}")
        digest = verified.get(relative) or sha256(file)
        if relative in ("R1.npy", "R2.npy"):
            if digest != report["whole_population"][relative[:2].lower()]["npy_sha256"]:
                raise ValueError(f"Published array changed after validation: {relative}")
        manifests.append(f"{digest}  {relative}\n")
    write_text(root / "SHA256SUMS", "".join(manifests))
    return {"dataset": info["dataset"], "documented_files": len(manifests),
            "builder_commit": builder_commit, "validation_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--source-research", type=Path, required=True)
    parser.add_argument("--builder-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.dataset_dir, args.source_research, args.builder_commit)))


if __name__ == "__main__":
    main()
