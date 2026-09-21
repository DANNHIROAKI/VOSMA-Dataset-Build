# Kenya building-footprint relations

This fourth dataset compares Microsoft Kenya Building Footprints with Google
Open Buildings V3. Both relations contain machine-generated building polygons;
neither is designated as ground truth. The output is a rectangle sampling
workload, not a set of matched buildings or original-polygon intersection pairs.

## Reproduce a published release

Use a Linux server with Git, curl, Python 3.12, and `sha256sum`. Choose a working
directory with room for approximately 10.3 GB of compressed upstream sources,
plus the checkpoint files, final arrays, metadata, and validation outputs. The
commands below keep the checkout, virtual environment, downloads, caches,
temporary files, logs, and generated release inside one experiment directory.

First clone the builder and select the immutable 40-character builder commit
linked from the published Kenya dataset's README. Replace the placeholder below
with that actual commit; do not substitute a moving branch name when reproducing
an existing release.

```sh
export EXPERIMENT_ROOT="$PWD/kenya-reproduction"
mkdir -p "$EXPERIMENT_ROOT"
git clone https://github.com/DANNHIROAKI/VOSMA-Dataset-Build.git \
  "$EXPERIMENT_ROOT/builder"
cd "$EXPERIMENT_ROOT/builder"

KENYA_BUILDER_COMMIT='<published-40-character-builder-commit>'
git checkout --detach "$KENYA_BUILDER_COMMIT"
KENYA_BUILDER_COMMIT="$(git rev-parse HEAD)"

export TMPDIR="$EXPERIMENT_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export XDG_CACHE_HOME="$EXPERIMENT_ROOT/cache"
export MPLCONFIGDIR="$EXPERIMENT_ROOT/cache/matplotlib"
export PIP_CACHE_DIR="$EXPERIMENT_ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$EXPERIMENT_ROOT/cache/python"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" \
  "$PYTHONPYCACHEPREFIX" "$EXPERIMENT_ROOT/logs"

python3.12 -m venv "$EXPERIMENT_ROOT/venv"
. "$EXPERIMENT_ROOT/venv/bin/activate"
python -m pip install -r requirements.txt -r requirements-kenya.txt
```

The two requirements files contain runtime dependencies. For development tests,
install pytest separately and run the repository's tests before construction:

```sh
python -m pip install pytest
python -m pytest -q
```

Download the exact archives and boundary listed in the committed,
hash-complete `kenya-sources.lock.json`. The downloader resumes partial files and
checks their frozen content hashes. Ordinary reproduction must use this lock as
provided: **do not use `--freeze-lock`**, regenerate the tile selection, or replace
the boundary with a current release.

```sh
python fetch_kenya_sources.py \
  --source-lock kenya-sources.lock.json \
  --raw-root "$EXPERIMENT_ROOT/raw" \
  --log-root "$EXPERIMENT_ROOT/logs/downloads" \
  --workers 4 \
  > "$EXPERIMENT_ROOT/logs/fetch.jsonl" 2>&1
```

After every download passes verification, build and validate the fourth dataset.
The builder appends `kenya_buildings` to `--output-root`; no other dataset is
rebuilt by this entry point. Keep the same command and checkpoint directory to
resume an interrupted build.

```sh
python build_kenya.py \
  --raw-root "$EXPERIMENT_ROOT/raw" \
  --source-lock kenya-sources.lock.json \
  --output-root "$EXPERIMENT_ROOT/release" \
  --work-root "$EXPERIMENT_ROOT/checkpoints" \
  --workers 4 \
  > "$EXPERIMENT_ROOT/logs/build.jsonl" 2>&1

KENYA_DATASET_DIR="$EXPERIMENT_ROOT/release/kenya_buildings"
python validate_kenya.py --dataset-dir "$KENYA_DATASET_DIR" \
  > "$EXPERIMENT_ROOT/logs/validation.jsonl" 2>&1
```

Validation scans every retained record and its aligned metadata, checks the
common translation and country inclusion, verifies source identities, and
recomputes the sampled raw-polygon projections. It writes `validation.json` and
`qa_overlay.png`. It also computes explanatory overlap statistics in the fixed
small diagnostic windows; those statistics never filter the published relations
and are not national overlap estimates. A successful construction alone is not
a substitute for this full validation step.

Independently re-read original polygons for the fixed local visual checks. This
can run in parallel with full validation after construction has completed; both
checks must finish successfully before publication documentation is generated.

```sh
python qa_kenya_local_overlay.py \
  --dataset-dir "$KENYA_DATASET_DIR" \
  --raw-root "$EXPERIMENT_ROOT/raw" \
  > "$EXPERIMENT_ROOT/logs/local-overlay.jsonl" 2>&1
```

This creates `qa_local_overlay.png` and `qa_local_overlay.json`. Three fixed
200 m square windows retain the predeclared Nairobi, Kisumu and Turkana centers.
For each side and window, select at most 12 records by ascending published ID,
requiring the projected polygon centroid to lie inside. No matching, IoU or
appearance criterion is used. Empty windows are retained without repositioning.
The original source polygons and final rectangles are independently rechecked
and overlaid in their true relative positions. Each needed raw file is streamed
only through its greatest selected source-record index. These illustrations are
local QA samples, not population-quality estimates or benchmark selection rules.

Finally generate the publication README, attribution, license, provenance
snapshots, and updated complete file-checksum manifest. The small source
snapshots required by this command are committed under
`docs/kenya/source-research`; there is no need to fetch live replacement metadata.

```sh
python write_kenya_release_docs.py \
  --dataset-dir "$KENYA_DATASET_DIR" \
  --source-research docs/kenya/source-research \
  --builder-commit "$KENYA_BUILDER_COMMIT" \
  > "$EXPERIMENT_ROOT/logs/release-docs.jsonl" 2>&1

(
  cd "$KENYA_DATASET_DIR"
  sha256sum --check SHA256SUMS
)
```

The resulting `release/kenya_buildings` directory is the publication payload.
Publish it as the new `kenya_buildings/` directory within the existing dataset
repository, preserving the first three dataset directories. Upstream archives,
checkpoint shards, the virtual environment, caches, and operational logs are
construction artifacts and are not part of the dataset payload. The dataset
release includes its ODbL 1.0 license and both providers' attribution; this does
not change the code repository's license.

The builder verifies its inputs again before processing and never silently
overwrites an existing dataset. Completed output can be reused only after
checking its manifest. Keep the frozen lock and build code unchanged during a
resume. Checkpoints can be archived or removed after successful validation and
publication, but retaining them makes an interrupted publication or local
verification easier to recover.

## Frozen source lock

`kenya-sources.lock.json` is a UTF-8 JSON object with these fields:

```json
{
  "dataset": "kenya_buildings",
  "schema_version": "vosma-kenya-sources-v1",
  "boundary": {
    "path": "boundary/geoBoundaries-KEN-ADM0.geojson",
    "url": "https://official-source.example/frozen-boundary.geojson",
    "sha256": "64 lowercase hexadecimal characters",
    "license": "the boundary source license"
  },
  "sources": [
    {
      "side": "r1",
      "format": "geojsonl.zip",
      "path": "microsoft/Kenya.geojsonl.zip",
      "member": "Kenya.geojsonl",
      "url": "https://official-source.example/Kenya.geojsonl.zip",
      "sha256": "64 lowercase hexadecimal characters",
      "expected_records": 14748685
    },
    {
      "side": "r2",
      "format": "csv.gz",
      "path": "google/frozen-s2-token_buildings.csv.gz",
      "geometry_column": "geometry",
      "url": "https://official-source.example/frozen-s2-token_buildings.csv.gz",
      "sha256": "64 lowercase hexadecimal characters"
    }
  ]
}
```

This is a schema illustration, not a usable lock. The committed reproduction lock
contains the actual source URLs and hashes for every input. The frozen Google
download list consists of these **11 available Level 4 tiles**, in this order:
`171, 177, 179, 17b, 17d, 17f, 181, 183, 185, 19b, 19d`.

The download list is the union of the spherical S2 Level 4 cover of the fixed
country's bounding box and the official tile-index geometries intersecting the
country, restricted to published objects. This is a conservative geographic
cover. Tile `19b` is downloaded even though it is only needed by the bounding-box
cover. Candidate `187` is absent from the official index and storage objects;
the archived index, object-query results, and spherical-edge check show that
it lies outside the actual frozen country polygon. Its absence is not evidence
of an officially declared zero-detection cell. The available selected tiles cover
the complete frozen national boundary.

S2 files are download partitions, **never independent scenes or groups**. Do not
translate, pack, or otherwise reposition individual tiles. The source boundary,
file list, versions, and input order were fixed before cross-source overlap
analysis or algorithm execution. Additional provenance fields such as upstream
release, source license, byte size, immutable object generation, ETag, and
retrieval time are preserved in the released `SOURCE_LOCK.json`.

## Geometry and identity contract

1. Read every source record in lock order, then original file order. Verify all
   source SHA-256 values before processing. A source path must not occur twice.
2. Parse and validate each full Polygon or MultiPolygon. Invalid, empty,
   unsupported, nonfinite, or quantization-degenerate geometry is excluded with
   an individually traceable reason. Do not repair, simplify, clip, deduplicate,
   split multipart buildings, filter by confidence, or match across sources.
3. Project the full geometry and the fixed national boundary from WGS84 to
   `+proj=laea +lat_0=0 +lon_0=37.5 +datum=WGS84 +x_0=0 +y_0=0 +units=m +no_defs`, explicitly
   using the WGS84 datum and `always_xy=True`. Coordinate sequences are projected
   as provided; no edge densification is added.
4. Retain a building when the projected boundary covers the centroid of its full
   projected polygon. Polygon holes and every part of a MultiPolygon contribute
   to the centroid. A retained cross-border footprint remains complete.
5. Take the bounds of the projected full geometry, multiply each endpoint by
   1000, and round to the nearest integer with ties to even (`numpy.rint` on
   projected binary64 values). The unit is one millimetre.
6. Compute minima across both complete retained relations, then use one shared
   integer offset `(1-min_x, 1-min_y)` for every rectangle. This preserves all
   relative positions, intersection areas, and IoUs in the quantized rectangle
   representation. Keep a single national `group_id=0`.

`R1.npy` and `R2.npy` are little-endian int64 arrays with columns
`[record_id, group_id, x0, y0, x1, y1, weight]`, and unit weights. Record IDs start
at zero independently for each relation. Rectangles are half-open. Metadata
Parquet files are in exactly the same order. Their `local_x0` through `local_y1`
are quantized endpoints before translation; `source_bbox` is the full projected
polygon bounds in metres. `attributes` preserves source properties, projected
centroid, original degree bounds, geometry type, and physical source line number.
`source_record_index` is one-based: original JSONL line for Microsoft, data row
excluding the header for Google CSV. Source file plus record index identifies a
record even when annotation IDs or geometry repeat.

## Bounded-memory construction and resume

The builder parses bounded batches (default 8192 records), performs vectorized
geometry operations, and commits a shard after each 250,000 original input
records. Each shard contains raw int64 rectangle endpoints, batched Parquet
metadata, compressed exclusions, a few QA geometries, and a SHA-256 checkpoint.
No list of all polygons, metadata records, or cross-source pairs is built.
Final arrays use NumPy memory maps; metadata is merged in bounded Arrow batches.
`--workers` schedules independent input files in separate processes; final rows
still follow the frozen source-lock order. Changing worker count does not change
the protocol or output bytes and does not invalidate checkpoints. This concerns
offline dataset construction only, not benchmark algorithm execution. Compressed
exclusion logs use a fixed zero gzip timestamp for reproducible output bytes.

Reusing the same command verifies and resumes completed shards. A truncated
in-progress shard is rebuilt. Compressed input may need to be streamed past the
last committed record, but complete source files are skipped. Changes to source
lock, builder code, geometry-library versions, projection, quantization, or batch
settings require a different checkpoint directory. They cannot silently reuse
old processed records. Progress and exclusion counts are emitted as JSON lines.

The release contains a source lock, fixed boundary, row-aligned metadata, group
metadata, exact filter statistics, small raw-polygon QA samples, and final file
checksums. Descriptive overlap diagnostics must not alter this construction or
select an algorithm branch. Scale subsets, if created later, must inherit the
full dataset's exact projection, quantization, shared offset, and source identity.
