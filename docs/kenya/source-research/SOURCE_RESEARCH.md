# Kenya cross-source building dataset: pinned source research

This directory contains small public upstream metadata and a prospective download lock, not building records. `source-lock.research.json` is intentionally marked incomplete until every full download has a computed SHA-256; never present null hashes as a completed release lock.

## Microsoft side

- Official project: https://github.com/microsoft/KenyaNigeriaBuildingFootprints
- Pinned README: https://github.com/microsoft/KenyaNigeriaBuildingFootprints/blob/13adff8da80c4980f00f586459dcef4e7ba438c8/README.md
- Official legacy download: https://minedbuildings.z5.web.core.windows.net/legacy/africa/kenya.geojsonl.zip
- Official declared unfiltered count: 14,748,685. HTTP size: 1,046,134,828 bytes; ETag `0x8DCFF4EE129ABDA`.
- Building extraction uses 2020–2021 Maxar imagery. The blob Last-Modified timestamp is 2024-11-07 and must not be presented as imagery capture or inference date.
- License: Open Data Commons Open Database License (ODbL) 1.0. Derived rectangles should retain attribution and be distributed under ODbL 1.0.

## Google side

- Official documentation: https://sites.research.google/open-buildings/ (redirects to https://sites.research.google/gr/open-buildings/).
- Product: Open Buildings v3 polygon CSVs; inference during May 2023. Official released rows already have model confidence >= 0.65. Apply no additional confidence filtering and no confidence weight.
- Official level-4 tile index: https://openbuildings-public-dot-gweb-research.uw.r.appspot.com/public/tiles.geojson
- Frozen index: 333 entries, 250,366 bytes, SHA-256 `f63227f3da41a40381d16b5bf6a9a1101a3a462650a69ff3049c6d775e6f1e77`.
- Conservative download set, in deterministic token order: `171, 177, 179, 17b, 17d, 17f, 181, 183, 185, 19b, 19d`; 9,285,829,234 bytes. URLs in `google-tile-downloads.json` pin each immutable GCS generation and record provider MD5 and CRC32C.
- Selection uses the union of true spherical S2 level-4 bounding-box cover and all tile GeoJSON geometries intersecting the country. The bounding-box cover is a deliberate geographic superset, not an empirically selected performance region. The bbox cover also contains `187`, which has no official index entry, returns HTTP 404 at both the public object and GCS object metadata endpoints, and yields an empty GCS prefix listing. Evidence is retained in `google-extra-tiles.json` and `google-187-prefix-list.json`.
- Tile 187 lies southeast of the actual Kenya polygon; a densified spherical-edge check is disjoint from the boundary with 0.16438119518431188-degree clearance. The union of the ten directly intersecting official tile polygons already covers the complete boundary with no uncovered area and an approximately 18 km outer-edge margin; 19b is downloaded additionally. This does not claim the unprovided tile is a formally declared zero-detection cell.
- Google permits either CC BY 4.0 or ODbL 1.0. Select ODbL 1.0 for this two-source derived dataset; retain Google Research attribution.

## Country boundary

- Source: geoBoundaries gbOpen Kenya ADM0, full resolution (not simplified); upstream RCMRD GeoPortal.
- Pinned Git commit: `9469f09592ced973a3448cf66b6100b741b64c0d`. This is a commit-pinned build, not a claim that these exact bytes are the v6.0.0 tag.
- Boundary ID: `KEN-ADM0-21065076`; year represented 2020; source update 2023-01-19; build 2023-12-12.
- Download: https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09592ced973a3448cf66b6100b741b64c0d/releaseData/gbOpen/KEN/ADM0/geoBoundaries-KEN-ADM0.geojson
- Full file: 1,333,814 bytes, SHA-256 `89248c71d10e86d76a2ca336aa2444aac9901651c9cdb974a83f29fbb72c55fa`.
- Valid MultiPolygon, 114 component polygons, 30,581 vertices. Bounds in longitude/latitude: `(33.91181945760195, -4.702270507977516, 41.90625762938612, 5.430648327058691)`.
- Metadata declares Public Domain. geoBoundaries requests acknowledgement; include “Administrative boundaries courtesy of geoBoundaries.org; source RCMRD GeoPortal” and/or Runfola et al. (2020), https://doi.org/10.1371/journal.pone.0231866.
- Preserve the supplied polygon exactly, including islands, lakes/coastal choices and disputed-area representation; do not claim it is a political adjudication or mix borders from other sources. geoBoundaries single-country products may reflect national boundary representations; the fixed shape defines this benchmark's operational study area.

## Row inclusion rule (RealDataset.md chapter 4)

For BOTH sides, project the original polygon and the fixed country boundary to the single fixed CRS, compute the centroid of the projected original polygon, and retain it exactly when the projected boundary covers that centroid. Keep the full original building polygon for projected axis-aligned bounds. Do not use the Google CSV centroid fields for filtering; do not use polygon-intersects-country as the row rule; do not clip buildings to the border. Tile intersection is only download planning and never the inclusion predicate.

The official Google downloader notebook is useful for download conventions but is NOT copied as this benchmark's region rule: its current version uses level-6 shards and raw longitude/latitude centroid-within filtering. Frozen notebook source commit: `ae3833b9fc4730c383e900aa7dcfffad2f21a064`, path `building_detection/open_buildings_download_region_polygons.ipynb`.
