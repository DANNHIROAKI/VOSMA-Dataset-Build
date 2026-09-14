# DocLayNet — pending original repeated annotations

No DocLayNet runtime arrays are released in v0.1.0. The source audit found:

| Split | Images | Annotation records | Precedence 0 | Precedence 1 or 2 |
|---|---:|---:|---:|---:|
| Train | 69,375 | 941,123 | 941,123 | 0 |
| Validation | 6,489 | 99,816 | 99,816 | 0 |
| Test | 4,999 | 66,531 | 66,531 | 0 |
| Total | 80,863 | 1,107,470 | 1,107,470 | 0 |

These are observed source counts, not the size of a constructed dual-relation dataset. The requested physical-page pairing of original annotation precedence 0 and 1 cannot be constructed from this snapshot.

The official Core archive at `dax-doclaynet/1.0.0/DocLayNet_core.zip` has ETag `52242220284e12538949c5b06eb2777c-448`, size 30,012,083,650 bytes, and last modification 2022-08-02. All three COCO members were fully extracted and checked against ZIP CRC32; their SHA-256 values are in `sources/doclaynet.json`. ETags are archive identifiers, not asserted SHA-256 values; the full image archive was not downloaded.

The official public object catalog under `dax-doclaynet/` lists only the Core and Extra archives and their directory entries. The Core contains no additional annotation JSON. The Extra files are PDF-derived text cells, not independent human layout annotations: see the [maintainer's explanation](https://github.com/DS4SD/DocLayNet/issues/15#issuecomment-1400120626). The [reported absence of nonzero precedence](https://github.com/DS4SD/DocLayNet/issues/5) remains unresolved. The official [v1.1 schema](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1/blob/5e89392376049f4d589ea339ed64468310ed5c3f/README.md) also omits annotation precedence and cannot recover the missing layers.

The construction code supports annotation-level precedence and is tested on small explicit multilayer inputs. On the pinned official snapshot it deliberately raises `No original precedence 0/1 pages present; refuse substitute data`. It has not been validated against an unavailable full repeated-annotation source. Completing this portion requires the original precedence-1 annotations, with verifiable page identities and provenance. No cloned, perturbed, model-generated, or PDF-cell rectangles have been substituted.

Source license: CDLA-Permissive-1.0. Credit Pfitzmann et al., [DocLayNet (2022)](https://arxiv.org/abs/2206.01062).
