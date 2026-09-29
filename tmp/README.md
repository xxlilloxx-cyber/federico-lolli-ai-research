# Recoverable migration archive

This directory is not a deletion mechanism. It stores tracked, non-private
files that are no longer canonical but remain useful for provenance. Every
archived file is listed in `MANIFEST.csv` with its original path and pre-move
SHA-256 hash.

Private raw data, checkpoints, ZIP packages, and local previews are not moved
here; they remain ignored in their original locations. Nothing in `tmp/` is
part of the GitHub Pages payload.
