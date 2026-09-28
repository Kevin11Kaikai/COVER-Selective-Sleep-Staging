# Source attribution

This repository extracts the COVER implementations and saved numerical evidence associated with the manuscript *COVER: Learning to Accept More in Selective Sleep Staging*. The code authors and manuscript authors are identified in the source files and `CITATION.cff`; existing source headers are retained.

The research project's GNU General Public License, version 3, is carried forward unchanged in `LICENSE`. Historical EEG-Conformer/component sources are retained under `archive/source_code/training_dependencies/`, and the corresponding staging and DuoF implementations are under the adjacent directories. Import origins and file hashes are recorded in `provenance/import_manifest.json`.

Sleep-EDF identities in the frozen arrays refer to the public dataset's pseudonymous record identifiers. The repository contains derived scores, errors, masks, and aggregate counts; it does not redistribute raw EEG recordings. The code license does not replace the terms of any external dataset or dependency.

Portability changes are limited to input/output paths, an isolated synthetic checkpoint-test fixture, and new release/reproduction scripts and documentation. The imported numerical algorithms and frozen numerical input files are preserved.
