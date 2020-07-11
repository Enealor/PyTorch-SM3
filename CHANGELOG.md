# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Types of changes
- `Added` for new features.
- `Changed` for changes in existing functionality.
- `Deprecated` for soon-to-be removed features.
- `Removed` for now removed features.
- `Fixed` for any bug fixes.
- `Security` in case of vulnerabilities.

## [0.1.0] - 2020-07-11
### Changed
- Updated optimizer to use new PyTorch function signatures as the previous
signatures were being deprecated. This means prior releases of PyTorch may not
work with this optimizer.
- Updated repository structure for packaging. The code for the optimizer has
been moved to `src/`. The code for testing the optimizer against NumPy has been
moved to `tests/`.
- Updated GitHub workflow to create `pip` packaging and test

## [0.0.0] - 2020-02-04
### Added
- Created `PyTorch-SM3` repository and finalized initial code.
