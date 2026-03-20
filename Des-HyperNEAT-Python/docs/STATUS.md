# Port Status

## Current Position

The package is usable as a standalone DES-HyperNEAT core project.

That means:
- the core is separated into its own repository-shaped package
- external code can import `deshyperneat`
- the package is no longer only a copy on disk

## What Is Already Done

- Core package split from project-specific code
- Standalone package path established at `Des-HyperNEAT-Python/deshyperneat`
- Legacy tree separated out of the active path
- Standalone package skeleton created
- Core modules reorganized around:
  - `genome`
  - `developer`
  - `search`
  - `substrate`
  - `network`
  - `state`
  - `population`
  - `reporting`
  - `evolution`

## What Is Not Fully Closed Yet

- Upstream Rust source-equivalence is not fully guaranteed
- Execution validation has not been completed
- Training validation has not been completed
- Some core semantics may still be adjusted to match upstream more closely

## Recommended Repository Label

Use one of:
- `alpha`
- `experimental`
- `work in progress`

Avoid:
- `stable`
- `production ready`
- `fully complete port`

## Short Truthful Summary

This is a serious standalone Python DES-HyperNEAT project candidate, but it should still be presented as an in-progress alpha rather than a finished release.
