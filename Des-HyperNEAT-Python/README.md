# Des-HyperNEAT-Python

Python port of the DES-HyperNEAT core.

## Status

This repository is not a final, verified release yet.

Current state:
- The package has been split out as a standalone project.
- The core is separated from project-specific topology and adapter code.
- The repository is suitable to publish as an `alpha` or `work-in-progress` project.

Not finished yet:
- Full source-equivalence with upstream Rust DES-HyperNEAT is not guaranteed.
- Runtime verification and training validation have not been completed.
- Some naming and structural cleanup may still continue.

If you push this now, describe it as:
- `Python port / experimental project`
- not `stable release`
- not `fully verified upstream-equivalent`

## Scope

Included:
- `deshyperneat/`
  - genome
  - developer
  - search
  - substrate
  - network
  - state
  - population
  - reporting
  - evolution
  - executor
- `configs/deshyperneat_cppn.ini`

Excluded:
- Matgo-specific topology/control code
- game runtime adapters
- project-specific training assets outside the generic core

## Install

```bash
pip install -e .
```

## Package Layout

```text
Des-HyperNEAT-Python/
  deshyperneat/
  configs/
  pyproject.toml
  README.md
  docs/STATUS.md
```

## Entry Points

Main module:
- `deshyperneat.mod`

Main objects:
- `Deshyperneat`
- `Genome`
- `Developer`
- `Executor`

Typical imports:

```python
from deshyperneat import Config, SearchConfig
from deshyperneat.mod import Deshyperneat, Developer, compile_executor
from deshyperneat.environment import EnvironmentDescription
```

## Configuration

Example config:
- `configs/deshyperneat_cppn.ini`

The active Matgo training path currently points to this config file.

## Notes

This repository is intended to become the standalone home of the Python DES-HyperNEAT core.

At the moment, the correct description is:
- standalone
- usable as a Python project
- still under heavy porting and cleanup

See:
- `docs/STATUS.md`
