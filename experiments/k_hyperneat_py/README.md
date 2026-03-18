# K-HyperNEAT

`K-HyperNEAT` is the Python core workspace for a HyperNEAT / ES-HyperNEAT /
DES-HyperNEAT style engine.

This folder is intentionally kept generic so it can be forked publicly without
project-specific game logic.

## Source direction

This project is informed by:

- `tenstad/des-hyperneat`
  - core DES idea
  - multi-substrate developer flow
  - node/link CPPN separation
- `ukuleleplayer/pureples`
  - pure-Python substrate and CPPN adapter flow
  - simple HyperNEAT phenotype creation path

This is not a line-by-line port. The current implementation keeps the module
boundaries close to those projects, but starts with a smaller Python-first
architecture.

## Current status

Implemented here:

- package/layout for a separate `K-HyperNEAT` experiment
- substrate and topology dataclasses
- CPPN adapter protocol
- static multi-substrate phenotype builder
- ES-HyperNEAT style local quadtree search core
- substrate action ordering for topological development
- reachable-path pruning between input/output nodes
- hidden/output substrate growth through node/link search
- Python phenotype executor compiler/runtime
- executor JSON export base (`k_hyperneat_executor_v1`)
- generic runtime/export surface for downstream adapters

Not implemented yet:

- coevolutional variants
- fully faithful upstream DES reverse/output growth
- downstream game adapters and training runners

## Install

Core only:

```powershell
pip install -e experiments/k_hyperneat_py
```

If you want CPPN evolution backed by `neat-python 2.x`:

```powershell
pip install -e "experiments/k_hyperneat_py[evolve]"
```

Scope note:

- the core package does not include a population/evolution loop by itself
- current CPPN evolution is expected to be supplied by an external backend
- in this repo, that backend is `neat-python 2.x`

## Quick smoke

Executor-only smoke:

```powershell
python experiments/k_hyperneat_py/smoke_executor.py
```

Generic developer/search smoke:

```powershell
python experiments/k_hyperneat_py/smoke_developer.py
```

## Planned module map

- `k_hyperneat/config.py`
  - experiment and search configs
- `k_hyperneat/coordinates.py`
  - geometry primitives
- `k_hyperneat/substrate.py`
  - substrate and topology definitions
- `k_hyperneat/cppn.py`
  - CPPN protocol and query helpers
- `k_hyperneat/network.py`
  - phenotype graph container
- `k_hyperneat/search.py`
  - ES/DES substrate search hooks
- `k_hyperneat/developer.py`
  - substrate expansion and phenotype assembly

## Next work

1. Tighten reverse/output growth to match upstream DES more faithfully
2. Keep the public core free of project-specific game adapters
3. Add a generic example topology that does not depend on local game code
4. Stabilize CPPN/runtime export contracts for downstream forks
