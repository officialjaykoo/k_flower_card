# Porting Map

This file maps the upstream source structure to the current Python workspace.

## Upstream: `tenstad/des-hyperneat`

- `src/deshyperneat/developer.rs`
  - target: `k_hyperneat/developer.py`
  - status: topological substrate actions, growth, and pruning ported in simplified form
- `src/deshyperneat/desgenome.rs`
  - target: `k_hyperneat/developer.py` (`DesGenomeProtocol`)
  - status: protocol defined
- `src/eshyperneat/search.rs`
  - target: `k_hyperneat/search.py`
  - status: initial Python port implemented, including iterative substrate exploration
- `src/hyperneat/substrate.rs`
  - target: `k_hyperneat/substrate.py`
  - status: initial dataclasses done
- `network/src/*`
  - target: `k_hyperneat/network.py`
  - status: graph container plus reachable pruning ported
- `network/src/execute.rs`
  - target: `k_hyperneat/executor.py`
  - status: Python executor/compiler port implemented
- `runtime bridge`
  - target: `src/ai/kHyperneatExecutor.js`
  - status: JSON runtime loader + forward-pass bridge implemented
- `project-specific runtime adapters`
  - target: downstream/local integration layer
  - status: moved out of the public core package

## Upstream: `ukuleleplayer/pureples`

- `pureples/shared/substrate.py`
  - used as the baseline for Python substrate representation
- `pureples/shared/create_cppn.py`
  - used as the baseline for Python CPPN adapter flow
- `pureples/hyperneat/hyperneat.py`
  - used as the baseline for static coordinate-to-coordinate CPPN querying

## K-HyperNEAT stages

### Stage 0
- Separate experiment package
- Static multi-substrate topology
- Direct CPPN edge query
- ES-style local substrate search

### Stage 1
- Add discovered hidden substrate points
- Add substrate-local depth expansion
- Add reachable pruning

### Stage 2
- Port DES node/link CPPN split more faithfully
- Expand reverse/output growth cases
- Add Node runtime bridge for exported executor JSON

### Stage 3
- Publish a stable generic executor/runtime surface
- Let downstream projects attach duel/eval/training harnesses

## Current bridge note

`k_hyperneat_executor_v1` now has:

- Python export path
- JS executor runtime

Project-specific Matgo integration and phase runner wiring now live under
`experiments/k_hyperneat_matgo/`.
