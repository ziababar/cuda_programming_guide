# Agent Instructions

This repository contains a comprehensive CUDA programming guide.

## Code Organization

- **Documentation**: Located in numbered chapter directories (e.g., `01_execution_model/`, `04_streams_concurrency/`).
- **Source Code**: Complex C++ implementations and reusable components are extracted into the `src/` directory, mirroring the chapter structure.
  - Example: Code for `04_streams_concurrency` is in `src/04_streams_concurrency/`.
  - Header files (`.cuh`) are used for CUDA C++ code to facilitate inclusion and syntax highlighting.

## Coding Standards

- **Modern CUDA**: Prefer modern CUDA features (e.g., `cuda::barrier`, `__shfl_sync`) over legacy ones.
- **Error Handling**: All CUDA API calls should be checked for errors.
- **Self-Contained**: Code snippets in documentation should be illustrative. Full implementations in `src/` should be compilation-ready (include necessary headers).
- **SoA vs AoS**: Prefer Structure-of-Arrays (SoA) for better memory coalescing.

## Maintenance

- When modifying documentation that references code, ensure the corresponding code in `src/` is updated.
- When adding new complex examples, extract the implementation to `src/` and reference it in the markdown.
