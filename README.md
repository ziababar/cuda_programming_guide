# CUDA Programming Guide

A comprehensive, hands-on guide to CUDA programming covering everything from basic concepts to advanced optimization techniques. This guide is structured as a practical reference for developers at all levels.

## Table of Contents

### Quick Start - [`00_quick_start/`](00_quick_start/)
- **[CUDA Cheat Sheet](00_quick_start/0_cuda_cheat_sheet.md)** - Essential commands, concepts, and quick references

### Execution Model - [`01_execution_model/`](01_execution_model/)
- **[Execution Model Overview](01_execution_model/1_cuda_execution_model_overview.md)** - High-level concepts and navigation
- **[Thread Hierarchy Complete](01_execution_model/1a_thread_hierarchy_complete.md)** - Thread/Block/Grid organization and indexing
- **[Warp Execution Advanced](01_execution_model/1b_warp_execution_advanced.md)** - SIMT model, divergence, and optimization
- **[Streaming Multiprocessors Deep Dive](01_execution_model/1c_streaming_multiprocessors_deep.md)** - SM architecture and occupancy
- **[Synchronization Complete](01_execution_model/1d_synchronization_complete.md)** - Thread barriers and cooperative groups
- **[Execution Constraints Guide](01_execution_model/1e_execution_constraints_guide.md)** - Hardware limits and best practices

### Memory Hierarchy - [`02_memory_hierarchy/`](02_memory_hierarchy/)
- **[Memory Hierarchy Overview](02_memory_hierarchy/2_cuda_memory_hierarchy_overview.md)** - Memory types and access patterns
- **[Global Memory Advanced](02_memory_hierarchy/2b_global_memory_advanced.md)** - Coalescing and optimization strategies
- **[Shared Memory Complete](02_memory_hierarchy/2c_shared_memory_complete.md)** - Bank conflicts and performance tuning
- **[Constant Memory Complete](02_memory_hierarchy/2d_constant_memory_complete.md)** - Broadcast patterns and use cases
- **[Unified Memory Complete](02_memory_hierarchy/2e_unified_memory_complete.md)** - Advanced techniques and multi-GPU
- **[Memory Debugging Complete](02_memory_hierarchy/2f_memory_debugging_complete.md)** - Profiling and troubleshooting

### Streams & Concurrency - [`03_streams_concurrency/`](03_streams_concurrency/)
- **[CUDA Streams Overview](03_streams_concurrency/3_cuda_streams_overview.md)** - Asynchronous execution concepts
- **[CUDA Streams Concurrency](03_streams_concurrency/3_cuda_streams_concurrency.md)** - Advanced patterns and optimization

### Performance & Profiling - [`04_performance_profiling/`](04_performance_profiling/)
- **[CUDA Profiling](04_performance_profiling/4_cuda_profiling.md)** - Tools and techniques for performance analysis

## Learning Path Recommendations

### **Beginner Path**
1. Start with [CUDA Cheat Sheet](00_quick_start/0_cuda_cheat_sheet.md) for essential concepts
2. Read [Execution Model Overview](01_execution_model/1_cuda_execution_model_overview.md)
3. Study [Thread Hierarchy Complete](01_execution_model/1a_thread_hierarchy_complete.md)
4. Learn [Memory Hierarchy Overview](02_memory_hierarchy/2_cuda_memory_hierarchy_overview.md)
5. Practice with [Global Memory Advanced](02_memory_hierarchy/2b_global_memory_advanced.md)

### **Intermediate Path**
1. Deep dive into [Warp Execution Advanced](01_execution_model/1b_warp_execution_advanced.md)
2. Master [Shared Memory Complete](02_memory_hierarchy/2c_shared_memory_complete.md)
3. Understand [Streaming Multiprocessors Deep Dive](01_execution_model/1c_streaming_multiprocessors_deep.md)
4. Learn [CUDA Streams Overview](03_streams_concurrency/3_cuda_streams_overview.md)
5. Apply [CUDA Profiling](04_performance_profiling/4_cuda_profiling.md) techniques

### **Advanced Path**
1. Optimize with [Execution Constraints Guide](01_execution_model/1e_execution_constraints_guide.md)
2. Master [Unified Memory Complete](02_memory_hierarchy/2e_unified_memory_complete.md)
3. Implement [CUDA Streams Concurrency](03_streams_concurrency/3_cuda_streams_concurrency.md)
4. Debug with [Memory Debugging Complete](02_memory_hierarchy/2f_memory_debugging_complete.md)
5. Apply all concepts in real projects

## Prerequisites

- Basic understanding of C/C++ programming
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit installed (version 11.0 or later recommended)
- Basic familiarity with parallel programming concepts

## Development Environment Setup

```bash
# Check CUDA installation
nvcc --version

# Verify GPU availability
nvidia-smi

# Compile a simple CUDA program
nvcc -o hello hello.cu
```

## How to Use This Guide

### **Folder Structure**
The guide is organized into logical sections with numbered folders:
- **`00_quick_start/`** - Essential references and cheat sheets
- **`01_execution_model/`** - Thread hierarchy, warps, and execution concepts
- **`02_memory_hierarchy/`** - Memory types, optimization, and debugging
- **`03_streams_concurrency/`** - Asynchronous execution and concurrency
- **`04_performance_profiling/`** - Performance analysis and optimization tools

Each file is designed to be:
- **Self-contained** - Can be read independently
- **Practical** - Includes code examples and benchmarks
- **Reference-friendly** - Quick lookup tables and summaries
- **Progressive** - Builds complexity gradually

### **Navigation Tips**
- Use the overview files (ending in `_overview.md`) for quick orientation
- Detailed guides (ending in `_complete.md`) provide comprehensive coverage
- Cross-references between files help you find related concepts
- Code examples are optimized and ready to run

## Key Features

- **Comprehensive Coverage** - From basics to advanced optimization
- **Practical Examples** - Real-world code patterns and benchmarks
- **Performance Focus** - Optimization strategies and profiling techniques
- **Best Practices** - Industry-proven approaches and common pitfalls
- **Quick Reference** - Cheat sheets and lookup tables
- **Progressive Learning** - Multiple learning paths for different levels

## Contributing

This guide is designed to be a living reference. If you find errors, have suggestions, or want to add content:

1. Focus on practical, tested examples
2. Include performance analysis where relevant
3. Maintain the cross-reference structure
4. Follow the formatting conventions

## License

This guide is provided for educational purposes. Code examples are free to use and modify.

---

**Happy CUDA Programming!**

*Last updated: August 2025*
