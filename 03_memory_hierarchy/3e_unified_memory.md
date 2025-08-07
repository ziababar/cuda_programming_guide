#  Unified Memory Complete Optimization Guide

Unified Memory simplifies GPU programming by providing a single memory space accessible from both CPU and GPU. This guide covers advanced unified memory techniques, migration optimization, and performance tuning for maximum efficiency.

**[Back to Overview](3_cuda_memory_hierarchy.md)** | **Previous: [Constant Memory Guide](3d_constant_memory.md)** | **Next: [Memory Debugging Guide](3f_memory_debugging.md)**

---

##  **Table of Contents**

1. [ Unified Memory Architecture](#-unified-memory-architecture)
2. [ Migration and Prefetching Strategies](#-migration-and-prefetching-strategies)
3. [ Memory Advice API Mastery](#-memory-advice-api-mastery)
4. [ Advanced Access Patterns](#-advanced-access-patterns)
5. [ Performance Optimization Techniques](#-performance-optimization-techniques)
6. [ Profiling and Analysis](#-profiling-and-analysis)
7. [ Real-World Applications](#-real-world-applications)

---

##  **Unified Memory Architecture**

Unified Memory creates a managed memory pool that can be accessed by both CPU and GPU with automatic migration between host and device memory. Understanding the migration behavior is key to optimal performance.

###  **Hardware Evolution**

| Architecture | Unified Memory Support | Migration Granularity | Max Bandwidth | Key Features |
|-------------|----------------------|---------------------|---------------|--------------|
| **Kepler** | Basic (CUDA 6.0) | 2MB pages | ~8 GB/s | Manual prefetching only |
| **Maxwell** | Enhanced | 2MB pages | ~12 GB/s | Demand paging |
| **Pascal** | Full | 64KB pages | ~25 GB/s | Page migration engine |
| **Volta** | Advanced | 64KB pages | ~32 GB/s | Address Translation Services |
| **Turing** | Optimized | 64KB pages | ~35 GB/s | Enhanced prefetching |
| **Ampere** | High-performance | 64KB pages | ~40 GB/s | Fine-grained migration |
| **Ada/Hopper** | Ultra-fast | 4KB pages | ~50 GB/s | Hardware-accelerated migration |

###  **Migration Behavior Visualization**
```
Initial State (CPU allocation):
CPU Memory: [] (10 pages allocated)
GPU Memory: [] (empty)

First GPU Access (demand paging):
CPU Memory: [] (pages 0-2 migrated)
GPU Memory: [] (pages 0-2 now on GPU)
           ↑ ↑ ↑
           Fault + Migrate

Optimized (with prefetching):
CPU Memory: [] (all pages migrated)
GPU Memory: [] (anticipatory migration)
           ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑ ↑
           Prefetched before kernel launch
```

###  **Memory Management States**

```cpp
// Unified Memory allocation and states
void* unified_ptr;
cudaMallocManaged(&unified_ptr, size);

/*
State Transitions:
1. ALLOCATED → CPU_RESIDENT (initial state)
2. CPU_RESIDENT → MIGRATING (on GPU access)
3. MIGRATING → GPU_RESIDENT (migration complete)
4. GPU_RESIDENT → MIGRATING (on CPU access)
5. MIGRATING → CPU_RESIDENT (migration complete)
*/
```

---

##  **Migration and Prefetching Strategies**

Effective migration control is crucial for unified memory performance. Understanding when and how to migrate data can dramatically improve application performance.

###  **Basic Prefetching Patterns**

#### **Pattern 1: Kernel Launch Prefetching**
```cpp
void optimized_kernel_execution(float* managed_data, int N, int device_id) {
    // Prefetch data to GPU before kernel launch
    cudaMemPrefetchAsync(managed_data, N * sizeof(float), device_id);

    // Launch kernel with prefetched data
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    process_data_kernel<<<grid, block>>>(managed_data, N);

    // Optionally prefetch results back to CPU
    cudaMemPrefetchAsync(managed_data, N * sizeof(float), cudaCpuDeviceId);
}

__global__ void process_data_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Data is already resident - no page faults
        data[idx] = sqrtf(data[idx] * data[idx] + 1.0f);
    }
}
```

#### **Pattern 2: Overlapped Migration**
```cpp
void overlapped_processing(float* input, float* output, int N, int device_id) {
    const int chunk_size = N / 4;  // Process in chunks

    // Process first chunk while prefetching next
    for (int chunk = 0; chunk < 4; ++chunk) {
        int offset = chunk * chunk_size;
        int current_size = (chunk == 3) ? N - offset : chunk_size;

        // Prefetch current chunk to GPU
        cudaMemPrefetchAsync(&input[offset], current_size * sizeof(float), device_id);

        // Prefehuman:

        // Prefetch next chunk while processing current (overlap)
        if (chunk < 3) {
            int next_offset = (chunk + 1) * chunk_size;
            cudaMemPrefetchAsync(&input[next_offset], chunk_size * sizeof(float), device_id);
        }

        // Launch kernel for current chunk
        dim3 block(256);
        dim3 grid((current_size + block.x - 1) / block.x);

        process_chunk_kernel<<<grid, block>>>(&input[offset], &output[offset], current_size);

        // Prefetch results back to CPU for immediate use
        cudaMemPrefetchAsync(&output[offset], current_size * sizeof(float), cudaCpuDeviceId);

        cudaDeviceSynchronize();  // Wait for current chunk completion
    }
}
```

###  **Advanced Migration Patterns**

#### **Producer-Consumer Pipeline**
```cpp
class UnifiedMemoryPipeline {
private:
    float* stage_buffers[3];  // Triple buffering
    cudaStream_t streams[3];
    int current_stage = 0;
    int buffer_size;
    int device_id;

public:
    UnifiedMemoryPipeline(int size, int dev_id) : buffer_size(size), device_id(dev_id) {
        // Allocate managed buffers
        for (int i = 0; i < 3; ++i) {
            cudaMallocManaged(&stage_buffers[i], size * sizeof(float));
            cudaStreamCreate(&streams[i]);
        }
    }

    void process_pipeline(float* host_input, int num_batches) {
        for (int batch = 0; batch < num_batches + 2; ++batch) {
            int stage = batch % 3;

            // Stage 1: CPU → Managed Memory (if not last iterations)
            if (batch < num_batches) {
                memcpy(stage_buffers[stage], &host_input[batch * buffer_size],
                       buffer_size * sizeof(float));

                // Prefetch to GPU for upcoming processing
                cudaMemPrefetchAsync(stage_buffers[stage], buffer_size * sizeof(float),
                                   device_id, streams[stage]);
            }

            // Stage 2: GPU Processing (if not first iterations)
            if (batch >= 1 && batch <= num_batches) {
                int process_stage = (stage - 1 + 3) % 3;

                dim3 block(256);
                dim3 grid((buffer_size + block.x - 1) / block.x);

                gpu_process_kernel<<<grid, block, 0, streams[process_stage]>>>(
                    stage_buffers[process_stage], buffer_size);

                // Prefetch back to CPU for output
                cudaMemPrefetchAsync(stage_buffers[process_stage],
                                   buffer_size * sizeof(float),
                                   cudaCpuDeviceId, streams[process_stage]);
            }

            // Stage 3: Output (if not first two iterations)
            if (batch >= 2) {
                int output_stage = (stage - 2 + 3) % 3;

                cudaStreamSynchronize(streams[output_stage]);

                // Process results on CPU
                process_cpu_results(stage_buffers[output_stage], buffer_size);
            }
        }
    }

    ~UnifiedMemoryPipeline() {
        for (int i = 0; i < 3; ++i) {
            cudaFree(stage_buffers[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
};
```

###  **Smart Migration Based on Access Patterns**

#### **Read-Heavy Workloads**
```cpp
void optimize_for_read_heavy(float* managed_data, int N, int device_id) {
    // For read-heavy workloads, keep data on GPU and use read-only hint
    cudaMemAdvise(managed_data, N * sizeof(float),
                  cudaMemAdviseSetReadMostly, device_id);

    // Create read-only replicas on multiple GPUs
    for (int dev = 0; dev < num_gpus; ++dev) {
        cudaMemAdvise(managed_data, N * sizeof(float),
                      cudaMemAdviseSetAccessedBy, dev);
    }

    // Prefetch to primary GPU
    cudaMemPrefetchAsync(managed_data, N * sizeof(float), device_id);

    // Launch read-heavy kernels on multiple GPUs
    for (int dev = 0; dev < num_gpus; ++dev) {
        cudaSetDevice(dev);

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);

        read_only_kernel<<<grid, block>>>(managed_data, N);
    }
}
```

#### **Write-Heavy Workloads**
```cpp
void optimize_for_write_heavy(float* managed_data, int N, int device_id) {
    // For write-heavy workloads, establish preferred location
    cudaMemAdvise(managed_data, N * sizeof(float),
                  cudaMemAdviseSetPreferredLocation, device_id);

    // Map to GPU for direct access
    cudaMemAdvise(managed_data, N * sizeof(float),
                  cudaMemAdviseSetAccessedBy, device_id);

    // Prefetch and keep on GPU
    cudaMemPrefetchAsync(managed_data, N * sizeof(float), device_id);

    // Launch write-intensive kernel
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    write_intensive_kernel<<<grid, block>>>(managed_data, N);

    // Only migrate back if CPU needs results immediately
    if (cpu_needs_results) {
        cudaMemPrefetchAsync(managed_data, N * sizeof(float), cudaCpuDeviceId);
    }
}
```

---

##  **Memory Advice API Mastery**

The Memory Advice API provides fine-grained control over unified memory behavior. Proper use of memory advice can dramatically improve performance.

###  **Complete Memory Advice Reference**

| Advice Type | Effect | Use Case | Performance Impact |
|-------------|--------|----------|-------------------|
| `cudaMemAdviseSetReadMostly` | Creates read-only replicas | Multiple readers | 2-5x faster reads |
| `cudaMemAdviseSetPreferredLocation` | Sets home location | Frequent access device | Reduces migrations |
| `cudaMemAdviseSetAccessedBy` | Hints device access | Multi-GPU scenarios | Better placement |
| `cudaMemAdviseUnsetReadMostly` | Removes read-only state | Return to normal | Enables writes |
| `cudaMemAdviseUnsetPreferredLocation` | Removes location preference | Dynamic workloads | Flexible migration |
| `cudaMemAdviseUnsetAccessedBy` | Removes access hint | Device change | Cleanup |

###  **Advanced Advice Patterns**

#### **Pattern 1: Dynamic Workload Adaptation**
```cpp
class AdaptiveMemoryManager {
private:
    struct MemoryRegion {
        void* ptr;
        size_t size;
        int current_advice;
        int access_count[8];  // Per-device access counting
        bool is_read_mostly;
    };

    std::vector<MemoryRegion> managed_regions;

public:
    void register_region(void* ptr, size_t size) {
        MemoryRegion region;
        region.ptr = ptr;
        region.size = size;
        region.current_advice = -1;
        region.is_read_mostly = false;
        memset(region.access_count, 0, sizeof(region.access_count));

        managed_regions.push_back(region);
    }

    void optimize_for_access_pattern(void* ptr, int device_id, bool is_write) {
        auto it = std::find_if(managed_regions.begin(), managed_regions.end(),
                              [ptr](const MemoryRegion& r) { return r.ptr == ptr; });

        if (it != managed_regions.end()) {
            it->access_count[device_id]++;

            // Determine optimal advice based on access pattern
            int total_accesses = 0;
            int max_device = 0;
            int max_count = 0;

            for (int i = 0; i < 8; ++i) {
                total_accesses += it->access_count[i];
                if (it->access_count[i] > max_count) {
                    max_count = it->access_count[i];
                    max_device = i;
                }
            }

            // Set preferred location if one device dominates
            if (max_count > total_accesses * 0.7f) {
                if (it->current_advice != max_device) {
                    cudaMemAdvise(ptr, it->size, cudaMemAdviseSetPreferredLocation, max_device);
                    it->current_advice = max_device;
                }
            }

            // Set read-mostly if no writes and multiple readers
            bool should_be_read_mostly = !is_write && total_accesses > 10;
            if (should_be_read_mostly != it->is_read_mostly) {
                if (should_be_read_mostly) {
                    cudaMemAdvise(ptr, it->size, cudaMemAdviseSetReadMostly, 0);
                } else {
                    cudaMemAdvise(ptr, it->size, cudaMemAdviseUnsetReadMostly, 0);
                }
                it->is_read_mostly = should_be_read_mostly;
            }
        }
    }
};

// Usage example
AdaptiveMemoryManager memory_manager;

void adaptive_processing(float* managed_data, int N) {
    memory_manager.register_region(managed_data, N * sizeof(float));

    // Training phase - collect access patterns
    for (int iteration = 0; iteration < 100; ++iteration) {
        for (int device = 0; device < num_gpus; ++device) {
            // Hint the access pattern
            memory_manager.optimize_for_access_pattern(managed_data, device, false);

            cudaSetDevice(device);
            process_on_device<<<grid, block>>>(managed_data, N);
        }
    }
}
```

#### **Pattern 2: Multi-GPU Workload Distribution**
```cpp
void setup_multi_gpu_unified_memory(float* data, int N, int num_gpus) {
    // Phase 1: Mark as read-mostly for replication
    cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetReadMostly, 0);

    // Phase 2: Inform all GPUs they will access this data
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseSetAccessedBy, gpu);
    }

    // Phase 3: Prefetch to all GPUs to create replicas
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaMemPrefetchAsync(data, N * sizeof(float), gpu);
    }

    // Phase 4: Launch parallel kernels on all GPUs
    for (int gpu = 0; gpu < num_gpus; ++gpu) {
        cudaSetDevice(gpu);

        int offset = gpu * (N / num_gpus);
        int count = (gpu == num_gpus - 1) ? N - offset : N / num_gpus;

        parallel_read_kernel<<<grid, block>>>(&data[offset], count);
    }

    cudaDeviceSynchronize();

    // Phase 5: Clean up read-mostly state if writes needed later
    cudaMemAdvise(data, N * sizeof(float), cudaMemAdviseUnsetReadMostly, 0);
}
```

###  **Dynamic Advice Management**

#### **Workload-Aware Advice Switching**
```cpp
enum class WorkloadType {
    READ_INTENSIVE,
    WRITE_INTENSIVE,
    MIXED_ACCESS,
    COMPUTE_BOUND
};

class DynamicAdviceManager {
private:
    struct ManagedBuffer {
        void* ptr;
        size_t size;
        WorkloadType current_workload;
        int preferred_device;
    };

    std::unordered_map<void*, ManagedBuffer> buffers;

public:
    void transition_workload(void* ptr, WorkloadType new_workload, int device) {
        auto& buffer = buffers[ptr];

        if (buffer.current_workload != new_workload) {
            // Clean up previous advice
            cleanup_previous_advice(buffer);

            // Apply new advice based on workload
            switch (new_workload) {
                case WorkloadType::READ_INTENSIVE:
                    cudaMemAdvise(ptr, buffer.size, cudaMemAdviseSetReadMostly, 0);
                    for (int i = 0; i < num_gpus; ++i) {
                        cudaMemAdvise(ptr, buffer.size, cudaMemAdviseSetAccessedBy, i);
                    }
                    break;

                case WorkloadType::WRITE_INTENSIVE:
                    cudaMemAdvise(ptr, buffer.size, cudaMemAdviseSetPreferredLocation, device);
                    cudaMemAdvise(ptr, buffer.size, cudaMemAdviseSetAccessedBy, device);
                    cudaMemPrefetchAsync(ptr, buffer.size, device);
                    break;

                case WorkloadType::MIXED_ACCESS:
                    cudaMemAdvise(ptr, buffer.size, cudaMemAdviseSetPreferredLocation, device);
                    break;

                case WorkloadType::COMPUTE_BOUND:
                    // Minimal advice - let demand paging handle it
                    break;
            }

            buffer.current_workload = new_workload;
            buffer.preferred_device = device;
        }
    }

private:
    void cleanup_previous_advice(const ManagedBuffer& buffer) {
        cudaMemAdvise(buffer.ptr, buffer.size, cudaMemAdviseUnsetReadMostly, 0);
        cudaMemAdvise(buffer.ptr, buffer.size, cudaMemAdviseUnsetPreferredLocation, 0);

        for (int i = 0; i < num_gpus; ++i) {
            cudaMemAdvise(buffer.ptr, buffer.size, cudaMemAdviseUnsetAccessedBy, i);
        }
    }
};
```

---

##  **Advanced Access Patterns**

Understanding and optimizing access patterns is crucial for unified memory performance. Different patterns require different optimization strategies.

###  **Pattern Analysis and Optimization**

#### **Pattern 1: Streaming Access**
```cpp
// Optimized streaming pattern with prefetching
template<typename T>
void streaming_process(T* managed_data, int N, int device_id) {
    const int STREAM_CHUNK = 64 * 1024;  // 64K elements per chunk
    const int PREFETCH_AHEAD = 2;         // Prefetch 2 chunks ahead

    for (int chunk = 0; chunk < (N + STREAM_CHUNK - 1) / STREAM_CHUNK; ++chunk) {
        int offset = chunk * STREAM_CHUNK;
        int count = min(STREAM_CHUNK, N - offset);

        // Prefetch upcoming chunks
        for (int prefetch = 1; prefetch <= PREFETCH_AHEAD; ++prefetch) {
            int prefetch_chunk = chunk + prefetch;
            if (prefetch_chunk * STREAM_CHUNK < N) {
                int prefetch_offset = prefetch_chunk * STREAM_CHUNK;
                int prefetch_count = min(STREAM_CHUNK, N - prefetch_offset);

                cudaMemPrefetchAsync(&managed_data[prefetch_offset],
                                   prefetch_count * sizeof(T), device_id);
            }
        }

        // Process current chunk
        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);

        streaming_kernel<<<grid, block>>>(&managed_data[offset], count);

        // Optional: Prefetch processed data back to CPU
        cudaMemPrefetchAsync(&managed_data[offset], count * sizeof(T), cudaCpuDeviceId);
    }
}

__global__ void streaming_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Simple streaming operation
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}
```

#### **Pattern 2: Random Access Optimization**
```cpp
// Optimize random access patterns using memory advice
void optimize_random_access(float* managed_data, int N, int device_id) {
    // For random access, establish preferred location to minimize migrations
    cudaMemAdvise(managed_data, N * sizeof(float),
                  cudaMemAdviseSetPreferredLocation, device_id);

    // Prefetch entire dataset to avoid page faults
    cudaMemPrefetchAsync(managed_data, N * sizeof(float), device_id);

    // Use larger thread blocks to improve memory access efficiency
    dim3 block(512);  // Larger blocks for random access
    dim3 grid((N + block.x - 1) / block.x);

    random_access_kernel<<<grid, block>>>(managed_data, N);
}

__global__ void random_access_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Random access pattern using hash function
        int random_idx = (idx * 17 + 23) % N;
        float temp = data[random_idx];
        data[idx] = sqrtf(temp * temp + 1.0f);
    }
}
```

#### **Pattern 3: Hierarchical Data Structures**
```cpp
// Optimize tree-like data structures in unified memory
struct TreeNode {
    float value;
    TreeNode* left;
    TreeNode* right;
    int level;
};

class UnifiedMemoryTree {
private:
    TreeNode* root;
    int total_nodes;
    int device_id;

public:
    UnifiedMemoryTree(int max_nodes, int dev_id) : total_nodes(max_nodes), device_id(dev_id) {
        // Allocate entire tree in managed memory
        cudaMallocManaged(&root, max_nodes * sizeof(TreeNode));

        // Set preferred location based on usage pattern
        cudaMemAdvise(root, max_nodes * sizeof(TreeNode),
                      cudaMemAdviseSetPreferredLocation, device_id);
    }

    void optimize_for_traversal() {
        // For tree traversal, mark as read-mostly if no modifications
        cudaMemAdvise(root, total_nodes * sizeof(TreeNode),
                      cudaMemAdviseSetReadMostly, 0);

        // Prefetch root levels first (breadth-first optimization)
        int level_size = 1;
        int offset = 0;

        for (int level = 0; level < 10 && offset < total_nodes; ++level) {
            int nodes_in_level = min(level_size, total_nodes - offset);

            cudaMemPrefetchAsync(&root[offset], nodes_in_level * sizeof(TreeNode), device_id);

            offset += nodes_in_level;
            level_size *= 2;

            // Small delay to ensure level-by-level prefetching
            cudaStreamSynchronize(0);
        }
    }

    void parallel_tree_operation() {
        optimize_for_traversal();

        // Launch tree processing kernel
        dim3 block(256);
        dim3 grid((total_nodes + block.x - 1) / block.x);

        tree_process_kernel<<<grid, block>>>(root, total_nodes);
    }
};

__global__ void tree_process_kernel(TreeNode* nodes, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Process tree node (simplified)
        nodes[idx].value = fmaxf(nodes[idx].value, 0.0f);  // ReLU activation
    }
}
```

---

##  **Performance Optimization Techniques**

###  **1. Migration Overlap Strategies**

#### **Technique 1: Asynchronous Pipeline Processing**
```cpp
class AsyncUnifiedMemoryPipeline {
private:
    struct Stage {
        float* buffer;
        cudaStream_t stream;
        cudaEvent_t ready_event;
        bool is_processing;
    };

    Stage stages[4];  // 4-stage pipeline
    int buffer_size;
    int current_stage;

public:
    AsyncUnifiedMemoryPipeline(int size) : buffer_size(size), current_stage(0) {
        for (int i = 0; i < 4; ++i) {
            cudaMallocManaged(&stages[i].buffer, size * sizeof(float));
            cudaStreamCreate(&stages[i].stream);
            cudaEventCreate(&stages[i].ready_event);
            stages[i].is_processing = false;
        }
    }

    void process_batch(float* input_data, float* output_data) {
        Stage& stage = stages[current_stage];

        // Stage 1: CPU → Managed Memory (async copy)
        cudaMemcpyAsync(stage.buffer, input_data, buffer_size * sizeof(float),
                       cudaMemcpyHostToDevice, stage.stream);

        // Stage 2: Prefetch to GPU
        cudaMemPrefetchAsync(stage.buffer, buffer_size * sizeof(float),
                           0, stage.stream);

        // Stage 3: GPU processing
        dim3 block(256);
        dim3 grid((buffer_size + block.x - 1) / block.x);

        gpu_pipeline_kernel<<<grid, block, 0, stage.stream>>>(stage.buffer, buffer_size);

        // Stage 4: Prefetch back and copy out
        cudaMemPrefetchAsync(stage.buffer, buffer_size * sizeof(float),
                           cudaCpuDeviceId, stage.stream);

        cudaMemcpyAsync(output_data, stage.buffer, buffer_size * sizeof(float),
                       cudaMemcpyDeviceToHost, stage.stream);

        cudaEventRecord(stage.ready_event, stage.stream);
        stage.is_processing = true;

        current_stage = (current_stage + 1) % 4;
    }

    void wait_for_completion() {
        for (int i = 0; i < 4; ++i) {
            if (stages[i].is_processing) {
                cudaEventSynchronize(stages[i].ready_event);
                stages[i].is_processing = false;
            }
        }
    }
};
```

#### **Technique 2: Intelligent Page-Level Prefetching**
```cpp
class PageAwarePrefetcher {
private:
    static constexpr size_t PAGE_SIZE = 64 * 1024;  // 64KB pages on modern GPUs

    struct PageRegion {
        void* start_addr;
        size_t size;
        int access_count;
        std::chrono::steady_clock::time_point last_access;
    };

    std::vector<PageRegion> tracked_pages;

public:
    void track_region(void* ptr, size_t size) {
        // Align to page boundaries
        uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t aligned_start = (start / PAGE_SIZE) * PAGE_SIZE;
        size_t aligned_size = ((start + size + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE - aligned_start;

        PageRegion region;
        region.start_addr = reinterpret_cast<void*>(aligned_start);
        region.size = aligned_size;
        region.access_count = 0;
        region.last_access = std::chrono::steady_clock::now();

        tracked_pages.push_back(region);
    }

    void smart_prefetch(void* access_addr, int device_id) {
        auto now = std::chrono::steady_clock::now();

        // Find the page containing the access
        for (auto& page : tracked_pages) {
            uintptr_t page_start = reinterpret_cast<uintptr_t>(page.start_addr);
            uintptr_t page_end = page_start + page.size;
            uintptr_t access = reinterpret_cast<uintptr_t>(access_addr);

            if (access >= page_start && access < page_end) {
                page.access_count++;
                page.last_access = now;

                // Prefetch current page
                cudaMemPrefetchAsync(page.start_addr, PAGE_SIZE, device_id);

                // Prefetch adjacent pages if access pattern suggests it
                if (page.access_count > 5) {  // Hot page
                    // Prefetch next page
                    void* next_page = reinterpret_cast<void*>(page_start + PAGE_SIZE);
                    cudaMemPrefetchAsync(next_page, PAGE_SIZE, device_id);

                    // Prefetch previous page
                    if (page_start >= PAGE_SIZE) {
                        void* prev_page = reinterpret_cast<void*>(page_start - PAGE_SIZE);
                        cudaMemPrefetchAsync(prev_page, PAGE_SIZE, device_id);
                    }
                }
                break;
            }
        }
    }
};
```

###  **2. Multi-GPU Unified Memory Optimization**

#### **Technique 1: NUMA-Aware Allocation**
```cpp
class NUMAUnifiedMemory {
private:
    struct GPUAffinity {
        int gpu_id;
        int numa_node;
        float bandwidth_to_cpu;
        std::vector<int> peer_gpus;
    };

    std::vector<GPUAffinity> gpu_topology;

public:
    void initialize_topology() {
        int num_gpus;
        cudaGetDeviceCount(&num_gpus);

        for (int i = 0; i < num_gpus; ++i) {
            GPUAffinity affinity;
            affinity.gpu_id = i;

            // Query NUMA affinity (simplified)
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            // Determine optimal allocation strategy based on topology
            affinity.numa_node = query_numa_node(i);
            affinity.bandwidth_to_cpu = measure_bandwidth_to_cpu(i);

            // Find peer GPUs for direct access
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    int can_access_peer;
                    cudaDeviceCanAccessPeer(&can_access_peer, i, j);
                    if (can_access_peer) {
                        affinity.peer_gpus.push_back(j);
                    }
                }
            }

            gpu_topology.push_back(affinity);
        }
    }

    void* allocate_for_workload(size_t size, const std::vector<int>& accessing_gpus) {
        void* ptr;
        cudaMallocManaged(&ptr, size);

        // Determine optimal location based on access pattern
        if (accessing_gpus.size() == 1) {
            // Single GPU access - prefer that GPU
            int target_gpu = accessing_gpus[0];
            cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, target_gpu);
            cudaMemPrefetchAsync(ptr, size, target_gpu);

        } else if (accessing_gpus.size() <= gpu_topology.size() / 2) {
            // Few GPUs - use read-mostly optimization
            cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, 0);

            for (int gpu : accessing_gpus) {
                cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, gpu);
                cudaMemPrefetchAsync(ptr, size, gpu);
            }

        } else {
            // Many GPUs - let demand paging handle it
            // No specific advice, rely on automatic migration
        }

        return ptr;
    }

private:
    int query_numa_node(int gpu_id) {
        // Platform-specific NUMA node query
        // This would use hwloc or similar library in practice
        return gpu_id / 2;  // Simplified assumption
    }

    float measure_bandwidth_to_cpu(int gpu_id) {
        // Micro-benchmark CPU-GPU bandwidth
        const int test_size = 64 * 1024 * 1024;  // 64MB test
        void* managed_ptr;
        cudaMallocManaged(&managed_ptr, test_size);

        cudaSetDevice(gpu_id);

        auto start = std::chrono::high_resolution_clock::now();

        // Touch all pages on GPU
        dim3 block(256);
        dim3 grid((test_size / sizeof(float) + block.x - 1) / block.x);
        touch_pages_kernel<<<grid, block>>>(static_cast<float*>(managed_ptr),
                                           test_size / sizeof(float));
        cudaDeviceSynchronize();

        // Touch all pages on CPU
        volatile float* cpu_ptr = static_cast<float*>(managed_ptr);
        for (int i = 0; i < test_size / sizeof(float); i += 4096) {
            cpu_ptr[i] = 1.0f;
        }

        auto end = std::chrono::high_resolution_clock::now();

        float elapsed = std::chrono::duration<float>(end - start).count();
        float bandwidth = (test_size * 2) / (elapsed * 1e9);  // GB/s

        cudaFree(managed_ptr);
        return bandwidth;
    }
};

__global__ void touch_pages_kernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = 1.0f;
    }
}
```

---

##  **Profiling and Analysis**

###  **Nsight Compute Unified Memory Metrics**

```bash
# Check unified memory migration efficiency
ncu --metrics unified_memory__bytes_transferred_device_to_host,unified_memory__bytes_transferred_host_to_device ./app

# Analyze page fault behavior
ncu --metrics unified_memory__page_faults_virtual_memory_backed,unified_memory__page_faults_non_replayable ./app

# Check migration bandwidth utilization
ncu --metrics unified_memory__migration_bandwidth_utilization ./app

# Comprehensive unified memory analysis
ncu --set full --section UnifiedMemoryStatistics ./app
```

###  **Nsight Systems Timeline Analysis**

```bash
# Profile unified memory behavior over time
nsys profile --trace=cuda,unified-memory --unified-memory-page-faults=true ./app

# Focus on memory transfers and page faults
nsys profile --trace=cuda --sample=none --unified-memory-page-faults=true --unified-memory-usage=true ./app

# Export detailed CSV for analysis
nsys profile --trace=cuda,unified-memory --output=detailed --export=sqlite ./app
```

###  **Custom Performance Analysis Tools**

#### **Migration Pattern Analyzer**
```cpp
class UnifiedMemoryProfiler {
private:
    struct MigrationEvent {
        std::chrono::steady_clock::time_point timestamp;
        void* address;
        size_t size;
        int from_device;
        int to_device;
        float bandwidth;
    };

    std::vector<MigrationEvent> migration_history;
    std::mutex history_mutex;
    bool profiling_enabled = false;

public:
    void start_profiling() {
        profiling_enabled = true;
        migration_history.clear();
    }

    void record_migration(void* addr, size_t size, int from_dev, int to_dev) {
        if (!profiling_enabled) return;

        std::lock_guard<std::mutex> lock(history_mutex);

        MigrationEvent event;
        event.timestamp = std::chrono::steady_clock::now();
        event.address = addr;
        event.size = size;
        event.from_device = from_dev;
        event.to_device = to_dev;

        migration_history.push_back(event);
    }

    void analyze_migration_patterns() {
        std::map<std::pair<int, int>, size_t> migration_matrix;
        std::map<void*, int> hot_addresses;

        for (const auto& event : migration_history) {
            auto key = std::make_pair(event.from_device, event.to_device);
            migration_matrix[key] += event.size;
            hot_addresses[event.address]++;
        }

        printf("Migration Matrix (bytes transferred):\n");
        for (const auto& entry : migration_matrix) {
            printf("  Device %d → Device %d: %zu bytes\n",
                   entry.first.first, entry.first.second, entry.second);
        }

        printf("\nHot Addresses (frequent migrations):\n");
        for (const auto& entry : hot_addresses) {
            if (entry.second > 5) {
                printf("  Address %p: %d migrations\n", entry.first, entry.second);
            }
        }
    }

    float calculate_migration_efficiency() {
        if (migration_history.empty()) return 0.0f;

        size_t total_bytes = 0;
        float total_time = 0.0f;

        auto start_time = migration_history[0].timestamp;
        auto end_time = migration_history.back().timestamp;

        for (const auto& event : migration_history) {
            total_bytes += event.size;
        }

        total_time = std::chrono::duration<float>(end_time - start_time).count();

        return (total_bytes / (1024.0f * 1024.0f * 1024.0f)) / total_time;  // GB/s
    }
};

// Global profiler instance
UnifiedMemoryProfiler g_profiler;

// Hook into CUDA runtime (simplified)
void profile_memory_prefetch(void* ptr, size_t size, int device) {
    g_profiler.record_migration(ptr, size, cudaCpuDeviceId, device);
    cudaMemPrefetchAsync(ptr, size, device);
}
```

#### **Access Pattern Detection**
```cpp
class AccessPatternDetector {
private:
    struct AccessInfo {
        void* address;
        std::chrono::steady_clock::time_point timestamp;
        int thread_id;
        bool is_write;
    };

    std::vector<AccessInfo> access_log;
    std::mutex log_mutex;

public:
    void log_access(void* addr, bool is_write) {
        std::lock_guard<std::mutex> lock(log_mutex);

        AccessInfo info;
        info.address = addr;
        info.timestamp = std::chrono::steady_clock::now();
        info.thread_id = std::hash<std::thread::id>{}(std::this_thread::get_id());
        info.is_write = is_write;

        access_log.push_back(info);
    }

    void detect_patterns() {
        if (access_log.size() < 10) return;

        // Analyze temporal locality
        std::map<void*, std::vector<std::chrono::steady_clock::time_point>> temporal_accesses;
        for (const auto& access : access_log) {
            temporal_accesses[access.address].push_back(access.timestamp);
        }

        // Analyze spatial locality
        std::vector<uintptr_t> addresses;
        for (const auto& access : access_log) {
            addresses.push_back(reinterpret_cast<uintptr_t>(access.address));
        }
        std::sort(addresses.begin(), addresses.end());

        // Calculate access stride
        std::map<int, int> stride_histogram;
        for (size_t i = 1; i < addresses.size(); ++i) {
            int stride = addresses[i] - addresses[i-1];
            stride_histogram[stride]++;
        }

        printf("Access Pattern Analysis:\n");
        printf("  Temporal locality: %.2f%% repeated accesses\n",
               calculate_temporal_locality());
        printf("  Most common stride: %d bytes (%d occurrences)\n",
               get_most_common_stride(), stride_histogram[get_most_common_stride()]);
    }

private:
    float calculate_temporal_locality() {
        std::set<void*> unique_addresses;
        for (const auto& access : access_log) {
            unique_addresses.insert(access.address);
        }

        return 100.0f * (1.0f - float(unique_addresses.size()) / access_log.size());
    }

    int get_most_common_stride() {
        // Implementation details for stride calculation
        return 4;  // Simplified
    }
};
```

---

##  **Real-World Applications**

###  **Scientific Computing: Molecular Dynamics with Unified Memory**

```cpp
class UnifiedMemoryMolecularDynamics {
private:
    struct Particle {
        float3 position;
        float3 velocity;
        float3 force;
        float mass;
        int type;
    };

    Particle* particles;
    float* interaction_matrix;
    int num_particles;
    int num_gpus;

public:
    UnifiedMemoryMolecularDynamics(int N, int gpus) : num_particles(N), num_gpus(gpus) {
        // Allocate particles in unified memory
        cudaMallocManaged(&particles, N * sizeof(Particle));

        // Allocate interaction matrix (read-mostly)
        int num_types = 10;
        cudaMallocManaged(&interaction_matrix,
                         num_types * num_types * sizeof(float));

        // Optimize for multi-GPU access
        setup_multi_gpu_access();
    }

    void setup_multi_gpu_access() {
        // Particles: distributed across GPUs, frequent writes
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            int start = gpu * (num_particles / num_gpus);
            int count = (gpu == num_gpus - 1) ?
                       num_particles - start : num_particles / num_gpus;

            cudaMemAdvise(&particles[start], count * sizeof(Particle),
                         cudaMemAdviseSetPreferredLocation, gpu);
        }

        // Interaction matrix: read-only, replicate on all GPUs
        cudaMemAdvise(interaction_matrix,
                     100 * sizeof(float), cudaMemAdviseSetReadMostly, 0);

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaMemAdvise(interaction_matrix, 100 * sizeof(float),
                         cudaMemAdviseSetAccessedBy, gpu);
            cudaMemPrefetchAsync(interaction_matrix, 100 * sizeof(float), gpu);
        }
    }

    void simulate_timestep(float dt) {
        // Phase 1: Force calculation (requires access to all particles)
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);

            // Prefetch all particle data for force calculation
            cudaMemPrefetchAsync(particles, num_particles * sizeof(Particle), gpu);

            int start = gpu * (num_particles / num_gpus);
            int count = (gpu == num_gpus - 1) ?
                       num_particles - start : num_particles / num_gpus;

            dim3 block(256);
            dim3 grid((count + block.x - 1) / block.x);

            calculate_forces_kernel<<<grid, block>>>(
                particles, interaction_matrix, start, count, num_particles);
        }

        // Synchronize force calculations
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);
            cudaDeviceSynchronize();
        }

        // Phase 2: Integration (local particle updates)
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);

            int start = gpu * (num_particles / num_gpus);
            int count = (gpu == num_gpus - 1) ?
                       num_particles - start : num_particles / num_gpus;

            // Only need local particles for integration
            cudaMemPrefetchAsync(&particles[start], count * sizeof(Particle), gpu);

            dim3 block(256);
            dim3 grid((count + block.x - 1) / block.x);

            integrate_particles_kernel<<<grid, block>>>(
                &particles[start], count, dt);
        }
    }

    void collect_results_on_cpu() {
        // Prefetch all results back to CPU for analysis
        cudaMemPrefetchAsync(particles, num_particles * sizeof(Particle),
                           cudaCpuDeviceId);

        cudaDeviceSynchronize();

        // Now safe to access from CPU
        analyze_system_properties();
    }

private:
    void analyze_system_properties() {
        double total_energy = 0.0;
        float3 center_of_mass = {0.0f, 0.0f, 0.0f};

        for (int i = 0; i < num_particles; ++i) {
            // Kinetic energy
            float3 v = particles[i].velocity;
            total_energy += 0.5 * particles[i].mass * (v.x*v.x + v.y*v.y + v.z*v.z);

            // Center of mass
            center_of_mass.x += particles[i].position.x * particles[i].mass;
            center_of_mass.y += particles[i].position.y * particles[i].mass;
            center_of_mass.z += particles[i].position.z * particles[i].mass;
        }

        printf("System Energy: %.6f, Center of Mass: (%.3f, %.3f, %.3f)\n",
               total_energy, center_of_mass.x, center_of_mass.y, center_of_mass.z);
    }
};

__global__ void calculate_forces_kernel(Particle* particles, float* interaction_matrix,
                                       int start_idx, int local_count, int total_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= local_count) return;

    int global_idx = start_idx + idx;
    Particle& p1 = particles[global_idx];

    float3 total_force = {0.0f, 0.0f, 0.0f};

    // Calculate forces from all other particles
    for (int j = 0; j < total_particles; ++j) {
        if (j == global_idx) continue;

        Particle& p2 = particles[j];

        float3 dr = {p2.position.x - p1.position.x,
                     p2.position.y - p1.position.y,
                     p2.position.z - p1.position.z};

        float r2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
        float r = sqrtf(r2);

        // Lookup interaction strength
        float interaction = interaction_matrix[p1.type * 10 + p2.type];

        // Lennard-Jones force (simplified)
        float force_magnitude = 24.0f * interaction * (2.0f / (r2*r2*r2*r2) - 1.0f / (r2*r2*r2));

        total_force.x += force_magnitude * dr.x / r;
        total_force.y += force_magnitude * dr.y / r;
        total_force.z += force_magnitude * dr.z / r;
    }

    p1.force = total_force;
}

__global__ void integrate_particles_kernel(Particle* particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Particle& p = particles[idx];

    // Velocity Verlet integration
    float3 acceleration = {p.force.x / p.mass, p.force.y / p.mass, p.force.z / p.mass};

    p.velocity.x += acceleration.x * dt;
    p.velocity.y += acceleration.y * dt;
    p.velocity.z += acceleration.z * dt;

    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
    p.position.z += p.velocity.z * dt;
}
```

###  **Machine Learning: Distributed Training with Unified Memory**

```cpp
class UnifiedMemoryDistributedTraining {
private:
    struct TrainingBatch {
        float* input_data;
        float* target_data;
        float* gradients;
        int batch_size;
        int feature_size;
    };

    TrainingBatch* batches;
    float* model_weights;
    float* weight_gradients;
    int num_batches;
    int num_gpus;
    int model_size;

public:
    UnifiedMemoryDistributedTraining(int batches, int gpus, int weights)
        : num_batches(batches), num_gpus(gpus), model_size(weights) {

        // Allocate training batches
        cudaMallocManaged(&batches, num_batches * sizeof(TrainingBatch));

        // Allocate model weights (shared across all GPUs)
        cudaMallocManaged(&model_weights, model_size * sizeof(float));
        cudaMallocManaged(&weight_gradients, model_size * sizeof(float));

        setup_distributed_memory();
    }

    void setup_distributed_memory() {
        // Model weights: read-mostly with occasional updates
        cudaMemAdvise(model_weights, model_size * sizeof(float),
                     cudaMemAdviseSetReadMostly, 0);

        // Make weights accessible from all GPUs
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaMemAdvise(model_weights, model_size * sizeof(float),
                         cudaMemAdviseSetAccessedBy, gpu);

            // Prefetch weights to each GPU
            cudaMemPrefetchAsync(model_weights, model_size * sizeof(float), gpu);
        }

        // Distribute batches across GPUs
        for (int batch = 0; batch < num_batches; ++batch) {
            int target_gpu = batch % num_gpus;

            cudaMemAdvise(&batches[batch], sizeof(TrainingBatch),
                         cudaMemAdviseSetPreferredLocation, target_gpu);
        }
    }

    void train_epoch() {
        // Phase 1: Forward pass (read weights, compute gradients)
        std::vector<cudaStream_t> streams(num_gpus);

        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);
            cudaStreamCreate(&streams[gpu]);
        }

        // Launch forward pass on all GPUs
        for (int batch = 0; batch < num_batches; ++batch) {
            int gpu = batch % num_gpus;
            cudaSetDevice(gpu);

            // Ensure batch data is on correct GPU
            cudaMemPrefetchAsync(&batches[batch], sizeof(TrainingBatch), gpu, streams[gpu]);

            dim3 block(256);
            dim3 grid((batches[batch].batch_size + block.x - 1) / block.x);

            forward_pass_kernel<<<grid, block, 0, streams[gpu]>>>(
                batches[batch].input_data, model_weights, batches[batch].target_data,
                batches[batch].gradients, batches[batch].batch_size);
        }

        // Synchronize all forward passes
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);
            cudaStreamSynchronize(streams[gpu]);
        }

        // Phase 2: Gradient aggregation
        aggregate_gradients();

        // Phase 3: Weight update
        update_weights();

        // Cleanup
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
            cudaSetDevice(gpu);
            cudaStreamDestroy(streams[gpu]);
        }
    }

private:
    void aggregate_gradients() {
        // Collect all gradients on GPU 0 for aggregation
        cudaSetDevice(0);

        // Reset aggregated gradients
        cudaMemset(weight_gradients, 0, model_size * sizeof(float));

        for (int batch = 0; batch < num_batches; ++batch) {
            // Prefetch batch gradients to GPU 0
            cudaMemPrefetchAsync(batches[batch].gradients,
                               model_size * sizeof(float), 0);

            dim3 block(256);
            dim3 grid((model_size + block.x - 1) / block.x);

            accumulate_gradients_kernel<<<grid, block>>>(
                weight_gradients, batches[batch].gradients, model_size);
        }

        cudaDeviceSynchronize();
    }

    void update_weights() {
        // Update weights on GPU 0, then broadcast
        cudaSetDevice(0);

        // Temporarily remove read-mostly to allow writes
        cudaMemAdvise(model_weights, model_size * sizeof(float),
                     cudaMemAdviseUnsetReadMostly, 0);

        // Ensure weights are on GPU 0 for update
        cudaMemPrefetchAsync(model_weights, model_size * sizeof(float), 0);

        dim3 block(256);
        dim3 grid((model_size + block.x - 1) / block.x);

        float learning_rate = 0.001f;
        update_weights_kernel<<<grid, block>>>(
            model_weights, weight_gradients, model_size, learning_rate);

        cudaDeviceSynchronize();

        // Restore read-mostly and broadcast to all GPUs
        cudaMemAdvise(model_weights, model_size * sizeof(float),
                     cudaMemAdviseSetReadMostly, 0);

        for (int gpu = 1; gpu < num_gpus; ++gpu) {
            cudaMemPrefetchAsync(model_weights, model_size * sizeof(float), gpu);
        }
    }
};

__global__ void forward_pass_kernel(float* input, float* weights, float* targets,
                                   float* gradients, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Simplified forward pass and gradient computation
    float prediction = 0.0f;
    for (int i = 0; i < 512; ++i) {  // Assume 512 features
        prediction += input[idx * 512 + i] * weights[i];
    }

    float error = prediction - targets[idx];

    // Compute gradients
    for (int i = 0; i < 512; ++i) {
        gradients[i] += error * input[idx * 512 + i];
    }
}

__global__ void accumulate_gradients_kernel(float* total_gradients,
                                           float* batch_gradients, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    total_gradients[idx] += batch_gradients[idx];
}

__global__ void update_weights_kernel(float* weights, float* gradients,
                                     int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    weights[idx] -= learning_rate * gradients[idx];
}
```

---

##  **Key Takeaways**

1. ** Prefetch Strategically**: Use `cudaMemPrefetchAsync` to avoid page faults during kernel execution
2. ** Leverage Memory Advice**: Use the Memory Advice API to optimize for specific access patterns
3. ** Plan for Migration**: Design algorithms considering data movement costs between CPU and GPU
4. ** Profile Migration Behavior**: Use Nsight Systems to understand and optimize memory transfers
5. ** Match Pattern to Strategy**: Different workloads benefit from different unified memory approaches

##  **Related Guides**

- **Next Step**: [Memory Debugging Complete Guide](3f_memory_debugging.md) - Troubleshoot memory issues
- **Previous**: [Constant Memory Complete Guide](3d_constant_memory.md) - Cached read-only access
- **Performance**: [ Performance Benchmarking Guide](2g_performance_benchmarking.md) - Systematic performance analysis
- **Overview**: [ Memory Hierarchy Overview](2_cuda_memory_hierarchy.md) - Quick reference and navigation

---

** Pro Tip**: Start with basic unified memory allocation, add prefetching for performance, then use memory advice for fine-tuning. Always profile to understand migration behavior!
