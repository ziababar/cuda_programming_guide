#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <cstdio>

// Advanced CUDA Graph management for production applications
class GraphManager {
private:
    struct GraphInfo {
        std::string name;
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        bool is_instantiated;
        bool is_captured;
        std::chrono::high_resolution_clock::time_point creation_time;
        int execution_count;
        float total_execution_time;
        std::vector<std::string> dependencies;
    };

    std::map<std::string, GraphInfo> graphs;
    std::map<std::string, cudaStream_t> capture_streams;

public:
    GraphManager() {
        printf("GraphManager initialized\n");
    }

    // Create a stream for graph capture
    cudaStream_t create_capture_stream(const std::string& stream_name) {
        if (capture_streams.find(stream_name) != capture_streams.end()) {
            printf("Stream '%s' already exists\n", stream_name.c_str());
            return capture_streams[stream_name];
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        capture_streams[stream_name] = stream;

        printf("Created capture stream '%s'\n", stream_name.c_str());
        return stream;
    }

    // Begin graph capture on a stream
    bool begin_capture(const std::string& graph_name, const std::string& stream_name,
                      cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal) {

        if (graphs.find(graph_name) != graphs.end()) {
            printf("Graph '%s' already exists\n", graph_name.c_str());
            return false;
        }

        auto stream_it = capture_streams.find(stream_name);
        if (stream_it == capture_streams.end()) {
            printf("Stream '%s' not found\n", stream_name.c_str());
            return false;
        }

        // Initialize graph info
        GraphInfo info;
        info.name = graph_name;
        info.graph = nullptr;
        info.graph_exec = nullptr;
        info.is_instantiated = false;
        info.is_captured = false;
        info.creation_time = std::chrono::high_resolution_clock::now();
        info.execution_count = 0;
        info.total_execution_time = 0.0f;

        graphs[graph_name] = info;

        // Begin capture
        cudaError_t result = cudaStreamBeginCapture(stream_it->second, mode);
        if (result != cudaSuccess) {
            printf("Failed to begin capture for graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            graphs.erase(graph_name);
            return false;
        }

        printf("Started capturing graph '%s' on stream '%s' (mode: %d)\n",
               graph_name.c_str(), stream_name.c_str(), mode);
        return true;
    }

    // End graph capture
    bool end_capture(const std::string& graph_name, const std::string& stream_name) {
        auto graph_it = graphs.find(graph_name);
        if (graph_it == graphs.end()) {
            printf("Graph '%s' not found\n", graph_name.c_str());
            return false;
        }

        auto stream_it = capture_streams.find(stream_name);
        if (stream_it == capture_streams.end()) {
            printf("Stream '%s' not found\n", stream_name.c_str());
            return false;
        }

        // End capture
        cudaError_t result = cudaStreamEndCapture(stream_it->second, &graph_it->second.graph);
        if (result != cudaSuccess) {
            printf("Failed to end capture for graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            return false;
        }

        graph_it->second.is_captured = true;

        printf("Completed capturing graph '%s'\n", graph_name.c_str());
        return true;
    }

    // Instantiate a captured graph for execution
    bool instantiate_graph(const std::string& graph_name) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end() || !it->second.is_captured) {
            printf("Graph '%s' not found or not captured\n", graph_name.c_str());
            return false;
        }

        if (it->second.is_instantiated) {
            printf("Graph '%s' already instantiated\n", graph_name.c_str());
            return true;
        }

        // Instantiate the graph
        cudaError_t result = cudaGraphInstantiate(&it->second.graph_exec, it->second.graph,
                                                 nullptr, nullptr, 0);

        if (result != cudaSuccess) {
            printf("Failed to instantiate graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            return false;
        }

        it->second.is_instantiated = true;

        printf("Instantiated graph '%s' for execution\n", graph_name.c_str());
        return true;
    }

    // Launch an instantiated graph
    bool launch_graph(const std::string& graph_name, cudaStream_t stream = 0) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end() || !it->second.is_instantiated) {
            printf("Graph '%s' not found or not instantiated\n", graph_name.c_str());
            return false;
        }

        // Time the execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, stream);

        cudaError_t result = cudaGraphLaunch(it->second.graph_exec, stream);

        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        if (result != cudaSuccess) {
            printf("Failed to launch graph '%s': %s\n",
                   graph_name.c_str(), cudaGetErrorString(result));
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return false;
        }

        // Update execution statistics
        float execution_time;
        cudaEventElapsedTime(&execution_time, start, stop);

        it->second.execution_count++;
        it->second.total_execution_time += execution_time;

        printf("Executed graph '%s' in %.3f ms (execution #%d)\n",
               graph_name.c_str(), execution_time, it->second.execution_count);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return true;
    }

    // Update graph parameters (for dynamic graphs)
    bool update_graph_parameters(const std::string& graph_name,
                                const std::vector<cudaGraphNode_t>& nodes,
                                const std::vector<cudaKernelNodeParams>& new_params) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end() || !it->second.is_instantiated) {
            printf("Graph '%s' not found or not instantiated\n", graph_name.c_str());
            return false;
        }

        if (nodes.size() != new_params.size()) {
            printf("Mismatch between nodes and parameters count\n");
            return false;
        }

        for (size_t i = 0; i < nodes.size(); i++) {
            cudaError_t result = cudaGraphExecKernelNodeSetParams(
                it->second.graph_exec, nodes[i], &new_params[i]);

            if (result != cudaSuccess) {
                printf("Failed to update node %zu in graph '%s': %s\n",
                       i, graph_name.c_str(), cudaGetErrorString(result));
                return false;
            }
        }

        printf("Updated %zu nodes in graph '%s'\n", nodes.size(), graph_name.c_str());
        return true;
    }

    // Clone an existing graph
    bool clone_graph(const std::string& source_name, const std::string& target_name) {
        auto source_it = graphs.find(source_name);
        if (source_it == graphs.end() || !source_it->second.is_captured) {
            printf("Source graph '%s' not found or not captured\n", source_name.c_str());
            return false;
        }

        if (graphs.find(target_name) != graphs.end()) {
            printf("Target graph '%s' already exists\n", target_name.c_str());
            return false;
        }

        // Clone the graph
        GraphInfo clone_info = source_it->second;
        clone_info.name = target_name;
        clone_info.graph_exec = nullptr;
        clone_info.is_instantiated = false;
        clone_info.execution_count = 0;
        clone_info.total_execution_time = 0.0f;
        clone_info.creation_time = std::chrono::high_resolution_clock::now();

        cudaError_t result = cudaGraphClone(&clone_info.graph, source_it->second.graph);
        if (result != cudaSuccess) {
            printf("Failed to clone graph '%s': %s\n",
                   source_name.c_str(), cudaGetErrorString(result));
            return false;
        }

        graphs[target_name] = clone_info;

        printf("Cloned graph '%s' to '%s'\n", source_name.c_str(), target_name.c_str());
        return true;
    }

    // Get graph execution statistics
    void print_graph_statistics(const std::string& graph_name) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end()) {
            printf("Graph '%s' not found\n", graph_name.c_str());
            return;
        }

        const GraphInfo& info = it->second;

        printf("=== Graph '%s' Statistics ===\n", graph_name.c_str());
        printf("Status: %s%s\n",
               info.is_captured ? "Captured " : "Not captured ",
               info.is_instantiated ? "Instantiated" : "Not instantiated");
        printf("Executions: %d\n", info.execution_count);

        if (info.execution_count > 0) {
            float avg_time = info.total_execution_time / info.execution_count;
            printf("Total execution time: %.3f ms\n", info.total_execution_time);
            printf("Average execution time: %.3f ms\n", avg_time);

            // Calculate throughput if we have timing data
            auto current_time = std::chrono::high_resolution_clock::now();
            auto lifetime = std::chrono::duration_cast<std::chrono::milliseconds>(
                current_time - info.creation_time);

            if (lifetime.count() > 0) {
                float executions_per_sec = (info.execution_count * 1000.0f) / lifetime.count();
                printf("Execution rate: %.2f executions/second\n", executions_per_sec);
            }
        }

        printf("Dependencies: %zu\n", info.dependencies.size());
        printf("===============================\n");
    }

    // Print all graphs statistics
    void print_all_statistics() {
        printf("=== GraphManager Statistics ===\n");
        printf("Total graphs: %zu\n", graphs.size());
        printf("Capture streams: %zu\n", capture_streams.size());

        int captured = 0, instantiated = 0, total_executions = 0;
        float total_time = 0.0f;

        for (const auto& pair : graphs) {
            if (pair.second.is_captured) captured++;
            if (pair.second.is_instantiated) instantiated++;
            total_executions += pair.second.execution_count;
            total_time += pair.second.total_execution_time;
        }

        printf("Captured graphs: %d\n", captured);
        printf("Instantiated graphs: %d\n", instantiated);
        printf("Total executions: %d\n", total_executions);
        printf("Total execution time: %.3f ms\n", total_time);

        if (total_executions > 0) {
            printf("Average execution time: %.3f ms\n", total_time / total_executions);
        }

        printf("===============================\n");
    }

    // Destroy a graph
    void destroy_graph(const std::string& graph_name) {
        auto it = graphs.find(graph_name);
        if (it == graphs.end()) {
            printf("Graph '%s' not found\n", graph_name.c_str());
            return;
        }

        // Clean up graph resources
        if (it->second.graph_exec != nullptr) {
            cudaGraphExecDestroy(it->second.graph_exec);
        }

        if (it->second.graph != nullptr) {
            cudaGraphDestroy(it->second.graph);
        }

        graphs.erase(it);
        printf("Destroyed graph '%s'\n", graph_name.c_str());
    }

    ~GraphManager() {
        printf("Destroying GraphManager...\n");

        // Clean up all graphs
        for (auto& pair : graphs) {
            if (pair.second.graph_exec != nullptr) {
                cudaGraphExecDestroy(pair.second.graph_exec);
            }
            if (pair.second.graph != nullptr) {
                cudaGraphDestroy(pair.second.graph);
            }
            printf("Destroyed graph '%s'\n", pair.first.c_str());
        }

        // Clean up capture streams
        for (auto& pair : capture_streams) {
            cudaStreamDestroy(pair.second);
            printf("Destroyed stream '%s'\n", pair.first.c_str());
        }

        printf("GraphManager cleanup complete\n");
    }
};
