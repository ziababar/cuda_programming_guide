#pragma once
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <string>
#include <queue>
#include <functional>
#include <thread>
#include <cstdio>
#include <algorithm>

// Complex event-driven coordination system
class EventCoordinator {
private:
    struct DependencyNode {
        std::string name;
        cudaEvent_t completion_event;
        std::vector<std::string> dependencies;
        std::vector<std::string> dependents;
        std::function<void(cudaStream_t)> work_function;
        cudaStream_t assigned_stream;
        bool is_completed;
        bool is_scheduled;
    };

    std::map<std::string, DependencyNode> nodes;
    std::vector<cudaStream_t> stream_pool;
    std::queue<std::string> ready_queue;

public:
    EventCoordinator(const std::vector<cudaStream_t>& streams) : stream_pool(streams) {
        printf("EventCoordinator initialized with %zu streams\n", streams.size());
    }

    // Add a work node with dependencies
    void add_node(const std::string& name,
                  const std::vector<std::string>& dependencies,
                  std::function<void(cudaStream_t)> work_func) {

        if (nodes.find(name) != nodes.end()) {
            printf("Error: Node '%s' already exists\n", name.c_str());
            return;
        }

        DependencyNode node;
        node.name = name;
        node.dependencies = dependencies;
        node.work_function = work_func;
        node.is_completed = false;
        node.is_scheduled = false;
        node.assigned_stream = nullptr;

        // Create completion event
        cudaEventCreate(&node.completion_event);

        // Update dependent relationships
        for (const auto& dep : dependencies) {
            if (nodes.find(dep) != nodes.end()) {
                nodes[dep].dependents.push_back(name);
            }
        }

        nodes[name] = node;

        printf("Added node '%s' with %zu dependencies\n", name.c_str(), dependencies.size());
    }

    // Execute the dependency graph
    void execute_graph() {
        printf("=== Executing Dependency Graph ===\n");

        // Find nodes with no dependencies (ready to execute)
        for (auto& pair : nodes) {
            if (pair.second.dependencies.empty()) {
                ready_queue.push(pair.first);
                printf("Node '%s' ready for execution (no dependencies)\n", pair.first.c_str());
            }
        }

        int completed_nodes = 0;
        int total_nodes = nodes.size();

        while (completed_nodes < total_nodes) {
            // Schedule ready nodes
            while (!ready_queue.empty() && !stream_pool.empty()) {
                std::string node_name = ready_queue.front();
                ready_queue.pop();

                schedule_node(node_name);
            }

            // Check for completed nodes
            check_completions();

            // Small delay to prevent busy waiting
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }

        printf("Dependency graph execution complete\n");
    }

private:
    void schedule_node(const std::string& node_name) {
        auto& node = nodes[node_name];

        if (node.is_scheduled) {
            return;
        }

        // Assign stream
        node.assigned_stream = stream_pool[0]; // Simple round-robin
        std::rotate(stream_pool.begin(), stream_pool.begin() + 1, stream_pool.end());

        // Wait for all dependencies
        for (const auto& dep_name : node.dependencies) {
            const auto& dep_node = nodes[dep_name];
            cudaStreamWaitEvent(node.assigned_stream, dep_node.completion_event, 0);
        }

        // Execute work
        printf("Executing node '%s' on stream %p\n", node_name.c_str(), node.assigned_stream);
        node.work_function(node.assigned_stream);

        // Record completion event
        cudaEventRecord(node.completion_event, node.assigned_stream);

        node.is_scheduled = true;
    }

    void check_completions() {
        for (auto& pair : nodes) {
            auto& node = pair.second;

            if (node.is_scheduled && !node.is_completed) {
                cudaError_t status = cudaEventQuery(node.completion_event);

                if (status == cudaSuccess) {
                    node.is_completed = true;
                    printf("Node '%s' completed\n", pair.first.c_str());

                    // Check if any dependents are now ready
                    for (const auto& dependent_name : node.dependents) {
                        if (is_node_ready(dependent_name)) {
                            ready_queue.push(dependent_name);
                            printf("Node '%s' now ready (dependencies satisfied)\n",
                                   dependent_name.c_str());
                        }
                    }
                }
            }
        }
    }

    bool is_node_ready(const std::string& node_name) {
        const auto& node = nodes[node_name];

        if (node.is_scheduled) {
            return false;
        }

        // Check if all dependencies are completed
        for (const auto& dep_name : node.dependencies) {
            if (!nodes[dep_name].is_completed) {
                return false;
            }
        }

        return true;
    }

public:
    // Print graph structure
    void print_graph_structure() {
        printf("=== Dependency Graph Structure ===\n");

        for (const auto& pair : nodes) {
            const auto& node = pair.second;

            printf("Node '%s':\n", pair.first.c_str());

            if (!node.dependencies.empty()) {
                printf("  Dependencies: ");
                for (const auto& dep : node.dependencies) {
                    printf("'%s' ", dep.c_str());
                }
                printf("\n");
            }

            if (!node.dependents.empty()) {
                printf("  Dependents: ");
                for (const auto& dep : node.dependents) {
                    printf("'%s' ", dep.c_str());
                }
                printf("\n");
            }

            printf("  Status: %s\n",
                   node.is_completed ? "Completed" :
                   (node.is_scheduled ? "Scheduled" : "Waiting"));
        }

        printf("===============================\n");
    }

    ~EventCoordinator() {
        printf("Destroying EventCoordinator...\n");

        for (auto& pair : nodes) {
            cudaEventDestroy(pair.second.completion_event);
        }

        printf("EventCoordinator cleanup complete\n");
    }
};
