/*
 * Task8: OpenMP multi-source shortest path search on an undirected graph.
 *
 * Usage:
 *   mssp_omp <graph.csv> <queries.csv> <output.csv> <threads> [repeat]
 *   mssp_omp --generate <graph.csv> <queries.csv> <num_queries> [seed]
 *
 * The graph CSV contains: source,target,distance
 * The query CSV contains: source,target
 * The output CSV contains: source,target,distance
 */

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INF (DBL_MAX / 4.0)

typedef struct {
    int u;
    int v;
    double w;
} InputEdge;

typedef struct {
    int vertices;
    int edge_lines;
    int directed_edges;
    int max_id;
    int *id_to_idx;
    int *idx_to_id;
    int *row_offsets;
    int *neighbors;
    double *weights;
} Graph;

typedef struct {
    int source_id;
    int target_id;
    int source_idx;
    int target_idx;
} Query;

typedef struct {
    int vertex;
    double distance;
} HeapNode;

typedef struct {
    HeapNode *data;
    int size;
    int capacity;
} MinHeap;

static void *xmalloc(size_t size) {
    if (size == 0) {
        size = 1;
    }
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Out of memory while allocating %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void *xcalloc(size_t count, size_t size) {
    if (count == 0) {
        count = 1;
    }
    if (size == 0) {
        size = 1;
    }
    void *ptr = calloc(count, size);
    if (!ptr) {
        fprintf(stderr, "Out of memory while allocating %zu bytes\n", count * size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    if (size == 0) {
        size = 1;
    }
    void *new_ptr = realloc(ptr, size);
    if (!new_ptr) {
        fprintf(stderr, "Out of memory while reallocating %zu bytes\n", size);
        exit(EXIT_FAILURE);
    }
    return new_ptr;
}

static const char *skip_separators(const char *p) {
    while (*p && (isspace((unsigned char)*p) || *p == ',')) {
        p++;
    }
    return p;
}

static int parse_two_ints(const char *line, int *a, int *b) {
    char *end = NULL;
    const char *p = skip_separators(line);

    if (*p == '\0' || *p == '#') {
        return 0;
    }

    long first = strtol(p, &end, 10);
    if (end == p) {
        return 0;
    }

    p = skip_separators(end);
    long second = strtol(p, &end, 10);
    if (end == p) {
        return 0;
    }

    *a = (int)first;
    *b = (int)second;
    return 1;
}

static int parse_edge(const char *line, int *u, int *v, double *w) {
    char *end = NULL;
    const char *p = skip_separators(line);

    if (*p == '\0' || *p == '#') {
        return 0;
    }

    long first = strtol(p, &end, 10);
    if (end == p) {
        return 0;
    }

    p = skip_separators(end);
    long second = strtol(p, &end, 10);
    if (end == p) {
        return 0;
    }

    p = skip_separators(end);
    double weight = strtod(p, &end);
    if (end == p || !isfinite(weight) || weight < 0.0) {
        return 0;
    }

    *u = (int)first;
    *v = (int)second;
    *w = weight;
    return 1;
}

static InputEdge *read_edge_list(const char *path, int *edge_count, int *max_id) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open graph file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    int capacity = 1024;
    int count = 0;
    int max_seen = -1;
    InputEdge *edges = (InputEdge *)xmalloc((size_t)capacity * sizeof(InputEdge));
    char line[512];

    while (fgets(line, sizeof(line), fp)) {
        int u = 0;
        int v = 0;
        double w = 0.0;
        if (!parse_edge(line, &u, &v, &w)) {
            continue;
        }
        if (u < 0 || v < 0) {
            fprintf(stderr, "Negative vertex id is not supported: %d, %d\n", u, v);
            exit(EXIT_FAILURE);
        }
        if (count == capacity) {
            capacity *= 2;
            edges = (InputEdge *)xrealloc(edges, (size_t)capacity * sizeof(InputEdge));
        }
        edges[count].u = u;
        edges[count].v = v;
        edges[count].w = w;
        count++;
        if (u > max_seen) {
            max_seen = u;
        }
        if (v > max_seen) {
            max_seen = v;
        }
    }

    fclose(fp);
    *edge_count = count;
    *max_id = max_seen;
    return edges;
}

static Graph read_graph(const char *path) {
    int edge_count = 0;
    int max_id = -1;
    InputEdge *input_edges = read_edge_list(path, &edge_count, &max_id);

    if (edge_count == 0 || max_id < 0) {
        fprintf(stderr, "Graph file has no valid edges: %s\n", path);
        exit(EXIT_FAILURE);
    }

    unsigned char *present = (unsigned char *)xcalloc((size_t)max_id + 1, sizeof(unsigned char));
    for (int i = 0; i < edge_count; i++) {
        present[input_edges[i].u] = 1;
        present[input_edges[i].v] = 1;
    }

    int vertices = 0;
    for (int id = 0; id <= max_id; id++) {
        if (present[id]) {
            vertices++;
        }
    }

    Graph graph;
    graph.vertices = vertices;
    graph.edge_lines = edge_count;
    graph.directed_edges = edge_count * 2;
    graph.max_id = max_id;
    graph.id_to_idx = (int *)xmalloc(((size_t)max_id + 1) * sizeof(int));
    graph.idx_to_id = (int *)xmalloc((size_t)vertices * sizeof(int));
    graph.row_offsets = (int *)xcalloc((size_t)vertices + 1, sizeof(int));
    graph.neighbors = (int *)xmalloc((size_t)graph.directed_edges * sizeof(int));
    graph.weights = (double *)xmalloc((size_t)graph.directed_edges * sizeof(double));

    for (int id = 0; id <= max_id; id++) {
        graph.id_to_idx[id] = -1;
    }

    int idx = 0;
    for (int id = 0; id <= max_id; id++) {
        if (present[id]) {
            graph.id_to_idx[id] = idx;
            graph.idx_to_id[idx] = id;
            idx++;
        }
    }

    for (int i = 0; i < edge_count; i++) {
        int u = graph.id_to_idx[input_edges[i].u];
        int v = graph.id_to_idx[input_edges[i].v];
        graph.row_offsets[u + 1]++;
        graph.row_offsets[v + 1]++;
    }
    for (int i = 1; i <= vertices; i++) {
        graph.row_offsets[i] += graph.row_offsets[i - 1];
    }

    int *cursor = (int *)xmalloc((size_t)vertices * sizeof(int));
    memcpy(cursor, graph.row_offsets, (size_t)vertices * sizeof(int));
    for (int i = 0; i < edge_count; i++) {
        int u = graph.id_to_idx[input_edges[i].u];
        int v = graph.id_to_idx[input_edges[i].v];
        double w = input_edges[i].w;

        int pos = cursor[u]++;
        graph.neighbors[pos] = v;
        graph.weights[pos] = w;

        pos = cursor[v]++;
        graph.neighbors[pos] = u;
        graph.weights[pos] = w;
    }

    free(cursor);
    free(present);
    free(input_edges);
    return graph;
}

static void free_graph(Graph *graph) {
    free(graph->id_to_idx);
    free(graph->idx_to_id);
    free(graph->row_offsets);
    free(graph->neighbors);
    free(graph->weights);
}

static Query *read_queries(const char *path, const Graph *graph, int *query_count) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Cannot open query file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    int capacity = 1024;
    int count = 0;
    Query *queries = (Query *)xmalloc((size_t)capacity * sizeof(Query));
    char line[512];

    while (fgets(line, sizeof(line), fp)) {
        int source = 0;
        int target = 0;
        if (!parse_two_ints(line, &source, &target)) {
            continue;
        }
        if (count == capacity) {
            capacity *= 2;
            queries = (Query *)xrealloc(queries, (size_t)capacity * sizeof(Query));
        }

        int source_idx = -1;
        int target_idx = -1;
        if (source >= 0 && source <= graph->max_id) {
            source_idx = graph->id_to_idx[source];
        }
        if (target >= 0 && target <= graph->max_id) {
            target_idx = graph->id_to_idx[target];
        }

        queries[count].source_id = source;
        queries[count].target_id = target;
        queries[count].source_idx = source_idx;
        queries[count].target_idx = target_idx;
        count++;
    }

    fclose(fp);
    if (count == 0) {
        fprintf(stderr, "Query file has no valid source,target pairs: %s\n", path);
        exit(EXIT_FAILURE);
    }

    *query_count = count;
    return queries;
}

static void heap_init(MinHeap *heap) {
    heap->size = 0;
    heap->capacity = 64;
    heap->data = (HeapNode *)xmalloc((size_t)heap->capacity * sizeof(HeapNode));
}

static void heap_free(MinHeap *heap) {
    free(heap->data);
    heap->data = NULL;
    heap->size = 0;
    heap->capacity = 0;
}

static void heap_push(MinHeap *heap, int vertex, double distance) {
    if (heap->size == heap->capacity) {
        heap->capacity *= 2;
        heap->data = (HeapNode *)xrealloc(heap->data, (size_t)heap->capacity * sizeof(HeapNode));
    }

    int i = heap->size++;
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap->data[parent].distance <= distance) {
            break;
        }
        heap->data[i] = heap->data[parent];
        i = parent;
    }
    heap->data[i].vertex = vertex;
    heap->data[i].distance = distance;
}

static HeapNode heap_pop(MinHeap *heap) {
    HeapNode result = heap->data[0];
    HeapNode last = heap->data[--heap->size];

    int i = 0;
    while (1) {
        int left = i * 2 + 1;
        int right = left + 1;
        if (left >= heap->size) {
            break;
        }

        int child = left;
        if (right < heap->size && heap->data[right].distance < heap->data[left].distance) {
            child = right;
        }
        if (heap->data[child].distance >= last.distance) {
            break;
        }
        heap->data[i] = heap->data[child];
        i = child;
    }
    if (heap->size > 0) {
        heap->data[i] = last;
    }
    return result;
}

static void dijkstra(const Graph *graph, int source, double *dist, unsigned char *visited,
                     MinHeap *heap) {
    for (int i = 0; i < graph->vertices; i++) {
        dist[i] = INF;
        visited[i] = 0;
    }
    heap->size = 0;

    dist[source] = 0.0;
    heap_push(heap, source, 0.0);

    while (heap->size > 0) {
        HeapNode current = heap_pop(heap);
        int u = current.vertex;
        if (visited[u]) {
            continue;
        }
        visited[u] = 1;

        for (int e = graph->row_offsets[u]; e < graph->row_offsets[u + 1]; e++) {
            int v = graph->neighbors[e];
            double next = current.distance + graph->weights[e];
            if (next < dist[v]) {
                dist[v] = next;
                heap_push(heap, v, next);
            }
        }
    }
}

static int build_query_groups(const Graph *graph, const Query *queries, int query_count,
                              int **unique_sources_out, int **source_offsets_out,
                              int **query_indices_out) {
    int *source_counts = (int *)xcalloc((size_t)graph->vertices, sizeof(int));
    int valid_source_queries = 0;

    for (int i = 0; i < query_count; i++) {
        if (queries[i].source_idx >= 0) {
            source_counts[queries[i].source_idx]++;
            valid_source_queries++;
        }
    }

    int unique_count = 0;
    for (int i = 0; i < graph->vertices; i++) {
        if (source_counts[i] > 0) {
            unique_count++;
        }
    }

    int *unique_sources = (int *)xmalloc((size_t)unique_count * sizeof(int));
    int *source_offsets = (int *)xcalloc((size_t)graph->vertices + 1, sizeof(int));
    int *query_indices = (int *)xmalloc((size_t)valid_source_queries * sizeof(int));

    for (int i = 0; i < graph->vertices; i++) {
        source_offsets[i + 1] = source_offsets[i] + source_counts[i];
    }

    int pos = 0;
    for (int i = 0; i < graph->vertices; i++) {
        if (source_counts[i] > 0) {
            unique_sources[pos++] = i;
        }
    }

    int *cursor = (int *)xmalloc((size_t)graph->vertices * sizeof(int));
    memcpy(cursor, source_offsets, (size_t)graph->vertices * sizeof(int));
    for (int i = 0; i < query_count; i++) {
        int source = queries[i].source_idx;
        if (source >= 0) {
            query_indices[cursor[source]++] = i;
        }
    }

    free(cursor);
    free(source_counts);

    *unique_sources_out = unique_sources;
    *source_offsets_out = source_offsets;
    *query_indices_out = query_indices;
    return unique_count;
}

static double compute_shortest_paths(const Graph *graph, const Query *queries, int query_count,
                                     int num_threads, int repeat, double *answers,
                                     int *unique_source_count_out) {
    int *unique_sources = NULL;
    int *source_offsets = NULL;
    int *query_indices = NULL;
    int unique_source_count = build_query_groups(graph, queries, query_count, &unique_sources,
                                                 &source_offsets, &query_indices);

    double start = omp_get_wtime();
    for (int r = 0; r < repeat; r++) {
        for (int i = 0; i < query_count; i++) {
            answers[i] = INF;
        }

#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
        for (int i = 0; i < unique_source_count; i++) {
            int source = unique_sources[i];
            double *dist = (double *)xmalloc((size_t)graph->vertices * sizeof(double));
            unsigned char *visited =
                (unsigned char *)xmalloc((size_t)graph->vertices * sizeof(unsigned char));
            MinHeap heap;
            heap_init(&heap);

            dijkstra(graph, source, dist, visited, &heap);

            for (int qpos = source_offsets[source]; qpos < source_offsets[source + 1]; qpos++) {
                int query_idx = query_indices[qpos];
                int target = queries[query_idx].target_idx;
                if (target >= 0) {
                    answers[query_idx] = dist[target];
                }
            }

            heap_free(&heap);
            free(visited);
            free(dist);
        }
    }
    double elapsed = omp_get_wtime() - start;

    free(unique_sources);
    free(source_offsets);
    free(query_indices);

    *unique_source_count_out = unique_source_count;
    return elapsed;
}

static void write_answers(const char *path, const Query *queries, const double *answers,
                          int query_count) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open output file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    fprintf(fp, "source,target,distance\n");
    for (int i = 0; i < query_count; i++) {
        fprintf(fp, "%d,%d,", queries[i].source_id, queries[i].target_id);
        if (answers[i] >= INF / 2.0) {
            fprintf(fp, "inf\n");
        } else {
            fprintf(fp, "%.10g\n", answers[i]);
        }
    }

    fclose(fp);
}

static void generate_queries(const char *graph_path, const char *query_path, int num_queries,
                             unsigned int seed) {
    Graph graph = read_graph(graph_path);
    FILE *fp = fopen(query_path, "w");
    if (!fp) {
        fprintf(stderr, "Cannot open generated query file: %s\n", query_path);
        exit(EXIT_FAILURE);
    }

    srand(seed);
    fprintf(fp, "source,target\n");
    for (int i = 0; i < num_queries; i++) {
        int s = rand() % graph.vertices;
        int t = rand() % graph.vertices;
        fprintf(fp, "%d,%d\n", graph.idx_to_id[s], graph.idx_to_id[t]);
    }

    fclose(fp);

    printf("generated_queries=%d\n", num_queries);
    printf("query_file=%s\n", query_path);
    printf("vertices=%d\n", graph.vertices);
    printf("edges=%d\n", graph.edge_lines);

    free_graph(&graph);
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <graph.csv> <queries.csv> <output.csv> <threads> [repeat]\n"
            "  %s --generate <graph.csv> <queries.csv> <num_queries> [seed]\n",
            program, program);
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "--generate") == 0) {
        if (argc < 5 || argc > 6) {
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        int num_queries = atoi(argv[4]);
        unsigned int seed = argc == 6 ? (unsigned int)strtoul(argv[5], NULL, 10) : 2026U;
        if (num_queries <= 0) {
            fprintf(stderr, "num_queries must be positive\n");
            return EXIT_FAILURE;
        }
        generate_queries(argv[2], argv[3], num_queries, seed);
        return EXIT_SUCCESS;
    }

    if (argc < 5 || argc > 6) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *graph_path = argv[1];
    const char *query_path = argv[2];
    const char *output_path = argv[3];
    int num_threads = atoi(argv[4]);
    int repeat = argc == 6 ? atoi(argv[5]) : 1;

    if (num_threads <= 0) {
        fprintf(stderr, "threads must be positive\n");
        return EXIT_FAILURE;
    }
    if (repeat <= 0) {
        fprintf(stderr, "repeat must be positive\n");
        return EXIT_FAILURE;
    }

    Graph graph = read_graph(graph_path);
    int query_count = 0;
    Query *queries = read_queries(query_path, &graph, &query_count);
    double *answers = (double *)xmalloc((size_t)query_count * sizeof(double));

    int unique_source_count = 0;
    double elapsed = compute_shortest_paths(&graph, queries, query_count, num_threads, repeat,
                                            answers, &unique_source_count);
    write_answers(output_path, queries, answers, query_count);

    double avg_degree = graph.vertices > 0
                            ? (double)graph.directed_edges / (double)graph.vertices
                            : 0.0;

    printf("graph=%s\n", graph_path);
    printf("vertices=%d\n", graph.vertices);
    printf("edges=%d\n", graph.edge_lines);
    printf("avg_degree=%.4f\n", avg_degree);
    printf("queries=%d\n", query_count);
    printf("unique_sources=%d\n", unique_source_count);
    printf("threads=%d\n", num_threads);
    printf("repeat=%d\n", repeat);
    printf("time_seconds=%.6f\n", elapsed);
    printf("avg_time_seconds=%.6f\n", elapsed / (double)repeat);
    printf("output=%s\n", output_path);

    free(answers);
    free(queries);
    free_graph(&graph);
    return EXIT_SUCCESS;
}
