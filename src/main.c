#include "base.h"
#include "onnx-ml.pb-c.h"

#define MAX_TENSOR_DIMS 4
#define MAX_INPUTS 8
#define MAX_OUTPUTS 8
#define GRAPH_MAX_INPUT_EDGES 16
#define GRAPH_MAX_OUTPUT_EDGES 16
#define MAX_WEIGHTS 1024

//////////////////////////////
// Tensor

struct tensor {
        f32 *data;
        u32 dim[MAX_TENSOR_DIMS];
        u32 dim_count;
};

//////////////////////////////
// App

struct inout_entry {
        char *name;
        char *data_type;
        u32 shape_dims[MAX_TENSOR_DIMS];
        u32 shape_dim_count;
};

struct inout_config {
        struct inout_entry inputs[MAX_INPUTS];
        u32 input_count;

        struct inout_entry outputs[MAX_OUTPUTS];
        u32 output_count;
};

struct weight_data {
        char *name;
        struct tensor tensor;
};

struct app_context {
        struct arena *global_arena;
        struct arena *inference_arena;
        i32 batch_size;
        struct inout_config inout_config;

        struct graph *graph;

        struct weight_data *weights;
        u32 weight_count;

        const char *config_path;
        const char *model_path;
        const char *input_path;
        i32 inference_runs;
        b8 should_print_graph;
        b8 should_measure_time;
        b8 normalize_input;
};

static struct weight_data *app_find_weight(struct app_context *app,
                                           const char *name)
{
        for (u32 i = 0; i < app->weight_count; i++) {
                if (cstr_equal(app->weights[i].name, name)) {
                        return &app->weights[i];
                }
        }
        return NULL;
}

static void app_insert_weight(struct app_context *app,
                              struct weight_data *weight)
{
        // First check if weight with the same name already exists and replace it
        for (u32 i = 0; i < app->weight_count; i++) {
                if (cstr_equal(app->weights[i].name, weight->name)) {
                        LOG_DEBUG("Replacing weight: %s", weight->name);
                        app->weights[i] = *weight;
                        return;
                }
        }

        LOG_DEBUG("Inserting weight: %s", weight->name);
        ENSURE(app->weight_count < MAX_WEIGHTS);
        app->weights[app->weight_count++] = *weight;
}

//////////////////////////////
// Graph

enum operator_type {
        OPERATOR_TYPE_UNKNOWN = 0,
        OPERATOR_TYPE_INPUT,
        OPERATOR_TYPE_OUTPUT,
        OPERATOR_TYPE_ADD,
        OPERATOR_TYPE_SUB,
        OPERATOR_TYPE_DIV,
        OPERATOR_TYPE_CLIP,
        OPERATOR_TYPE_GEMM,
        OPERATOR_TYPE_FLATTEN,
        OPERATOR_TYPE_RELU,
        OPERATOR_TYPE_CONV,
        OPERATOR_TYPE_MAX_POOL,
        OPERATOR_TYPE_CONSTANT,
        OPERATOR_TYPE_REDUCE_MEAN,
};

enum attribute_type {
        ATTRIBUTE_TYPE_F32 = 0,
        ATTRIBUTE_TYPE_I64,
        ATTRIBUTE_TYPE_I64_ARRAY,
};

struct attribute {
        char *name;
        enum attribute_type type;
        union {
                f32 f;
                i64 i;
                struct {
                        i64 *data;
                        u32 count;
                } i64_array;
        } value;
};

struct graph_node {
        u32 input_edges[GRAPH_MAX_INPUT_EDGES];
        u32 input_edges_count;

        u32 output_edges[GRAPH_MAX_OUTPUT_EDGES];
        u32 output_edges_count;

        char *name;
        enum operator_type type;

        char **inputs;
        char **outputs;
        struct attribute *attributes;
        u32 inputs_count;
        u32 outputs_count;
        u32 attributes_count;
};

struct graph {
        struct graph_node *nodes;
        struct graph_node **sorted_nodes;
        u32 sorted_nodes_count;
        u32 nodes_count;

        char **inputs;
        char **outputs;
        u32 inputs_count;
        u32 outputs_count;
};

static f32 graph_node_find_attribute_f32(struct graph_node *node,
                                         const char *attribute_name,
                                         f32 default_value)
{
        for (u32 i = 0; i < node->attributes_count; i++) {
                struct attribute *attr = &node->attributes[i];
                if (cstr_equal(attr->name, attribute_name) &&
                    attr->type == ATTRIBUTE_TYPE_F32) {
                        return attr->value.f;
                }
        }
        return default_value;
}

static i64 graph_node_find_attribute_i64(struct graph_node *node,
                                         const char *attribute_name,
                                         i64 default_value)
{
        for (u32 i = 0; i < node->attributes_count; i++) {
                struct attribute *attr = &node->attributes[i];
                if (cstr_equal(attr->name, attribute_name) &&
                    attr->type == ATTRIBUTE_TYPE_I64) {
                        return attr->value.i;
                }
        }
        return default_value;
}

static void graph_node_find_attribute_i64_array(struct graph_node *node,
                                                const char *attribute_name,
                                                i64 **out_data, u32 *out_count,
                                                i64 *default_data,
                                                u32 default_count)
{
        for (u32 i = 0; i < node->attributes_count; i++) {
                struct attribute *attr = &node->attributes[i];
                if (cstr_equal(attr->name, attribute_name) &&
                    attr->type == ATTRIBUTE_TYPE_I64_ARRAY) {
                        *out_data = attr->value.i64_array.data;
                        *out_count = attr->value.i64_array.count;
                        return;
                }
        }
        *out_data = default_data;
        *out_count = default_count;
}

static const char *operator_type_to_cstr(enum operator_type type)
{
        switch (type) {
        case OPERATOR_TYPE_INPUT:
                return "Input";
        case OPERATOR_TYPE_OUTPUT:
                return "Output";
        case OPERATOR_TYPE_ADD:
                return "Add";
        case OPERATOR_TYPE_SUB:
                return "Sub";
        case OPERATOR_TYPE_DIV:
                return "Div";
        case OPERATOR_TYPE_CLIP:
                return "Clip";
        case OPERATOR_TYPE_GEMM:
                return "Gemm";
        case OPERATOR_TYPE_FLATTEN:
                return "Flatten";
        case OPERATOR_TYPE_RELU:
                return "Relu";
        case OPERATOR_TYPE_CONV:
                return "Conv";
        case OPERATOR_TYPE_MAX_POOL:
                return "MaxPool";
        case OPERATOR_TYPE_CONSTANT:
                return "Constant";
        case OPERATOR_TYPE_REDUCE_MEAN:
                return "ReduceMean";
        default:
                return "Unknown";
        }
}

static enum operator_type operator_type_from_cstr(const char *str)
{
        if (cstr_equal(str, "Input")) {
                return OPERATOR_TYPE_INPUT;
        } else if (cstr_equal(str, "Output")) {
                return OPERATOR_TYPE_OUTPUT;
        } else if (cstr_equal(str, "Add")) {
                return OPERATOR_TYPE_ADD;
        } else if (cstr_equal(str, "Sub")) {
                return OPERATOR_TYPE_SUB;
        } else if (cstr_equal(str, "Div")) {
                return OPERATOR_TYPE_DIV;
        } else if (cstr_equal(str, "Clip")) {
                return OPERATOR_TYPE_CLIP;
        } else if (cstr_equal(str, "Gemm")) {
                return OPERATOR_TYPE_GEMM;
        } else if (cstr_equal(str, "Flatten")) {
                return OPERATOR_TYPE_FLATTEN;
        } else if (cstr_equal(str, "Relu")) {
                return OPERATOR_TYPE_RELU;
        } else if (cstr_equal(str, "Conv")) {
                return OPERATOR_TYPE_CONV;
        } else if (cstr_equal(str, "MaxPool")) {
                return OPERATOR_TYPE_MAX_POOL;
        } else if (cstr_equal(str, "Constant")) {
                return OPERATOR_TYPE_CONSTANT;
        } else if (cstr_equal(str, "ReduceMean")) {
                return OPERATOR_TYPE_REDUCE_MEAN;
        } else {
                return OPERATOR_TYPE_UNKNOWN;
        }
}

static void graph_find_node_output(struct graph *graph, const char *node_name,
                                   u32 *indices, u32 *count, u32 max_count)
{
        for (u32 i = 0; i < graph->nodes_count; i++) {
                struct graph_node *node = &graph->nodes[i];
                for (u32 j = 0; j < node->outputs_count; j++) {
                        if (cstr_equal(node->outputs[j], node_name)) {
                                if (*count < max_count) {
                                        indices[(*count)++] = (u32)i;
                                }
                        }
                }
        }
}

static void graph_find_node_input(struct graph *graph, const char *node_name,
                                  u32 *indices, u32 *count, u32 max_count)
{
        for (u32 i = 0; i < graph->nodes_count; i++) {
                struct graph_node *node = &graph->nodes[i];
                for (u32 j = 0; j < node->inputs_count; j++) {
                        if (cstr_equal(node->inputs[j], node_name)) {
                                if (*count < max_count) {
                                        indices[(*count)++] = (u32)i;
                                }
                        }
                }
        }
}

static b8 graph_node_alloc(struct arena *arena, Onnx__NodeProto *onnx_node,
                           struct graph_node *out_node)
{
        ASSERT(out_node);

        struct graph_node *node = out_node;
        node->name = arena_push_cstr(arena, onnx_node->name);
        node->type = operator_type_from_cstr(onnx_node->op_type);
        ENSURE_MSG(node->type != OPERATOR_TYPE_UNKNOWN,
                   "Unsupported operator type: %s", onnx_node->op_type);

        // Inputs
        node->inputs_count = (u32)onnx_node->n_input;
        node->inputs = arena_push_array(arena, char *, node->inputs_count);
        for (u32 j = 0; j < node->inputs_count; j++) {
                node->inputs[j] = arena_push_cstr(arena, onnx_node->input[j]);
        }

        // Outputs
        node->outputs_count = (u32)onnx_node->n_output;
        node->outputs = arena_push_array(arena, char *, node->outputs_count);
        for (u32 j = 0; j < node->outputs_count; j++) {
                node->outputs[j] = arena_push_cstr(arena, onnx_node->output[j]);
        }

        // Attributes
        node->attributes_count = (u32)onnx_node->n_attribute;
        node->attributes = arena_push_array(arena, struct attribute,
                                            node->attributes_count);
        for (u32 j = 0; j < node->attributes_count; j++) {
                Onnx__AttributeProto *onnx_attr = onnx_node->attribute[j];
                struct attribute *attr = &node->attributes[j];
                attr->name = arena_push_cstr(arena, onnx_attr->name);
                if (onnx_attr->has_f) {
                        attr->type = ATTRIBUTE_TYPE_F32;
                        attr->value.f = onnx_attr->f;
                } else if (onnx_attr->has_i) {
                        attr->type = ATTRIBUTE_TYPE_I64;
                        attr->value.i = onnx_attr->i;
                } else if (onnx_attr->n_ints > 0) {
                        attr->type = ATTRIBUTE_TYPE_I64_ARRAY;
                        attr->value.i64_array.count = (u32)onnx_attr->n_ints;
                        attr->value.i64_array.data = arena_push_array(
                                arena, i64, attr->value.i64_array.count);
                        for (u32 k = 0; k < attr->value.i64_array.count; k++) {
                                attr->value.i64_array.data[k] =
                                        onnx_attr->ints[k];
                        }
                } else {
                        FAIL_MSG("Unsupported attribute type in node: %s",
                                 onnx_node->name);
                }
        }

        return true;
}

static void topological_sort_dfs(struct graph *graph,
                                 struct graph_node **visited,
                                 u32 *visited_count, struct graph_node **stack,
                                 u32 *stack_count, struct graph_node *node)
{
        ASSERT(*visited_count < graph->nodes_count);
        visited[(*visited_count)++] = node;

        for (u32 o = 0; o < node->output_edges_count; ++o) {
                struct graph_node *child_node =
                        &graph->nodes[node->output_edges[o]];

                b8 is_visited = false;
                for (u32 v = 0; v < *visited_count; ++v) {
                        if (visited[v] == child_node) {
                                is_visited = true;
                                break;
                        }
                }

                if (is_visited) {
                        continue;
                }

                topological_sort_dfs(graph, visited, visited_count, stack,
                                     stack_count, child_node);
        }

        ASSERT(*stack_count < graph->nodes_count);
        stack[(*stack_count)++] = node;
}

static void graph_topological_sort(struct graph *graph)
{
        struct arena_temp scratch = arena_scratch_begin(NULL, 0);

        struct graph_node **node_stack = arena_push_array(
                scratch.arena, struct graph_node *, graph->nodes_count);
        u32 stack_size = 0;

        struct graph_node **visited_nodes = arena_push_array(
                scratch.arena, struct graph_node *, graph->nodes_count);
        u32 visited_count = 0;

        for (u32 i = 0; i < graph->nodes_count; i++) {
                struct graph_node *node = &graph->nodes[i];

                b8 is_input_node = false;
                for (u32 j = 0; j < node->inputs_count; ++j) {
                        char *input_name = node->inputs[j];
                        for (u32 k = 0; k < graph->inputs_count; ++k) {
                                if (cstr_equal(input_name, graph->inputs[k])) {
                                        is_input_node = true;
                                        break;
                                }
                        }
                }

                if (node->inputs_count != 0 && !is_input_node) {
                        continue;
                }

                topological_sort_dfs(graph, visited_nodes, &visited_count,
                                     node_stack, &stack_size, node);
        }

        for (u32 i = 0; i < stack_size; ++i) {
                struct graph_node *node = node_stack[stack_size - i - 1];
                graph->sorted_nodes[i] = node;
        }
        graph->sorted_nodes_count = stack_size;

        arena_scratch_end(scratch);
}

static void graph_update_node_edges(struct graph *graph,
                                    struct graph_node *node)
{
        for (u32 i = 0; i < node->inputs_count; i++) {
                const char *input_name = node->inputs[i];
                u32 indices[GRAPH_MAX_INPUT_EDGES];
                u32 count = 0;
                graph_find_node_output(graph, input_name, indices, &count,
                                       GRAPH_MAX_INPUT_EDGES);
                for (u32 j = 0; j < count; j++) {
                        u32 index = indices[j];
                        struct graph_node *src_node = &graph->nodes[index];
                        LOG_DEBUG("Node %s input %s found at index %d (%s)",
                                  node->name, input_name, index,
                                  src_node->name);
                        ENSURE(node->input_edges_count < GRAPH_MAX_INPUT_EDGES);
                        node->input_edges[node->input_edges_count++] =
                                (u32)index;
                }
        }

        for (u32 i = 0; i < node->outputs_count; i++) {
                const char *output_name = node->outputs[i];
                u32 indices[GRAPH_MAX_OUTPUT_EDGES];
                u32 count = 0;
                graph_find_node_input(graph, output_name, indices, &count,
                                      GRAPH_MAX_OUTPUT_EDGES);
                for (u32 j = 0; j < count; j++) {
                        u32 index = indices[j];
                        struct graph_node *dst_node = &graph->nodes[index];
                        LOG_DEBUG("Node %s output %s found at index %d (%s)",
                                  node->name, output_name, index,
                                  dst_node->name);
                        ENSURE(node->output_edges_count <
                               GRAPH_MAX_OUTPUT_EDGES);
                        node->output_edges[node->output_edges_count++] =
                                (u32)index;
                }
        }
}

static struct graph *graph_alloc(struct arena *arena,
                                 Onnx__GraphProto *onnx_graph)
{
        struct graph *graph = arena_push_struct(arena, struct graph);
        graph->nodes_count = (u32)onnx_graph->n_node;
        graph->nodes =
                arena_push_array(arena, struct graph_node, graph->nodes_count);
        graph->sorted_nodes = arena_push_array(arena, struct graph_node *,
                                               graph->nodes_count);

        for (u32 i = 0; i < graph->nodes_count; i++) {
                Onnx__NodeProto *onnx_node = onnx_graph->node[i];
                struct graph_node *node = &graph->nodes[i];
                if (!graph_node_alloc(arena, onnx_node, node)) {
                        return NULL;
                }
        }

        graph->inputs_count = (u32)onnx_graph->n_input;
        graph->inputs = arena_push_array(arena, char *, graph->inputs_count);
        for (u32 j = 0; j < graph->inputs_count; j++) {
                graph->inputs[j] =
                        arena_push_cstr(arena, onnx_graph->input[j]->name);
        }

        graph->outputs_count = (u32)onnx_graph->n_output;
        graph->outputs = arena_push_array(arena, char *, graph->outputs_count);
        for (u32 j = 0; j < graph->outputs_count; j++) {
                graph->outputs[j] =
                        arena_push_cstr(arena, onnx_graph->output[j]->name);
        }

        // Update the indices for inputs and outputs on each node
        for (u32 n = 0; n < graph->nodes_count; n++) {
                struct graph_node *node = &graph->nodes[n];
                graph_update_node_edges(graph, node);
        }

        graph_topological_sort(graph);

        // Print sorted nodes for debugging
        LOG_DEBUG("Topologically sorted nodes:");
        for (u32 i = 0; i < graph->sorted_nodes_count; i++) {
                struct graph_node *node = graph->sorted_nodes[i];
                LOG_DEBUG("Node %d: %s", i, node->name);
        }

        return graph;
}

static void graph_print(struct graph *graph)
{
        print_fmt("================= Graph ================\n");
        // Print inputs
        for (u32 i = 0; i < graph->inputs_count; i++) {
                print_fmt("Input[%d]: %s\n", i, graph->inputs[i]);
        }
        // Print outputs
        for (u32 i = 0; i < graph->outputs_count; i++) {
                print_fmt("Output[%d]: %s\n", i, graph->outputs[i]);
        }

        for (u32 n = 0; n < graph->nodes_count; n++) {
                struct graph_node *node = &graph->nodes[n];

                print_fmt("Node: %s, Type: %s\n", node->name,
                          operator_type_to_cstr(node->type));
                for (u32 i = 0; i < node->inputs_count; i++) {
                        print_fmt("\tInput[%d]: %s\n", i, node->inputs[i]);
                }
                for (u32 i = 0; i < node->outputs_count; i++) {
                        print_fmt("\tOutput[%d]: %s\n", i, node->outputs[i]);
                }
                for (u32 i = 0; i < node->attributes_count; i++) {
                        struct attribute *attr = &node->attributes[i];
                        switch (attr->type) {
                        case ATTRIBUTE_TYPE_F32:
                                print_fmt("\tAttribute[%d]: %s = %f\n", i,
                                          attr->name, (f64)attr->value.f);
                                break;
                        case ATTRIBUTE_TYPE_I64:
                                print_fmt("\tAttribute[%d]: %s = %ld\n", i,
                                          attr->name, attr->value.i);
                                break;
                        case ATTRIBUTE_TYPE_I64_ARRAY: {
                                print_fmt("\tAttribute[%d]: %s = [\n", i,
                                          attr->name);
                                for (u32 j = 0; j < attr->value.i64_array.count;
                                     j++) {
                                        print_fmt(
                                                "\t\t%ld\n",
                                                attr->value.i64_array.data[j]);
                                }
                                print_fmt("\t]\n");
                                break;
                        }
                        default:
                                print_fmt(
                                        "\tAttribute[%d]: %s = <unknown type>\n",
                                        i, attr->name);
                                break;
                        }
                }
        }
        print_fmt("=========================================\n");
}

//////////////////////////////
// Tensor operations

static struct tensor tensor_alloc(struct arena *arena, const u32 *dims,
                                  u32 dim_count)
{
        ASSERT(dim_count <= MAX_TENSOR_DIMS);

        struct tensor tensor;
        tensor.dim_count = dim_count;
        usize total_size = 1;
        for (u32 i = 0; i < dim_count; i++) {
                ASSERT(dims[i] > 0);
                tensor.dim[i] = dims[i];
                total_size *= dims[i];
        }
        tensor.data = arena_push_array(arena, f32, total_size);
        return tensor;
}

static struct tensor tensor_alloc_from_bytes(struct arena *arena, u8 *data,
                                             usize size, u32 *dims,
                                             u32 dim_count, b8 normalize)
{
        struct tensor tensor = { 0 };
        tensor.dim_count = dim_count;
        for (u32 i = 0; i < dim_count; i++) {
                ASSERT(dims[i] > 0);
                tensor.dim[i] = dims[i];
        }
        tensor.data = arena_push_array(arena, f32, size);
        usize total_size = 1;
        for (u32 i = 0; i < dim_count; i++) {
                total_size *= dims[i];
        }
        ENSURE(size == total_size);
        for (usize i = 0; i < size; i++) {
                if (normalize) {
                        tensor.data[i] = (f32)data[i] / 255.0f;
                } else {
                        tensor.data[i] = (f32)data[i];
                }
        }
        return tensor;
}

//////////////////////////////
// Model loading

static b8 validate_model(Onnx__ModelProto *model, struct app_context *app_ctx)
{
        if (!model) {
                LOG_ERROR("Model is NULL");
                return false;
        }
        if (!model->graph) {
                LOG_ERROR("Model does not contain a graph");
                return false;
        }

        if (model->graph->n_input != app_ctx->inout_config.input_count) {
                LOG_ERROR("Model input count (%zu) does not match config (%d)",
                          model->graph->n_input,
                          app_ctx->inout_config.input_count);
                return false;
        }
        if (model->graph->n_output != app_ctx->inout_config.output_count) {
                LOG_ERROR("Model output count (%zu) does not match config (%d)",
                          model->graph->n_output,
                          app_ctx->inout_config.output_count);
                return false;
        }

        for (u32 i = 0; i < model->graph->n_input; i++) {
                if (!cstr_equal(model->graph->input[i]->name,
                                app_ctx->inout_config.inputs[i].name)) {
                        LOG_ERROR(
                                "Model input name (%s) does not match config (%s)",
                                model->graph->input[i]->name,
                                app_ctx->inout_config.inputs[i].name);
                        return false;
                }
        }

        for (u32 i = 0; i < model->graph->n_output; i++) {
                if (!cstr_equal(model->graph->output[i]->name,
                                app_ctx->inout_config.outputs[i].name)) {
                        LOG_ERROR(
                                "Model output name (%s) does not match config (%s)",
                                model->graph->output[i]->name,
                                app_ctx->inout_config.outputs[i].name);
                        return false;
                }
        }

        return true;
}

static b8 load_model(struct arena *arena, struct app_context *app_ctx,
                     struct graph **out_graph)
{
        b8 success = false;
        Onnx__ModelProto *model = NULL;

        struct arena_temp scratch = arena_scratch_begin(&arena, 1);

        void *model_data = NULL;
        u64 model_size = 0;
        if (!os_file_read_all_cstr(app_ctx->model_path, scratch.arena,
                                   &model_data, &model_size)) {
                LOG_ERROR("Failed to read model file: %s", app_ctx->model_path);
                goto end;
        }

        model = onnx__model_proto__unpack(NULL, model_size, model_data);
        if (!model) {
                LOG_ERROR("Failed to parse model file: %s",
                          app_ctx->model_path);
                goto end;
        }

        if (!model->graph) {
                LOG_ERROR("Model does not contain a graph: %s",
                          app_ctx->model_path);
                goto end;
        }
        if (model->graph->n_node == 0) {
                LOG_ERROR("Model graph does not contain any nodes: %s",
                          app_ctx->model_path);
                goto end;
        }

        if (!validate_model(model, app_ctx)) {
                goto end;
        }

        // Load weights
        for (u32 i = 0; i < model->graph->n_initializer; i++) {
                Onnx__TensorProto *initializer = model->graph->initializer[i];
                LOG_DEBUG("Found initializer: %s", initializer->name);
                ENSURE_MSG(initializer->data_type ==
                                   ONNX__TENSOR_PROTO__DATA_TYPE__FLOAT,
                           "Only FLOAT initializers are supported: %s",
                           initializer->name);

                struct tensor tensor = { 0 };
                u32 total_size = 1;
                tensor.dim_count = (u32)initializer->n_dims;
                ENSURE_MSG(tensor.dim_count <= MAX_TENSOR_DIMS,
                           "Initializer exceeds max dims (%d): %s",
                           MAX_TENSOR_DIMS, initializer->name);
                for (u32 d = 0; d < tensor.dim_count; d++) {
                        tensor.dim[d] = (u32)initializer->dims[d];
                        total_size *= tensor.dim[d];
                }

                f32 *tensor_data = (f32 *)initializer->raw_data.data;
                ASSERT(initializer->raw_data.len % sizeof(f32) == 0);
                usize tensor_size =
                        (usize)(initializer->raw_data.len / sizeof(f32));
                ENSURE_MSG(tensor_size == total_size,
                           "Initializer size does not match dims: %s",
                           initializer->name);
                tensor.data = arena_push_array(arena, f32, tensor_size);
                for (usize j = 0; j < tensor_size; j++) {
                        tensor.data[j] = tensor_data[j];
                }

                struct weight_data weight = {
                        .name = arena_push_cstr(arena, initializer->name),
                        .tensor = tensor,
                };
                app_insert_weight(app_ctx, &weight);
        }

        struct graph *graph = graph_alloc(arena, model->graph);
        *out_graph = graph;

        success = true;
end:
        if (model) {
                onnx__model_proto__free_unpacked(model, NULL);
        }
        arena_scratch_end(scratch);
        return success;
}

//////////////////////////////
// Main

static void print_usage(const char *prog_name)
{
        print_fmt("Usage: %s -model <path>\n", prog_name);
        print_fmt("Options:\n");
        print_fmt("  -config <path> Specify the path to the config file.\n");
        print_fmt("  -input <path>  Specify the path to the input data.\n");
        print_fmt("  -verbose       Enable verbose logging.\n");
        print_fmt("  -print_graph   Print the loaded graph structure.\n");
        print_fmt(
                "  -loops <n>     Number of inference loops to run (default: 1).\n");
        print_fmt("  -measure_time  Measure and report inference time.\n");
        print_fmt("  -help          Show this help message.\n");
}

static b8 parse_config_inout(struct arena *arena, struct sd_node *array,
                             struct inout_entry *entries, u32 *out_count,
                             u32 max_count)
{
        for (struct sd_node *child = array->first_child; child != NULL;
             child = child->next) {
                if (!sd_node_is_object(child)) {
                        LOG_ERROR(
                                "Each entry in 'inputs' array must be an object");
                        return false;
                }

                struct string_view name_sv = { 0 };
                if (!sd_node_find_string(child, "name", NULL, &name_sv)) {
                        LOG_ERROR(
                                "Each input object must have a 'name' string");
                        return false;
                }

                struct string_view data_type_sv = { 0 };
                if (!sd_node_find_string(child, "data_type", NULL,
                                         &data_type_sv)) {
                        LOG_ERROR(
                                "Each input object must have a 'data_type' string");
                        return false;
                }

                u32 shape_dims[MAX_TENSOR_DIMS];
                u32 shape_dim_count = 0;
                struct sd_node *shape_array = NULL;
                if (!sd_node_find_array(child, "shape", &shape_array) ||
                    shape_array->child_count == 0) {
                        LOG_ERROR(
                                "Each input object must have a 'shape' array");
                        return false;
                }
                for (struct sd_node *dim_node = shape_array->first_child;
                     dim_node != NULL; dim_node = dim_node->next) {
                        f64 dim_value = 0;
                        if (!sd_node_number(dim_node, 0.0, &dim_value)) {
                                LOG_ERROR(
                                        "Each dimension in 'shape' array must be a number");
                                return false;
                        }
                        if (shape_dim_count < MAX_TENSOR_DIMS) {
                                shape_dims[shape_dim_count++] = (u32)dim_value;
                        } else {
                                LOG_ERROR("Input shape exceeds max dims (%d)",
                                          MAX_TENSOR_DIMS);
                                return false;
                        }
                }

                if (*out_count < max_count) {
                        struct inout_entry *entry = &entries[(*out_count)++];
                        entry->name = arena_push_cstr_fmt(arena, "%.*s",
                                                          (i32)name_sv.length,
                                                          name_sv.str);
                        entry->data_type = arena_push_cstr_fmt(
                                arena, "%.*s", (i32)data_type_sv.length,
                                data_type_sv.str);
                        for (u32 i = 0; i < shape_dim_count; i++) {
                                entry->shape_dims[i] = shape_dims[i];
                        }
                        entry->shape_dim_count = shape_dim_count;
                } else {
                        LOG_ERROR("Exceeded maximum number of entries (%d)",
                                  max_count);
                        return false;
                }
        }
        return true;
}

static b8 load_config_file(struct app_context *app_ctx)
{
        struct arena_temp scratch = arena_scratch_begin(NULL, 0);
        b8 success = false;

        char *config_source = NULL;
        if (!os_file_read_all_string_cstr(app_ctx->config_path, scratch.arena,
                                          &config_source)) {
                LOG_ERROR("Failed to read config file from path: %s",
                          app_ctx->config_path);
                goto end;
        }

        struct sd_parser *parser =
                sd_parse(scratch.arena, app_ctx->config_path, config_source);
        if (sd_parser_has_errors(parser)) {
                LOG_ERROR("Failed to parse config file: %s",
                          app_ctx->config_path);
                goto end;
        }

        struct sd_node *root = sd_parser_root(parser);

        struct string_view model_path_sv = { 0 };
        if (!sd_node_find_string(root, "model_path", NULL, &model_path_sv)) {
                LOG_ERROR("Config file %s is missing 'model_path' entry",
                          app_ctx->config_path);
                goto end;
        }
        app_ctx->model_path = arena_push_cstr_fmt(app_ctx->global_arena, "%.*s",
                                                  (i32)model_path_sv.length,
                                                  model_path_sv.str);

        f64 batch_size = 1;
        if (sd_node_find_number(root, "batch_size", 1.0, &batch_size)) {
                LOG_DEBUG("Config 'batch_size' set to %d", (i32)batch_size);
        }
        app_ctx->batch_size = (i32)batch_size;

        f64 normalize_input = 1.0;
        if (sd_node_find_number(root, "normalize_input", 1.0,
                                &normalize_input)) {
                LOG_DEBUG("Config 'normalize_input' set to %d",
                          (i32)normalize_input);
        }
        app_ctx->normalize_input = (b8)normalize_input;

        struct sd_node *inputs_array = NULL;
        if (!sd_node_find_array(root, "inputs", &inputs_array) ||
            inputs_array->child_count == 0) {
                LOG_ERROR("Config file %s is missing 'inputs' array",
                          app_ctx->config_path);
                goto end;
        }

        struct sd_node *outputs_array = NULL;
        if (!sd_node_find_array(root, "outputs", &outputs_array) ||
            outputs_array->child_count == 0) {
                LOG_ERROR("Config file %s is missing 'outputs' array",
                          app_ctx->config_path);
                goto end;
        }

        if (!parse_config_inout(app_ctx->global_arena, inputs_array,
                                app_ctx->inout_config.inputs,
                                &app_ctx->inout_config.input_count,
                                MAX_INPUTS)) {
                goto end;
        }

        if (!parse_config_inout(app_ctx->global_arena, outputs_array,
                                app_ctx->inout_config.outputs,
                                &app_ctx->inout_config.output_count,
                                MAX_OUTPUTS)) {
                goto end;
        }

        success = true;
end:
        arena_scratch_end(scratch);
        return success;
}

static struct tensor op_gemm(struct arena *arena, struct tensor a,
                             struct tensor b, struct tensor bias, b8 trans_a,
                             b8 trans_b, f32 alpha, f32 beta)
{
        ASSERT(a.dim_count == 2);
        ASSERT(b.dim_count == 2);
        ASSERT(bias.dim_count == 1 &&
               bias.dim[0] == (trans_b ? b.dim[0] : b.dim[1]));
        if (!trans_a && !trans_b) {
                ASSERT(a.dim[1] == b.dim[0]);
        }
        if (trans_a && !trans_b) {
                ASSERT(a.dim[0] == b.dim[0]);
        }
        if (!trans_a && trans_b) {
                ASSERT(a.dim[1] == b.dim[1]);
        }
        if (trans_a && trans_b) {
                ASSERT(a.dim[0] == b.dim[1]);
        }

        // Calculate output dimensions depending on transpositions.
        u32 N = trans_a ? a.dim[1] : a.dim[0];
        u32 M = trans_b ? b.dim[1] : b.dim[0];
        u32 K = trans_b ? b.dim[0] : b.dim[1];
        struct tensor out;
        out.dim_count = 2;
        out.dim[0] = N;
        out.dim[1] = K;
        out.data = arena_push_array(arena, f32, N * K);

        u32 n = N;
        u32 m = M;
        u32 k = K;
        f32 *A = a.data;
        f32 *B = b.data;
        f32 *out_data = out.data;
        f32 *bias_data = bias.data;

        for (u32 r = 0; r < n; ++r) {
                for (u32 c = 0; c < k; ++c) {
                        f32 res = 0;
                        for (u32 i = 0; i < m; ++i) {
                                f32 aVal = trans_a ? A[i * n + r] :
                                                     A[r * m + i];
                                f32 bVal = trans_b ? B[c * m + i] :
                                                     B[i * k + c];
                                res += aVal * bVal;
                        }
                        out_data[r * k + c] = res * alpha;
                }
        }

        // Apply bias term
        for (u32 r = 0; r < n; ++r) {
                for (u32 c = 0; c < k; ++c) {
                        out_data[r * k + c] += bias_data[c] * beta;
                }
        }

        return out;
}

static struct tensor eval_gemm(struct arena *arena, struct graph_node *node,
                               struct tensor *input_tensors)
{
        f32 alpha = graph_node_find_attribute_f32(node, "alpha", 1.0f);
        f32 beta = graph_node_find_attribute_f32(node, "beta", 1.0f);
        i64 trans_a = graph_node_find_attribute_i64(node, "transA", 0);
        i64 trans_b = graph_node_find_attribute_i64(node, "transB", 0);

        ENSURE_MSG(node->inputs_count == 3,
                   "Gemm node must have exactly 3 inputs");

        return op_gemm(arena, input_tensors[0], input_tensors[1],
                       input_tensors[2], trans_a, trans_b, alpha, beta);
}

static struct tensor op_flatten(struct tensor a, i64 axis)
{
        ASSERT(axis <= a.dim_count);

        i64 dim_before = 1;
        for (i64 i = 0; i < axis; i++) {
                dim_before *= a.dim[i];
        }

        i64 dim_after = 1;
        for (i64 i = axis; i < a.dim_count; i++) {
                dim_after *= a.dim[i];
        }

        a.dim_count = 2;
        a.dim[0] = (u32)dim_before;
        a.dim[1] = (u32)dim_after;
        return a;
}

static struct tensor eval_flatten(struct graph_node *node,
                                  struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 1,
                   "Flatten node must have exactly 1 input");

        i64 axis = graph_node_find_attribute_i64(node, "axis", 1);
        return op_flatten(input_tensors[0], (i32)axis);
}

static struct tensor op_relu(struct arena *arena, struct tensor a)
{
        usize total_size = 1;
        for (u32 i = 0; i < a.dim_count; i++) {
                total_size *= a.dim[i];
        }

        struct tensor out = tensor_alloc(arena, a.dim, a.dim_count);
        for (usize i = 0; i < total_size; i++) {
                out.data[i] = max(0.0f, a.data[i]);
        }
        return out;
}

static struct tensor eval_relu(struct arena *arena, struct graph_node *node,
                               struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 1,
                   "ReLU node must have exactly 1 input");

        return op_relu(arena, input_tensors[0]);
}

// Compute the broadcast-compatible output shape for two tensors.
// Returns false if shapes are not broadcast-compatible.
static b8 compute_broadcast_shape(u32 *out_dims, u32 *out_dim_count,
                                  struct tensor a, struct tensor b)
{
        u32 max_dims = a.dim_count > b.dim_count ? a.dim_count : b.dim_count;
        *out_dim_count = max_dims;

        // Compare dimensions from right to left
        for (u32 i = 0; i < max_dims; i++) {
                // Get dimension from right (or 1 if tensor has fewer dims)
                u32 a_idx = a.dim_count > i ? a.dim_count - 1 - i : 0;
                u32 b_idx = b.dim_count > i ? b.dim_count - 1 - i : 0;
                u32 a_dim = a.dim_count > i ? a.dim[a_idx] : 1;
                u32 b_dim = b.dim_count > i ? b.dim[b_idx] : 1;

                if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
                        return false; // Not broadcast-compatible
                }

                // Output dimension is the max of the two
                out_dims[max_dims - 1 - i] = a_dim > b_dim ? a_dim : b_dim;
        }
        return true;
}

// Compute flat index into tensor given output coordinates.
// Handles broadcasting: if tensor dim is 1, coord wraps to 0.
static usize broadcast_flat_index(u32 *out_coords, u32 out_dim_count,
                                  u32 *tensor_dims, u32 tensor_dim_count)
{
        usize idx = 0;
        usize stride = 1;

        // Process from rightmost dimension
        for (u32 i = 0; i < tensor_dim_count; i++) {
                u32 tensor_d = tensor_dim_count - 1 - i;
                u32 out_d = out_dim_count - 1 - i;
                // If tensor dim is 1, broadcast (use coord 0)
                u32 coord = tensor_dims[tensor_d] == 1 ? 0 : out_coords[out_d];
                idx += coord * stride;
                stride *= tensor_dims[tensor_d];
        }
        return idx;
}

typedef f32 (*binary_op_fn)(f32 a, f32 b);

static f32 binary_add(f32 a, f32 b)
{
        return a + b;
}
static f32 binary_sub(f32 a, f32 b)
{
        return a - b;
}
static f32 binary_div(f32 a, f32 b)
{
        return a / b;
}

static struct tensor op_binary_broadcast(struct arena *arena, struct tensor a,
                                         struct tensor b, binary_op_fn op)
{
        u32 out_dims[MAX_TENSOR_DIMS];
        u32 out_dim_count;

        b8 compatible = compute_broadcast_shape(out_dims, &out_dim_count, a, b);
        ENSURE_MSG(compatible, "Tensors are not broadcast-compatible");

        struct tensor out = tensor_alloc(arena, out_dims, out_dim_count);

        usize total_size = 1;
        for (u32 i = 0; i < out_dim_count; i++) {
                total_size *= out_dims[i];
        }

        u32 coords[MAX_TENSOR_DIMS] = { 0 };

        for (usize i = 0; i < total_size; i++) {
                usize a_idx = broadcast_flat_index(coords, out_dim_count, a.dim,
                                                   a.dim_count);
                usize b_idx = broadcast_flat_index(coords, out_dim_count, b.dim,
                                                   b.dim_count);
                out.data[i] = op(a.data[a_idx], b.data[b_idx]);

                // Increment coordinates (rightmost first)
                for (i32 d = (i32)out_dim_count - 1; d >= 0; d--) {
                        coords[d]++;
                        if (coords[d] < out_dims[d]) {
                                break;
                        }
                        coords[d] = 0;
                }
        }

        return out;
}

static struct tensor op_add(struct arena *arena, struct tensor a,
                            struct tensor b)
{
        return op_binary_broadcast(arena, a, b, binary_add);
}

static struct tensor eval_add(struct arena *arena, struct graph_node *node,
                              struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 2,
                   "Add node must have exactly 2 inputs");

        return op_add(arena, input_tensors[0], input_tensors[1]);
}

static struct tensor op_sub(struct arena *arena, struct tensor a,
                            struct tensor b)
{
        return op_binary_broadcast(arena, a, b, binary_sub);
}

static struct tensor eval_sub(struct arena *arena, struct graph_node *node,
                              struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 2,
                   "Sub node must have exactly 2 inputs");
        return op_sub(arena, input_tensors[0], input_tensors[1]);
}

static struct tensor op_div(struct arena *arena, struct tensor a,
                            struct tensor b)
{
        return op_binary_broadcast(arena, a, b, binary_div);
}

static struct tensor eval_div(struct arena *arena, struct graph_node *node,
                              struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 2,
                   "Div node must have exactly 2 inputs");
        return op_div(arena, input_tensors[0], input_tensors[1]);
}

static struct tensor op_clip(struct arena *arena, struct tensor a,
                             f32 min_value, f32 max_value)
{
        usize total_size = 1;
        for (u32 i = 0; i < a.dim_count; i++) {
                total_size *= a.dim[i];
        }

        struct tensor out = tensor_alloc(arena, a.dim, a.dim_count);
        for (usize i = 0; i < total_size; i++) {
                f32 val = a.data[i];
                if (val < min_value) {
                        val = min_value;
                } else if (val > max_value) {
                        val = max_value;
                }
                out.data[i] = val;
        }
        return out;
}

static struct tensor eval_clip(struct arena *arena, struct graph_node *node,
                               struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 3,
                   "Clip node must have exactly 3 inputs");

        ASSERT(input_tensors[1].dim_count <= 1);
        ASSERT(input_tensors[2].dim_count <= 1);
        f32 min_value = input_tensors[1].data[0];
        f32 max_value = input_tensors[2].data[0];

        return op_clip(arena, input_tensors[0], min_value, max_value);
}

static void im2col_nchw(const f32 *data_im, i64 channels, i64 height, i64 width,
                        i64 kernel_h, i64 kernel_w, i64 dilation_h,
                        i64 dilation_w, i64 pad_t, i64 pad_l, i64 pad_b,
                        i64 pad_r, i64 stride_h, i64 stride_w, f32 *data_col,
                        f32 padding_value)
{
        // clang-format off
        i64 output_h = (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
        i64 output_w = (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

        // From Intel, https://github.com/BVLC/caffe/pull/3536
        i64 channel_size = height * width;
        for (i64 channel = channels; channel--; data_im += channel_size) {
          for (i64 kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (i64 kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
              i64 input_row = -pad_t + kernel_row * dilation_h;
              for (i64 output_rows = output_h; output_rows; output_rows--) {
                if ((u64)input_row >= (u64)height) {
                  for (i64 x = 0; x < output_w; x++) {
                    *(data_col++) = padding_value;
                  }
                } else {
                  i64 input_col = -pad_l + kernel_col * dilation_w;
                  const f32* rdptr = data_im + input_row * width + input_col;
                  for (i64 i = 0; i < output_w;) {
                    i64 output_handled = 1;
                    if ((u64)input_col < (u64)width) {
                      if (stride_w == 1) {
                        // Compute the minimum of the number of input elements remaining
                        // and the number of output elements to produce.
                        output_handled = min(width - input_col, output_w - i);
                        memory_copy(data_col, (usize)output_handled * sizeof(f32), rdptr + i, (usize)output_handled * sizeof(f32));
                        data_col += output_handled;
                      } else if (stride_w == 2) {
                        // Same as above except using the number of strided input elements.
                        output_handled = min((width - input_col + 1) / 2, output_w - i);
                        const f32* local_rdptr = &rdptr[i * 2];
                        for (i64 x = output_handled; x > 0; x--) {
                          *(data_col++) = *local_rdptr;
                          local_rdptr += 2;
                        }
                      } else {
                        *(data_col++) = rdptr[i * stride_w];
                      }
                    } else {
                      *(data_col++) = padding_value;
                    }
                    input_col += output_handled * stride_w;
                    i += output_handled;
                  }
                }
                input_row += stride_h;
              }
            }
          }
        }
        // clang-format on
}

static inline i64 conv_out_dim(i64 in, i64 pad0, i64 pad1, i64 k, i64 dilation,
                               i64 stride)
{
        i64 eff_k = dilation * (k - 1) + 1;
        return (in + pad0 + pad1 - eff_k) / stride + 1;
}

static inline void conv_gemm(const f32 *W, const f32 *col, f32 *Y, i64 M, i64 K,
                             i64 P)
{
        // W:   [M, K]
        // col: [K, P]
        // Y:   [M, P]

        for (i64 m = 0; m < M; ++m) {
                for (i64 p = 0; p < P; ++p) {
                        f32 sum = 0.0f;

                        for (i64 k = 0; k < K; ++k) {
                                sum += W[m * K + k] * col[k * P + p];
                        }

                        Y[m * P + p] = sum;
                }
        }
}

static struct tensor op_conv(struct arena *arena, struct tensor input,
                             struct tensor weights, struct tensor bias,
                             i64 *dilations, u32 dilations_count,
                             i64 *kernel_shape, u32 kernel_shape_count,
                             i64 *pads, u32 pads_count, i64 *strides,
                             u32 strides_count, i64 group)
{
        ENSURE(input.dim_count == 4);
        ENSURE(weights.dim_count == 4);

        ENSURE(kernel_shape_count == 2);
        ENSURE(dilations_count == 2);
        ENSURE(pads_count == 4);
        ENSURE(strides_count == 2);

        i64 N = input.dim[0];
        i64 C = input.dim[1];
        i64 H = input.dim[2];
        i64 W = input.dim[3];
        i64 M = weights.dim[0];

        i64 kH = kernel_shape[0];
        i64 kW = kernel_shape[1];

        i64 stride_h = strides[0];
        i64 stride_w = strides[1];

        i64 dilation_h = dilations[0];
        i64 dilation_w = dilations[1];

        i64 pad_t = pads[0];
        i64 pad_l = pads[1];
        i64 pad_b = pads[2];
        i64 pad_r = pads[3];

        i64 out_h = conv_out_dim(H, pad_t, pad_b, kH, dilation_h, stride_h);
        i64 out_w = conv_out_dim(W, pad_l, pad_r, kW, dilation_w, stride_w);

        u32 out_dims[] = { (u32)N, (u32)M, (u32)out_h, (u32)out_w };

        struct tensor output = tensor_alloc(arena, out_dims, 4);

        i64 HW = H * W;
        i64 out_HW = out_h * out_w;
        i64 kHW = kH * kW;

        i64 Cg = C / group;
        i64 Mg = M / group;

        i64 X_stride = Cg * HW;
        i64 Y_stride = Mg * out_HW;
        i64 W_stride = Mg * Cg * kHW;

        i64 kernel_dim = Cg * kHW;

        i64 col_size = kernel_dim * out_HW;
        f32 *col = arena_push_array(arena, f32, (usize)col_size);

        const f32 *X = input.data;
        const f32 *w = weights.data;
        f32 *Y = output.data;

        for (i64 n = 0; n < N; ++n) {
                for (i64 g = 0; g < group; ++g) {
                        const f32 *Xg = X + g * X_stride;
                        const f32 *Wg = w + g * W_stride;
                        f32 *Yg = Y + g * Y_stride;

                        im2col_nchw(Xg, Cg, H, W, kH, kW, dilation_h,
                                    dilation_w, pad_t, pad_l, pad_b, pad_r,
                                    stride_h, stride_w, col, 0.0f);

                        conv_gemm(Wg, col, Yg, Mg, kernel_dim, out_HW);
                }

                for (i64 m = 0; m < M; ++m) {
                        f32 *Ym = Y + m * out_HW;
                        f32 b = bias.data[m];

                        for (i64 p = 0; p < out_HW; ++p) {
                                Ym[p] += b;
                        }
                }

                X += C * HW;
                Y += M * out_HW;
        }

        return output;
}

static struct tensor eval_conv(struct arena *arena, struct graph_node *node,
                               struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 2 || node->inputs_count == 3,
                   "Conv node must have 2 or 3 inputs");

        struct tensor input = input_tensors[0];
        struct tensor weights = input_tensors[1];
        struct tensor bias;
        if (node->inputs_count == 3) {
                bias = input_tensors[2];
        } else {
                // Create zero bias
                u32 out_channels = weights.dim[0];
                bias = tensor_alloc(arena, (u32[]){ out_channels }, 1);
                for (u32 i = 0; i < out_channels; i++) {
                        bias.data[i] = 0.0f;
                }
        }

        i64 group = graph_node_find_attribute_i64(node, "group", 1);

        i64 *dilations;
        u32 dilations_count;
        static i64 default_dilations[] = { 1, 1 };
        graph_node_find_attribute_i64_array(node, "dilations", &dilations,
                                            &dilations_count, default_dilations,
                                            2);

        i64 *kernel_shape;
        u32 kernel_shape_count;
        static i64 default_kernel_shape[] = { 3, 3 };
        graph_node_find_attribute_i64_array(node, "kernel_shape", &kernel_shape,
                                            &kernel_shape_count,
                                            default_kernel_shape, 2);

        i64 *pads;
        u32 pads_count;
        static i64 default_pads[] = { 0, 0, 0, 0 };
        graph_node_find_attribute_i64_array(node, "pads", &pads, &pads_count,
                                            default_pads, 4);

        i64 *strides;
        u32 strides_count;
        static i64 default_strides[] = { 1, 1 };
        graph_node_find_attribute_i64_array(node, "strides", &strides,
                                            &strides_count, default_strides, 2);

        return op_conv(arena, input, weights, bias, dilations, dilations_count,
                       kernel_shape, kernel_shape_count, pads, pads_count,
                       strides, strides_count, group);
}

static inline i64 pool_out_dim(i64 in, i64 pad0, i64 pad1, i64 eff_k,
                               i64 stride, i64 ceil_mode)
{
        if (ceil_mode) {
                return (in + pad0 + pad1 - eff_k + stride - 1) / stride + 1;
        }
        return (in + pad0 + pad1 - eff_k) / stride + 1;
}

static inline f32 maxpool_window(const f32 *X, i64 base, i64 H, i64 W, i64 oh,
                                 i64 ow, i64 kH, i64 kW, i64 stride_h,
                                 i64 stride_w, i64 dilation_h, i64 dilation_w,
                                 i64 pad_t, i64 pad_l)
{
        f32 max_val = -F32_MAX;

        for (i64 kh = 0; kh < kH; ++kh) {
                i64 ih = oh * stride_h - pad_t + kh * dilation_h;
                if (ih < 0 || ih >= H) {
                        continue;
                }

                for (i64 kw = 0; kw < kW; ++kw) {
                        i64 iw = ow * stride_w - pad_l + kw * dilation_w;
                        if (iw < 0 || iw >= W) {
                                continue;
                        }

                        f32 v = X[base + ih * W + iw];
                        if (v > max_val) {
                                max_val = v;
                        }
                }
        }

        return max_val;
}

static struct tensor op_maxpool(struct arena *arena, struct tensor input,
                                i64 *kernel_shape, u32 kernel_shape_count,
                                i64 *strides, u32 strides_count, i64 *pads,
                                u32 pads_count, i64 *dilations,
                                u32 dilations_count, i64 ceil_mode)
{
        ENSURE(input.dim_count == 4);
        ENSURE(kernel_shape_count == 2);
        ENSURE(dilations_count == 2);
        ENSURE(pads_count == 4);
        ENSURE(strides_count == 2);

        i64 N = input.dim[0];
        i64 C = input.dim[1];
        i64 H = input.dim[2];
        i64 W = input.dim[3];

        i64 kH = kernel_shape[0];
        i64 kW = kernel_shape[1];

        i64 stride_h = strides[0];
        i64 stride_w = strides[1];

        i64 dilation_h = dilations[0];
        i64 dilation_w = dilations[1];

        i64 pad_t = pads[0];
        i64 pad_l = pads[1];
        i64 pad_b = pads[2];
        i64 pad_r = pads[3];

        i64 eff_kH = (kH - 1) * dilation_h + 1;
        i64 eff_kW = (kW - 1) * dilation_w + 1;

        i64 out_h = pool_out_dim(H, pad_t, pad_b, eff_kH, stride_h, ceil_mode);
        i64 out_w = pool_out_dim(W, pad_l, pad_r, eff_kW, stride_w, ceil_mode);

        u32 out_dims[] = { (u32)N, (u32)C, (u32)out_h, (u32)out_w };

        struct tensor output = tensor_alloc(arena, out_dims, 4);

        const f32 *X = input.data;
        f32 *Y = output.data;

        i64 HW = H * W;
        i64 out_HW = out_h * out_w;
        i64 CHW = C * HW;
        i64 CoutHW = C * out_HW;

        for (i64 n = 0; n < N; ++n) {
                i64 n_in = n * CHW;
                i64 n_out = n * CoutHW;

                for (i64 c = 0; c < C; ++c) {
                        i64 base = n_in + c * HW;
                        i64 out_base = n_out + c * out_HW;

                        for (i64 oh = 0; oh < out_h; ++oh) {
                                for (i64 ow = 0; ow < out_w; ++ow) {
                                        f32 v = maxpool_window(
                                                X, base, H, W, oh, ow, kH, kW,
                                                stride_h, stride_w, dilation_h,
                                                dilation_w, pad_t, pad_l);

                                        Y[out_base + oh * out_w + ow] = v;
                                }
                        }
                }
        }

        return output;
}

static struct tensor eval_maxpool(struct arena *arena, struct graph_node *node,
                                  struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 1,
                   "MaxPool node must have exactly 1 input");

        struct tensor input = input_tensors[0];

        i64 ceil_mode = graph_node_find_attribute_i64(node, "ceil_mode", 0);

        i64 *dilations;
        u32 dilations_count;
        static i64 default_dilations[] = { 1, 1 };
        graph_node_find_attribute_i64_array(node, "dilations", &dilations,
                                            &dilations_count, default_dilations,
                                            2);

        i64 *kernel_shape;
        u32 kernel_shape_count;
        static i64 default_kernel_shape[] = { 2, 2 };
        graph_node_find_attribute_i64_array(node, "kernel_shape", &kernel_shape,
                                            &kernel_shape_count,
                                            default_kernel_shape, 2);

        i64 *pads;
        u32 pads_count;
        static i64 default_pads[] = { 0, 0, 0, 0 };
        graph_node_find_attribute_i64_array(node, "pads", &pads, &pads_count,
                                            default_pads, 4);

        i64 *strides;
        u32 strides_count;
        static i64 default_strides[] = { 1, 1 };
        graph_node_find_attribute_i64_array(node, "strides", &strides,
                                            &strides_count, default_strides, 2);

        return op_maxpool(arena, input, kernel_shape, kernel_shape_count,
                          strides, strides_count, pads, pads_count, dilations,
                          dilations_count, ceil_mode);
}

static struct tensor op_reduce_mean(struct arena *arena, struct tensor input,
                                    i64 *axes, u32 axes_count, i64 keepdims)
{
        // Normalize negative axes and mark which dimensions to reduce
        b8 reduce[MAX_TENSOR_DIMS] = { 0 };
        for (u32 i = 0; i < axes_count; i++) {
                i64 axis = axes[i];
                if (axis < 0) {
                        axis += input.dim_count;
                }
                ASSERT(axis >= 0 && axis < (i64)input.dim_count);
                reduce[axis] = true;
        }

        // Calculate output dimensions
        u32 out_dims[MAX_TENSOR_DIMS];
        u32 out_dim_count = 0;
        for (u32 i = 0; i < input.dim_count; i++) {
                if (reduce[i]) {
                        if (keepdims) {
                                out_dims[out_dim_count++] = 1;
                        }
                } else {
                        out_dims[out_dim_count++] = input.dim[i];
                }
        }
        if (out_dim_count == 0) {
                out_dims[0] = 1;
                out_dim_count = 1;
        }

        struct tensor out = tensor_alloc(arena, out_dims, out_dim_count);

        // Calculate reduction size (number of elements to average)
        usize reduce_size = 1;
        for (u32 i = 0; i < input.dim_count; i++) {
                if (reduce[i]) {
                        reduce_size *= input.dim[i];
                }
        }

        // Calculate total output size
        usize out_size = 1;
        for (u32 i = 0; i < out_dim_count; i++) {
                out_size *= out_dims[i];
        }

        // Initialize output to zero
        for (usize i = 0; i < out_size; i++) {
                out.data[i] = 0.0f;
        }

        // Calculate strides for input tensor
        usize in_strides[MAX_TENSOR_DIMS];
        in_strides[input.dim_count - 1] = 1;
        for (i32 i = (i32)input.dim_count - 2; i >= 0; i--) {
                in_strides[i] = in_strides[i + 1] * input.dim[i + 1];
        }

        // Iterate over all input elements and accumulate to output
        usize in_size = 1;
        for (u32 i = 0; i < input.dim_count; i++) {
                in_size *= input.dim[i];
        }

        for (usize in_idx = 0; in_idx < in_size; in_idx++) {
                // Convert flat index to coordinates
                u32 coords[MAX_TENSOR_DIMS];
                usize tmp = in_idx;
                for (u32 d = 0; d < input.dim_count; d++) {
                        coords[d] = (u32)(tmp / in_strides[d]);
                        tmp = tmp % in_strides[d];
                }

                // Compute output index (skip reduced dimensions if !keepdims)
                usize out_idx = 0;
                usize out_stride = 1;
                for (i32 d = (i32)out_dim_count - 1,
                         s = (i32)input.dim_count - 1;
                     s >= 0; s--) {
                        if (!reduce[s]) {
                                out_idx += coords[s] * out_stride;
                                out_stride *= out_dims[d];
                                d--;
                        } else if (keepdims) {
                                d--;
                        }
                }

                out.data[out_idx] += input.data[in_idx];
        }

        // Divide by reduction size to get mean
        for (usize i = 0; i < out_size; i++) {
                out.data[i] /= (f32)reduce_size;
        }

        return out;
}

static struct tensor eval_reduce_mean(struct arena *arena,
                                      struct graph_node *node,
                                      struct tensor *input_tensors)
{
        ENSURE_MSG(node->inputs_count == 1,
                   "ReduceMean node must have exactly 1 input");

        struct tensor input = input_tensors[0];

        i64 keepdims = graph_node_find_attribute_i64(node, "keepdims", 1);
        i64 *axes;
        u32 axes_count;
        static i64 default_axes[] = { 0 };
        graph_node_find_attribute_i64_array(node, "axes", &axes, &axes_count,
                                            default_axes, 1);

        return op_reduce_mean(arena, input, axes, axes_count, keepdims);
}

static struct tensor evaluate_node(struct arena *arena, struct graph_node *node,
                                   struct tensor *input_tensors)
{
        switch (node->type) {
        case OPERATOR_TYPE_ADD:
                return eval_add(arena, node, input_tensors);
        case OPERATOR_TYPE_SUB:
                return eval_sub(arena, node, input_tensors);
        case OPERATOR_TYPE_DIV:
                return eval_div(arena, node, input_tensors);
        case OPERATOR_TYPE_CLIP:
                return eval_clip(arena, node, input_tensors);
        case OPERATOR_TYPE_GEMM:
                return eval_gemm(arena, node, input_tensors);
        case OPERATOR_TYPE_FLATTEN:
                return eval_flatten(node, input_tensors);
        case OPERATOR_TYPE_RELU:
                return eval_relu(arena, node, input_tensors);

        case OPERATOR_TYPE_CONV:
                return eval_conv(arena, node, input_tensors);
        case OPERATOR_TYPE_MAX_POOL:
                return eval_maxpool(arena, node, input_tensors);
        case OPERATOR_TYPE_REDUCE_MEAN:
                return eval_reduce_mean(arena, node, input_tensors);
        case OPERATOR_TYPE_CONSTANT:
        case OPERATOR_TYPE_INPUT:
        case OPERATOR_TYPE_OUTPUT:
        case OPERATOR_TYPE_UNKNOWN:
                FAIL_MSG("Unimplemented operator type for node: %s",
                         node->name);
                break;
        }

        return (struct tensor){ 0 };
}

static b8 perform_inference(struct arena *arena, struct app_context *app_ctx)
{
        struct arena_temp scratch = arena_scratch_begin(&arena, 1);
        struct tensor *input_tensors =
                arena_push_array(scratch.arena, struct tensor, MAX_WEIGHTS);

        struct graph *graph = app_ctx->graph;
        for (u32 i = 0; i < graph->sorted_nodes_count; i++) {
                struct graph_node *node = graph->sorted_nodes[i];
                LOG_DEBUG("Evaluating node: %s", node->name);

                for (u32 j = 0; j < node->inputs_count; j++) {
                        char *input_name = node->inputs[j];
                        struct weight_data *input =
                                app_find_weight(app_ctx, input_name);
                        ENSURE_MSG(input, "Input weight not found: %s",
                                   input_name);
                        input_tensors[j] = input->tensor;
                }
                struct tensor output =
                        evaluate_node(arena, node, input_tensors);

                // Only single output supported
                ASSERT(node->outputs_count == 1);

                struct weight_data output_weight = {
                        .name = node->outputs[0],
                        .tensor = output,
                };
                app_insert_weight(app_ctx, &output_weight);
        }

        arena_scratch_end(scratch);
        return true;
}

i32 main(i32 argc, char **argv)
{
        base_init(NULL, "inference_engine", 0);
        struct arena *global_arena = arena_alloc();

        struct app_context *app_ctx =
                arena_push_struct(global_arena, struct app_context);
        app_ctx->global_arena = global_arena;
        app_ctx->inference_arena = arena_alloc();
        app_ctx->inference_runs = 1;

        enum log_level log_level = LOG_LEVEL_ERROR;

        // Parse command line arguments
        for (i32 i = 1; i < argc; i++) {
                if (cstr_equal(argv[i], "-config") && i + 1 < argc) {
                        app_ctx->config_path = argv[i + 1];
                        i++;
                } else if (cstr_equal(argv[i], "-input") && i + 1 < argc) {
                        app_ctx->input_path = argv[i + 1];
                        i++;
                } else if (cstr_equal(argv[i], "-print_graph")) {
                        app_ctx->should_print_graph = true;
                } else if (cstr_equal(argv[i], "-v") ||
                           cstr_equal(argv[i], "-verbose")) {
                        log_level = LOG_LEVEL_DEBUG;
                } else if (cstr_equal(argv[i], "-loops") && i + 1 < argc) {
                        app_ctx->inference_runs = cstr_to_i32(argv[i + 1]);
                        ENSURE_MSG(app_ctx->inference_runs > 0,
                                   "Number of loops must be greater than 0");
                        i++;
                } else if (cstr_equal(argv[i], "-measure_time")) {
                        app_ctx->should_measure_time = true;
                } else if (cstr_equal(argv[i], "-h") ||
                           cstr_equal(argv[i], "-help") ||
                           cstr_equal(argv[i], "--help")) {
                        print_usage(argv[0]);
                        return 0;
                } else {
                        print_usage(argv[0]);
                        return 1;
                }
        }

        log_set_level(log_level);

        if (!app_ctx->config_path || !app_ctx->input_path) {
                print_usage(argv[0]);
                return 1;
        }

        LOG_DEBUG("Loading config file from: %s", app_ctx->config_path);
        LOG_DEBUG("Loading input file from: %s", app_ctx->input_path);

        app_ctx->weights =
                arena_push_array(global_arena, struct weight_data, MAX_WEIGHTS);
        app_ctx->weight_count = 0;

        if (!load_config_file(app_ctx)) {
                return 1;
        }

        if (!load_model(global_arena, app_ctx, &app_ctx->graph)) {
                return 1;
        }

        if (app_ctx->should_print_graph) {
                graph_print(app_ctx->graph);
        }

        u8 *input_data_bytes = NULL;
        u64 input_size = 0;
        if (!os_file_read_all_cstr(app_ctx->input_path, global_arena,
                                   (void **)&input_data_bytes, &input_size)) {
                LOG_ERROR("Failed to read input data from path: %s",
                          app_ctx->input_path);
                return 1;
        }
        ENSURE_MSG(app_ctx->inout_config.input_count == 1,
                   "Only single input models are supported right now");
        u32 expected_input_size = 1;
        for (u32 i = 0; i < app_ctx->inout_config.inputs[0].shape_dim_count;
             i++) {
                expected_input_size *=
                        app_ctx->inout_config.inputs[0].shape_dims[i];
        }
        ENSURE_MSG(input_size == expected_input_size,
                   "Input data size (%lu) does not match expected size (%u)",
                   input_size, expected_input_size);

        struct os_timer inference_timer = { 0 };
        if (app_ctx->should_measure_time) {
                os_timer_start(&inference_timer);
        }

        for (i32 run = 0; run < app_ctx->inference_runs; run++) {
                LOG_DEBUG("Starting inference run %d/%d", run + 1,
                          app_ctx->inference_runs);

                arena_clear(app_ctx->inference_arena);

                struct tensor input_tensor = tensor_alloc_from_bytes(
                        app_ctx->inference_arena, input_data_bytes, input_size,
                        app_ctx->inout_config.inputs[0].shape_dims,
                        app_ctx->inout_config.inputs[0].shape_dim_count,
                        app_ctx->normalize_input);
                struct weight_data input_weight = {
                        .name = app_ctx->inout_config.inputs[0].name,
                        .tensor = input_tensor,
                };
                app_insert_weight(app_ctx, &input_weight);

                if (!perform_inference(app_ctx->inference_arena, app_ctx)) {
                        LOG_ERROR("Inference failed");
                        return 1;
                }

                struct weight_data *output_weight = app_find_weight(
                        app_ctx, app_ctx->inout_config.outputs[0].name);
                ENSURE_MSG(output_weight, "Output weight not found: %s",
                           app_ctx->inout_config.outputs[0].name);
                struct tensor output_tensor = output_weight->tensor;

                // Print the output tensor for debugging
                LOG_DEBUG("Output Tensor shape:");
                for (u32 i = 0; i < output_tensor.dim_count; i++) {
                        LOG_DEBUG("Dim[%d] = %d", i, output_tensor.dim[i]);
                }
                LOG_DEBUG("Output Tensor data:");
                usize total_size = 1;
                for (u32 i = 0; i < output_tensor.dim_count; i++) {
                        total_size *= output_tensor.dim[i];
                }
                for (usize i = 0; i < total_size; i++) {
                        LOG_DEBUG("Data[%lu] = %f", i,
                                  (f64)output_tensor.data[i]);
                }

                // Output raw tensor values for external processing
                print_fmt("TENSOR_OUTPUT_START\n");
                for (usize i = 0; i < total_size; i++) {
                        print_fmt("%.9e\n", (f64)output_tensor.data[i]);
                }
                print_fmt("TENSOR_OUTPUT_END\n");
        }

        if (app_ctx->should_measure_time) {
                f64 total_time_sec = os_timer_elapsed_seconds(&inference_timer);
                f64 avg_time_sec =
                        total_time_sec / (f64)app_ctx->inference_runs;
                print_fmt("Average inference time over %d runs: %.6f sec\n",
                          app_ctx->inference_runs, avg_time_sec);
        }

        arena_release(global_arena);
        base_shutdown();

        return 0;
}
