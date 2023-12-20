#include "dense_layer.h"

bool dense_layer_init(dense_layer * l, int n_inputs, int n_outputs, double lr){
    if (l == NULL || n_inputs < 1 || n_outputs < 1 || lr < 0){
        return false;
    }

    l->n_inputs = n_inputs;
    l->n_outputs = n_outputs;
    l->learning_rate = lr;

    matrix weights;
    matrix biases;
    matrix_random_init(-1, 1, -1, 5, n_inputs, n_outputs, &weights);
    matrix_random_init(-1, 1, -1, 5, n_outputs, 1, &biases);
    l->weights = weights;
    l->biases = biases;

    return true;

}

bool feedforward(dense_layer * l, matrix input){
    if (l == NULL || input.values == NULL) return false;

    l->input = input;

    matrix transposed_weights;
    matrix_init(l->n_outputs, l->n_inputs, &transposed_weights);
    matrix_transposition(&l->weights, &transposed_weights);
    matrix output;
    matrix_init(l->n_outputs, input.cols, &output);

    matrix_multiplication(&transposed_weights, &input, &output);
    matrix_sum_column_by_line(&output, &l->biases, &output);

    l->output = output;

    return true;

}

//bool backpropagation(dense_layer * l, matrix last_gradient);