import numpy as np


def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns:
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers
    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses
    #       the model in the reverse order.
    next_dv = dv_output
    for index in range(num_layers - 1, -1, -1):
        layer = model["layers"][index]
        if (index == num_layers - 1):
            # dv_out = dv_output
            layer_in = layer_acts[index - 1]
        elif (index == 0):
            layer_in = input
        else:
            layer_in = layer_acts[index - 1]
            # dv_out = grads[index + 1]
        # print(dv_out)
        out, dv_input, grad = layer['fwd_fn'](layer_in, layer['params'], layer['hyper_params'], True, dv_output = next_dv)
        next_dv = dv_input
        grads[index] = grad
##############

    return grads
