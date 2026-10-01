import numpy as np


def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    for index, layer in enumerate(model['layers']):
        if (index == 0):
            out, dv_input, grad = layer['fwd_fn'](input, layer['params'], layer['hyper_params'], False)
            activations[index] = out
        else:
            out, dv_input, grad = layer['fwd_fn'](activations[index - 1], layer['params'], layer['hyper_params'], False)
        activations[index] = out
    ##########################
    output = activations[-1]

    return output, activations
