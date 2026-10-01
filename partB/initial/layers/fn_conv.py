import numpy as np
import scipy.signal


def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    _, _, filter_depth, num_filters = params['W'].shape
    out_height = in_height - params['W'].shape[0] + 1
    out_width = in_width - params['W'].shape[1] + 1

    assert params['W'].shape[2] == input.shape[2], 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}

    # TODO: FORWARD CODE
    #       Update output with values
    for i in range(batch_size):
        for j in range(num_filters):
            for k in range(filter_depth):
                output[:, :, j, i] += np.squeeze(scipy.signal.convolve(input[:, :, k, i], params['W'][:, :, k, j], mode = 'valid'))
            output[:, :, j, i] += params['b'][j]
    # for imageIndex, image in enumerate(input):
        # for filterIndex, filter in enumerate(params['W']):
            # output[imageIndex][filterIndex] = scipy.signal.convolve(image, filter, mode = 'valid')
            # output[imageIndex][filterIndex] += params['b'][filterIndex]
    #######################

    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
#       Update dv_input and grad with values
        ###########################
        #dL/dW:
        #loop through every image in batch
        for i in range(batch_size):
            #loop for every filter
            for j in range(num_filters):
                #loop for every level
                for k in range(filter_depth):
                    #for every level within each filter, cross correlate between the image at the depth index and the output at the filter index
                    #sum for every image, since each level within each image is dependent on all weights in the filters
                    #also make sure to flip
                    grad['W'][:, :, k, j] += np.flip(scipy.signal.correlate(input[:, :, k, i], dv_output[:, :, j, i], mode = 'valid'), axis = (0,1))
        grad['W'] /= batch_size #normalize everything to batch size 
#dL/db
        #loop for every image
        for i in range(batch_size):
            #loop for every filter
            for j in range(num_filters):
                #dX/db for every point in dv_output is simply 0, so we just take the sum of dv_output corresponding to the specific filter
                #and then we sum it up for every image
                grad['b'][j] += np.sum(dv_output[:, :, j, i])
        grad['b'] /= batch_size #normalize everything to batch size
#dL/dI
        #sum loss over filters for each image to get derivative with respect to image
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(filter_depth):
                    dv_input[:, :, k, i] += scipy.signal.correlate(dv_output[:, :, j, i], params['W'][:, :, k, j],mode = 'full')
                    # for each level within each image, cross correlate between the output at the desired depth and the filter at the desired depth
                    # sum this for each depth level within the input, since each depth level in the input combines with a corresponding level in the filter to form a level in the output
        # no need to normalize–we get a dv_input for every image, since we'll use this to backpropogate further             
                
    return output, dv_input, grad
