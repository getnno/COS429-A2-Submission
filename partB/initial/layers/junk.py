# TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        ###########################
        #dL/dW:
        #loop through every image in batch
        for imageIndex, image in enumerate(output):
            #go through every filter
            for filterIndex, filter in enumerate([params['W']]):
                for depthIndex, depth in enumerate(input[imageIndex]):
                    # for each level within each filter, cross correlate between the image at the depth and the output. 
                    # Sum this for each depth level within each filter, since they all form the flat 2d output for each filter
                    #finally, flip
                    grad['W'][:, :, depthIndex, filterIndex, ] += np.flip(scipy.signal.correlate(input[:, :, depthIndex, imageIndex], dv_output[:, :, filterIndex, imageIndex], mode = 'valid'), axis = (0,1))
        grad['W'] /= batch_size #normalize everything to batch size
        #dL/dI
        #sum loss over filters for each image to get derivative with respect to image
        for imageIndex, image in enumerate(output):
            for filterIndex, filter in enumerate(params['W']):
                for depthIndex, depth in enumerate(input[imageIndex]):
                    # for each level within each image, cross correlate between the output at the desired depth and the filter at the desired depth
                    # sum this for each depth level within the input, since each depth level in the input combines with a corresponding level in the filter to form a level in the output
                    dv_input[:, :, depthIndex, imageIndex] += scipy.signal.correlate(dv_output[:, :, filterIndex, imageIndex], params['W'][:, :, depthIndex, filterIndex],mode = 'full')
        # no need to normalize–we get a dv_input for every image, since we'll use this to backpropogate further
        
        #dL/db
        for imageIndex, image in enumerate(output):
            for biasIndex, bias in enumerate(params['b']):
                grad['B'][biasIndex] += np.sum(dv_output[:, :, biasIndex, imageIndex])
        grad['B'] /= batch_size #normalize everything to batch size


# from inference.py
if (layer['type'] == 'pool'):
            if (index == 0):
                out, dv_input, grad = layer(input, layer['params'], layer['hyper_params'], False)
                activations[index] = out
            else:
                out, dv_input, grad = fn_pool(activations[index - 1], layer['params'], layer['hyper_params'], False)
                activations[index] = out

        if (layer['type'] == 'conv'):
            if (index == 0):
                out, dv_input, grad = fn_conv(input, layer['params'], layer['hyper_params'], False)
                activations[index] = out
            else:
                out, dv_input, grad = fn_conv(activations[index - 1], layer['params'], layer['hyper_params'], False)
                activations[index] = out

        if (layer['type'] == 'flatten'):
            if (index == 0):
                out, dv_input, grad = fn_flatten(input, layer['params'], layer['hyper_params'], False)
                activations[index] = out
            else:
                out, dv_input, grad = fn_flatten(activations[index - 1], layer['params'], layer['hyper_params'], False)
                activations[index] = out

        if (layer['type'] == 'linear'):
            if (index == 0):
                out, dv_input, grad = fn_linear(input, layer['params'], layer['hyper_params'], False)
                activations[index] = out
            else:
                out, dv_input, grad = fn_linear(activations[index - 1], layer['params'], layer['hyper_params'], False)
                activations[index] = out

        if (layer['type'] == 'relu'):
            if (index == 0):
                out, dv_input, grad = fn_relu(input, layer['params'], layer['hyper_params'], False)
                activations[index] = out
            else:
                out, dv_input, grad = fn_relu(activations[index - 1], layer['params'], layer['hyper_params'], False)
                activations[index] = out
        if (layer['type'] == 'softmax'):
            if (index == 0):
                out, dv_input, grad = fn_softmax(input, layer['params'], layer['hyper_params'], False)
                activations[index] = out
            else:
                out, dv_input, grad = fn_softmax(activations[index - 1], layer['params'], layer['hyper_params'], False)
                activations[index] = out

#from calc_gradient.py

#at the top
output, activations = inference(model, input)

#at the bottom
if (layer['type'] == 'pool'):
            if (index == num_layers - 1):
                dv_out = dv_output
            else:
                dv_out = grads[index + 1]
            if (index == 0):
                layer_in = input
            else:
                layer_in = activations[index - 1]
            out, dv_input, grad = fn_pool(layer_in, layer['params'], layer['hyper_params'], True, dv_output = dv_out)
            grads[index] = grad

        if (layer['type'] == 'conv'):
            if (index == num_layers - 1):
                dv_out = dv_output
            else:
                dv_out = grads[index + 1]
            if (index == 0):
                layer_in = input
            else:
                layer_in = activations[index - 1]
            out, dv_input, grad = fn_conv(layer_in, layer['params'], layer['hyper_params'], True, dv_output = dv_out)
            grads[index] = grad        
            if (layer['type'] == 'flatten'):
                if (index == num_layers - 1):
                    dv_out = dv_output
                else:
                    dv_out = grads[index + 1]
                if (index == 0):
                    layer_in = input
                else:
                    layer_in = activations[index - 1]
                out, dv_input, grad = fn_flatten(layer_in, layer['params'], layer['hyper_params'], True, dv_output = dv_out)
                grads[index] = grad        
            if (layer['type'] == 'linear'):
                if (index == num_layers - 1):
                    dv_out = dv_output
                else:
                    dv_out = grads[index + 1]
                if (index == 0):
                    layer_in = input
                else:
                    layer_in = activations[index - 1]
                out, dv_input, grad = fn_linear(layer_in, layer['params'], layer['hyper_params'], True, dv_output = dv_out)
                grads[index] = grad        
            if (layer['type'] == 'relu'):
                if (index == num_layers - 1):
                    dv_out = dv_output
                else:
                    dv_out = grads[index + 1]
                if (index == 0):
                    layer_in = input
                else:
                    layer_in = activations[index - 1]
                out, dv_input, grad = fn_relu(layer_in, layer['params'], layer['hyper_params'], True, dv_output = dv_out)
                grads[index] = grad        
            if (layer['type'] == 'softmax'):
                if (index == num_layers - 1):
                    dv_out = dv_output
                else:
                    dv_out = grads[index + 1]
                if (index == 0):
                    layer_in = input
                else:
                    layer_in = activations[index - 1]
                out, dv_input, grad = fn_softmax(layer_in, layer['params'], layer['hyper_params'], True, dv_output = dv_out)
                grads[index] = grad