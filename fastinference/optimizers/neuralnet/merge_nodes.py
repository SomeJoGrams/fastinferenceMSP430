from fastinference.models.nn.Conv2D import Conv2D
from fastinference.models.nn.MaxPool import MaxPool2d
from fastinference.models.nn.BatchNorm import BatchNorm
from fastinference.models.nn.Activations import Sigmoid, Step
from fastinference.models.nn.Reshape import Reshape
from fastinference.models.nn.Gemm import Gemm
from fastinference.models.nn.AveragePool import AvgPool2d
from fastinference.models.nn.Activations import LogSoftmax, LeakyRelu, Relu, Sigmoid, Sign

def optimize(model, **kwargs):
    """Merges subsequent BatchNorm and Step layers into a new Step layer with adapted thresholds in a single pass. Currently there is no recursive merging applied.

    TODO: Perform merging recursively. 

    Args:
        model (NeuralNet): The NeuralNet model.

    Returns:
        NeuralNet: The NeuralNet model with merged layers.
    """
    # Merge BN + Step layers for BNNs
    new_layers = []
    last_layer = None
    print("starting to optimize")
    for layer_id, layer in enumerate(model.layers):
        if last_layer is not None:
            if isinstance(last_layer, BatchNorm) and isinstance(layer, Step):
                # print("prevthreshold", layer.threshold)
                layer.threshold = layer.threshold -last_layer.bias / last_layer.scale
                # print("altered threshold", layer.threshold, last_layer.bias,last_layer.scale)
                # for ind,threshold in enumerate(layer.threshold):
                    # if (last_layer.scale[ind] < 0):
                    #     layer.threshold[ind] = layer.threshold[ind] * -1 
                # maybe use this as a multiplication version, 
                layer.signs = []
                for scale in last_layer.scale:
                    if (scale >= 0):
                        layer.signs.append(1)
                    else:
                        layer.signs.append(0)
            else:
                new_layers.append(last_layer)
        last_layer = layer
        
        
    new_layers.append(last_layer)
    model.layers = new_layers

    return model