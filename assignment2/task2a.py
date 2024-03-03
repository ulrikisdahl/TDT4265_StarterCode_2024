import numpy as np
import utils
import typing
import cProfile

np.random.seed(1)

 
def sigmoid(n: np.ndarray, use_improved_sigmoid: bool) -> np.ndarray:
    """
    Args:
        n: weighted sum batch
        use_improved_sigmoid:
    Returns:
        activation batch
    """
    if use_improved_sigmoid:
        return 1.7159*np.tanh(2*n/3)
    return 1 / (1 + np.exp(-n))


def sigmoid_dash(n: np.ndarray, use_improved_sigmoid: bool) -> np.ndarray:
    """
    Args:
        n: weighted sum batch
        use_improved_sigmoid:
    Returns:
        derivative of batch
    """
    if use_improved_sigmoid:
        return 1.14393 / pow(np.cosh(2*n/3),2)
    return np.exp(-n)/pow(np.exp(-n)+1, 2)


def relu(n: np.ndarray) -> np.ndarray:
    return n * (n > 0)


def relu_dash(n: np.ndarray) -> np.ndarray:
    return 1. * (n > 0)


def pre_process_images(X: np.ndarray) -> np.ndarray:
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    mean_pixel_value = np.sum(X) / X.size
    standard_deviation = np.std(X)
    print(mean_pixel_value)
    print(standard_deviation)
    x_norm = (X - mean_pixel_value) / standard_deviation
    batch_size = X.shape[0]

    return np.hstack((x_norm, np.ones(batch_size).reshape((batch_size, 1))))


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert (
        targets.shape == outputs.shape
    ), f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    # raise NotImplementedError

    return np.sum(-np.sum(targets * np.log(outputs), axis=1)) / targets.shape[0]


def c_softmax(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: input of shape [batch size, num_classes]
    Returns:
        Softmax output of shape [batch size, num_classes]
    """
    clip = 1e-15
    x = np.clip(x, -clip, None) #clip the inputs
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class SoftmaxModel:
    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 4 hyperparameter
    ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init
        self.hidden_layer_weighted_sum = list()
        self.hidden_layer_activation = list()
        self.delta_i = None

        #assert self.use_relu != use_improved_sigmoid
        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for i, (size) in enumerate(self.neurons_per_layer):
            w_shape = (prev, size)
            if self.use_improved_weight_init:
                sigma = 1/np.sqrt(w_shape[0])
                self.ws.append(sigma * np.random.standard_normal((prev, size)))
                # self.ws.append(np.random.normal(scale=sigma, size=w_shape))
            else:
                self.ws.append(np.random.uniform(-1, 1, w_shape)) #randomly sampled weights

            prev = size
        self.grads = [None for i in range(len(self.ws))]


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        self.hidden_layer_output = [X]

        #calculate hidden layers
        a = X
        #z = np.dot(X, self.ws[0]) #(b, hidden1)
        for layer_weight in self.ws[:-1]:
            z = np.dot(a, layer_weight) #weighted sum on current layer
            #a = np.vectorize(relu)(z) #activation on previous layer
            a = sigmoid(z, use_improved_sigmoid=self.use_improved_sigmoid)
            self.hidden_layer_output.append(z) #save the inputs to each layer (weighted sum pre-activations)
        self.last_hidden_activations = a #save the last hidden activations for computing gradients for the output layer

        #calculate softmax
        z = np.dot(a, self.ws[-1])
        softmax = c_softmax(z)
        
        return softmax

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"
        self.grads = []

        #Backprop
        delta = outputs - targets #(bs, out)
        gradientOutputs = np.dot(self.last_hidden_activations.T, delta) #(bs, hidden1).T * (bs, out) = (hidden1, out)
        self.grads.insert(0, gradientOutputs)

        for i in range(len(self.hidden_layer_output) - 1):
            z = self.hidden_layer_output[-(i + 1)] #(bs, hidden)
            layer_input = self.hidden_layer_output[-(i + 1) - 1] #(bs, input)

            weights = self.ws[-(i + 1)] #(hidden, out)
            d_relu = sigmoid_dash(z, self.use_improved_sigmoid) #(bs, hidden)

            error = np.dot(delta, weights.T) #(bs, out) * (hidden, out).T = (bs, hidden)
            delta = error * d_relu #(bs, hidden)
            gradient = np.dot(layer_input.T, delta) / X.shape[0] #(input, hidden)
            self.grads.insert(0, gradient)

        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    encode = np.zeros((Y.shape[0], num_classes), dtype=int)
    for i, (row) in enumerate(Y):
        encode[i, row[0]] = 1

    # raise NotImplementedError
    return encode


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            print(i)
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu
    )


    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    #gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    #cProfile.run('main()')
    main()


  
