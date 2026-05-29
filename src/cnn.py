import os
import math
import numpy as np
import cv2
from pathlib import Path
from scipy import signal
import pickle
import random

import dataset

center_crop_size = 256
BATCHSIZE = 50                 # images per batch
SHAPE     = (3, center_crop_size, center_crop_size)      # (channels, height, width) 
KERNELSIZE = 5                 # conv kernel is KERNELSIZE x KERNELSIZE
DEPTH      = 32                # number of conv filters (output channels)


CLASSES   = ["NONE", "LOW", "MEDIUM", "HIGH"]
MATERIALS = ["foam", "bitumen", "aluminium", "eps"]


# ---------------------------- Layers ----------------------------- #

class Convolutional:

    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth

        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) * 0.1
        self.biases = np.zeros((depth, 1, 1))

        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1
        self.output_shape = (depth, self.output_height, self.output_width)

    def forward(self, data):
        self.input = data
        self.output = np.zeros(self.output_shape)
        for filters in range(self.depth):
            for colours in range(self.input_depth):
                self.output[filters] += signal.correlate2d(
                    data[colours], self.kernels[filters, colours], "valid"
                )
        self.output += self.biases
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for filters in range(self.depth):
            for colours in range(self.input_depth):
                kernels_gradient[filters, colours] = signal.correlate2d(
                    self.input[colours], output_gradient[filters], "valid"
                )
                input_gradient[colours] += signal.convolve2d(
                    output_gradient[filters], self.kernels[filters, colours], "full"
                )

        # update weights
        self.kernels -= learning_rate * kernels_gradient
        bias_gradient = np.sum(output_gradient, axis=(1, 2), keepdims=True) 
        self.biases -= learning_rate * bias_gradient
        return input_gradient


class Dense:
   
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.bias = np.random.randn(output_size, 1) * 0.1

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = np.sum(output_gradient, axis=1, keepdims=True)
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient


class Reshape:

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate=None):
        return np.reshape(output_gradient, self.input_shape)


class Activation:

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, data):
        self.input = data
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * self.activation_prime(self.input)


class ReLU:

    def forward(self, data):
        self.input = data
        return np.maximum(0, data)

    def backward(self, grad, lr=None):
        return grad * (self.input > 0)


class MaxPooling:

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.max_positions = None
        self.input_shape = None

    def forward(self, data):
        self.input = data
        self.input_shape = data.shape
        channels, h, w = data.shape

        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        self.output = np.zeros((channels, out_h, out_w))
        self.max_positions = np.zeros_like(data, dtype=bool)

        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    start_h = i * self.stride
                    start_w = j * self.stride
                    window = data[c, start_h:start_h + self.pool_size,
                                     start_w:start_w + self.pool_size]

                    max_idx = np.argmax(window)
                    max_pos_h = max_idx // self.pool_size
                    max_pos_w = max_idx % self.pool_size

                    self.max_positions[c, start_h + max_pos_h, start_w + max_pos_w] = True
                    self.output[c, i, j] = np.max(window)

        return self.output

    def backward(self, output_gradient, learning_rate=None):
        input_gradient = np.zeros_like(self.input)
        channels, out_h, out_w = output_gradient.shape

        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    start_h = i * self.stride
                    start_w = j * self.stride
                    mask = self.max_positions[c, start_h:start_h + self.pool_size,
                                                 start_w:start_w + self.pool_size]
                    input_gradient[c, start_h:start_h + self.pool_size,
                                      start_w:start_w + self.pool_size][mask] = output_gradient[c, i, j]

        return input_gradient


class GroupedSoftmax:

    def __init__(self, n_groups=4, n_classes=4):
        self.n_groups = n_groups
        self.n_classes = n_classes

    def forward(self, x):                              
        z = x.reshape(self.n_groups, self.n_classes)
        z = z - np.max(z, axis=1, keepdims=True)       
        e = np.exp(z)
        self.output = (e / np.sum(e, axis=1, keepdims=True)).reshape(-1, 1)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        return output_gradient


# ---------------------- Loss Functions ---------------------------#

def cce(y_true, y_pred):
    # Categorical cross-entropy
    return -np.sum(y_true * np.log(y_pred + 1e-9))


def cce_prime(y_true, y_pred):
    return y_pred - y_true


#--------------------------- Model----------------------------------#


def build_network():
   
    conv = Convolutional(SHAPE, KERNELSIZE, DEPTH)
    pooled_h = (conv.output_height - 2) // 2 + 1
    pooled_w = (conv.output_width - 2) // 2 + 1
    flat = DEPTH * pooled_h * pooled_w

    return [
        conv,
        ReLU(),
        MaxPooling(pool_size=2, stride=2),
        Reshape((DEPTH, pooled_h, pooled_w), (flat, 1)),
        Dense(flat, len(MATERIALS) * len(CLASSES)),      
        GroupedSoftmax(n_groups=len(MATERIALS), n_classes=len(CLASSES)),
    ]

def predict(network, x):
    """Run one image (C, H, W) forward through every layer."""
    out = x
    for layer in network:
        out = layer.forward(out)
    return out                                  
 
 
def encode_label(label_dict):
    """Turn {'foam': 'HIGH', ...} into a one-hot (16, 1) target column."""
    vec = np.zeros((len(MATERIALS) * len(CLASSES), 1))
    for m, material in enumerate(MATERIALS):
        value = str(label_dict[material]).upper()
        if value in CLASSES:                    
            vec[m * len(CLASSES) + CLASSES.index(value), 0] = 1.0
    return vec
 
 
def decode_prediction(output):
    """Turn a (16, 1) output into {'foam': 'HIGH', ...} via argmax per material."""
    z = output.reshape(len(MATERIALS), len(CLASSES))
    return {mat: CLASSES[int(np.argmax(z[m]))] for m, mat in enumerate(MATERIALS)}


# -------------------------- Data loading -------------------------------

def load_batch(path, batch, batch_size):
    images, labels = [], []
    for file in sorted(path.glob("*.jpg"))[batch * batch_size:(batch + 1) * batch_size]:
        img = cv2.imread(str(file))
        img = np.array(img, dtype=np.float32) / 255.0   # normalize to [0, 1]
        img = img.transpose(2, 0, 1)                     # (C, H, W)
        images.append(img)

        parts = file.stem.split("_")
        labels.append({
            "foam":      parts[parts.index("FOAM") + 1],
            "bitumen":   parts[parts.index("BITUMEN") + 1],
            "aluminium": parts[parts.index("AL") + 1],
            "eps":       parts[parts.index("EPS") + 1],
        })

    data = np.array(images)
    print(f"Loaded batch {batch} from {path}.")
    return data, labels


def count_batches(path):
    imgnum = sum(1 for f in os.listdir(path) if f.endswith(".jpg"))
    batchnum = math.ceil(imgnum / BATCHSIZE)
    print(f"There are {batchnum} batches in {path}.\n")
    return batchnum

# ----------------- Train / test / validate ---------------------------------

def show_random_image(network, path, set_name):
    
    n_batches = count_batches(path)
    batch = random.randrange(n_batches)
    data, labels = load_batch(path, batch, BATCHSIZE)
 
    i = random.randrange(len(data))
    image, true = data[i], labels[i]
 
    output = predict(network, image)                      
    probs = output.reshape(len(MATERIALS), len(CLASSES))    
    pred = decode_prediction(output)
 
    print(f"\n=== Random {set_name} image: batch {batch}, position {i} ===")
    print(f"{'material':<11}{'predicted':<9}{'true':<9}" + "".join(f"{c:>8}" for c in CLASSES))
 
    correct = 0
    for m, material in enumerate(MATERIALS):
        true_val = str(true[material]).upper()
        if pred[material] == true_val:
            correct += 1
        row = "".join(f"{probs[m, c]:>8.2f}" for c in range(len(CLASSES)))
        print(f"{material:<11}{pred[material]:<9}{true_val:<9}{row}")
 
    loss = cce(encode_label(true), output)
    print(f"loss: {loss:.3f}   |   correct heads: {correct}/{len(MATERIALS)}")
    return pred
 
 
def train(training_path, network, epochs=10, learning_rate=0.01):
    n_batches = count_batches(training_path)
 
    for epoch in range(epochs):
        running_loss = 0.0
        correct_heads = 0
        num_images = 0
 
        for batch in range(n_batches):
            data, labels = load_batch(training_path, batch, BATCHSIZE)
 
            for image, label in zip(data, labels):
                output = predict(network, image)           
                y_true = encode_label(label)
 
                running_loss += cce(y_true, output)        
                pred = decode_prediction(output)
                correct_heads += sum(pred[m] == str(label[m]).upper() for m in MATERIALS)
                num_images += 1
 
                grad = cce_prime(y_true, output)            
                for layer in reversed(network):            
                    grad = layer.backward(grad, learning_rate)
 
        avg_loss = running_loss / max(num_images, 1)
        head_acc = correct_heads / max(num_images * len(MATERIALS), 1)
        print(f"epoch {epoch + 1}/{epochs}  avg loss {avg_loss:.3f}  head accuracy {head_acc:.3f}")
 
    return network
 
 
def test(testing_path, network):
    show_random_image(network, testing_path, "test")
 
 
def val(validating_path, network):
    show_random_image(network, validating_path, "validation")
 
 
def save_network(network, path="model.pkl"):
    params = []
    for layer in network:
        if isinstance(layer, Convolutional):
            params.append({"kernels": layer.kernels, "biases": layer.biases})
        elif isinstance(layer, Dense):
            params.append({"weights": layer.weights, "bias": layer.bias})
        else:
            params.append(None)                     
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"Successfully saved model to {path}.")

def load_network(path="model.pkl"):
    network = build_network()                       
    with open(path, "rb") as f:
        params = pickle.load(f)
    for layer, p in zip(network, params):
        if isinstance(layer, Convolutional):
            layer.kernels, layer.biases = p["kernels"], p["biases"]
        elif isinstance(layer, Dense):
            layer.weights, layer.bias = p["weights"], p["bias"]
    print("Loaded model successfully.")
    return network

if __name__ == "__main__":
    np.random.seed(0)      
    random.seed(0)
 
    network = build_network()                                       
    train(Path("data/train"), network, epochs=5, learning_rate=0.01) 
    test(Path("data/test"), network)                               
 
