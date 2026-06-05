import os
import math
import numpy as np
import cv2
from pathlib import Path
import pickle
import random
from colorama import Fore, Style, Back
import dataset
import matplotlib.pyplot as plt
import time
import output
from collections import defaultdict
import shutil

center_crop_size = 128
BATCHSIZE = 50                                           # images per batch
SHAPE     = (3, center_crop_size, center_crop_size)      # (channels, height, width)
KERNELSIZE = 5                                           # conv kernel is KERNELSIZE x KERNELSIZE
DEPTH      = 32                                          # number of conv filters (output channels)


CLASSES   = ["NONE", "LOW", "MEDIUM", "HIGH"]
MATERIALS = ["foam", "bitumen", "aluminium", "eps"]

terminal_char_len = shutil.get_terminal_size(fallback=(80, 24)).columns

# Convention: spatial tensors are (N, C, H, W); the flattened activations that
# flow through Dense / Softmax are (features, N) so the Dense weight matrix
# keeps its (out, in) orientation.  Parameter gradients are averaged over the
# batch (the `/ N`); activation gradients passed backward stay per-sample.

class Convolutional:

    def __init__(self, input_shape, kernel_size, depth):
        self.input_shape = input_shape
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth

        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        fan_in = self.input_depth * kernel_size * kernel_size
        self.kernels = (np.random.randn(*self.kernels_shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        self.biases = np.zeros((depth, 1, 1), dtype=np.float32)

        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1
        self.output_shape = (depth, self.output_height, self.output_width)

    def forward(self, data): 
                   
        self.input = data
        N = data.shape[0]
        C, k = self.input_depth, self.kernel_size
        oh, ow = self.output_height, self.output_width

        windows = np.lib.stride_tricks.sliding_window_view(data, (k, k), axis=(2, 3))
        self.cols = windows.transpose(0, 1, 4, 5, 2, 3).reshape(N, C * k * k, oh * ow)

        W_row = self.kernels.reshape(self.depth, C * k * k)     
        out = W_row @ self.cols
        self.output = out.reshape(N, self.depth, oh, ow) + self.biases


        return self.output

    def backward(self, output_gradient, learning_rate):
        N = output_gradient.shape[0]
        C, k = self.input_depth, self.kernel_size
        oh, ow = self.output_height, self.output_width
        dY = output_gradient.reshape(N, self.depth, -1)
        W_row = self.kernels.reshape(self.depth, C * k * k)

        kernels_gradient = np.matmul(dY, self.cols.transpose(0, 2, 1)).sum(axis=0)
        kernels_gradient = kernels_gradient.reshape(self.kernels_shape) / N
        bias_gradient = dY.sum(axis=(0, 2)).reshape(self.depth, 1, 1) / N

        # input gradient: dCols = W^T @ dY, then fold patches back (col2im)
        dcols = (W_row.T @ dY).reshape(N, C, k, k, oh, ow)
        dX = np.zeros((N, C, self.input_height, self.input_width), dtype=output_gradient.dtype)
        for a in range(k):
            for b in range(k):
                dX[:, :, a:a + oh, b:b + ow] += dcols[:, :, a, b]

        self.kernels -= learning_rate * kernels_gradient
        self.biases  -= learning_rate * bias_gradient
        return dX                             # first layer -> input grad discarded


class Dense:

    def __init__(self, input_size, output_size):
        self.weights = (np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.bias = (np.random.randn(output_size, 1) * 0.1).astype(np.float32)

    def forward(self, input):                      
        self.input = input
        return self.weights @ input + self.bias   

    def backward(self, output_gradient, learning_rate): 
        N = output_gradient.shape[1]
        weights_gradient = (output_gradient @ self.input.T) / N
        bias_gradient = output_gradient.mean(axis=1, keepdims=True)
        input_gradient = self.weights.T @ output_gradient       

        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        return input_gradient


class Reshape:
    

    def __init__(self, input_shape=None, output_shape=None):
        pass

    def forward(self, input):                    
        self.in_shape = input.shape
        return input.reshape(input.shape[0], -1).T 

    def backward(self, output_gradient, learning_rate=None):   
        return output_gradient.T.reshape(self.in_shape)


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

    def forward(self, data):
        self.in_shape = data.shape
        N, C, H, W = data.shape
        p = self.pool_size
        oh, ow = H // p, W // p
        base = data[:, :, :oh * p, :ow * p]
        cells = np.stack([base[:, :, i::p, j::p] for i in range(p) for j in range(p)], axis=0)
        self.argmax = cells.argmax(axis=0).astype(np.uint8) 
        return cells.max(axis=0)

    def backward(self, output_gradient, learning_rate=None):
        N, C, H, W = self.in_shape
        p = self.pool_size
        oh, ow = H // p, W // p
        input_gradient = np.zeros(self.in_shape, dtype=output_gradient.dtype)
        for k in range(p * p):
            i, j = divmod(k, p)
            input_gradient[:, :, i:oh * p:p, j:ow * p:p] = output_gradient * (self.argmax == k)
        return input_gradient

class GroupedSoftmax:

    def __init__(self, n_groups=4, n_classes=4):
        self.n_groups = n_groups
        self.n_classes = n_classes

    def forward(self, x):                         
        g, c, N = self.n_groups, self.n_classes, x.shape[1]
        z = x.reshape(g, c, N)
        z = z - np.max(z, axis=1, keepdims=True)  
        e = np.exp(z)
        self.output = (e / np.sum(e, axis=1, keepdims=True)).reshape(g * c, N)
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        return output_gradient                     


# ---------------------- Loss Functions ---------------------------#

def cce(y_true, y_pred):
    # Categorical cross-entropy, summed over the whole batch
    return -np.sum(y_true * np.log(y_pred + 1e-9))



def cce_prime(y_true, y_pred, weights=None):
    g = y_pred - y_true
    if weights is not None:
        nm, nc = weights.shape
        per_head_w = (y_true.reshape(nm, nc, -1) * weights[:, :, None]).sum(axis=1)  # true-class weight, (nm, N)
        g = g * np.repeat(per_head_w, nc, axis=0)                                    # scale each head block
    return g


#--------------------------- Model----------------------------------#

def build_network():
    conv1 = Convolutional(SHAPE, KERNELSIZE, DEPTH)                # (3,128,128) -> (32,124,124)
    h1, w1 = conv1.output_height // 2, conv1.output_width // 2     # 62, 62 after pool
    conv2 = Convolutional((DEPTH, h1, w1), KERNELSIZE, DEPTH)      # (32,62,62) -> (32,58,58)
    h2, w2 = conv2.output_height // 2, conv2.output_width // 2     # 29, 29 after pool
    flat = DEPTH * h2 * w2
    return [
        conv1, ReLU(), MaxPooling(2, 2),
        conv2, ReLU(), MaxPooling(2, 2),
        Reshape(),
        Dense(flat, len(MATERIALS) * len(CLASSES)),
        GroupedSoftmax(len(MATERIALS), len(CLASSES)),
    ]


def predict(network, x):
    out = x
    for layer in network:
        out = layer.forward(out)
    return out


def encode_label(label_dict):

    vec = np.zeros((len(MATERIALS) * len(CLASSES), 1), dtype=np.float32)
    for m, material in enumerate(MATERIALS):
        value = str(label_dict[material]).upper()
        if value in CLASSES:
            vec[m * len(CLASSES) + CLASSES.index(value), 0] = 1.0
    return vec


def encode_labels(labels):
    nm, nc = len(MATERIALS), len(CLASSES)
    vec = np.zeros((nm * nc, len(labels)), dtype=np.float32)
    for j, ld in enumerate(labels):
        for m, material in enumerate(MATERIALS):
            value = str(ld[material]).upper()
            if value in CLASSES:
                vec[m * nc + CLASSES.index(value), j] = 1.0
    return vec


def decode_prediction(output):
    z = output.reshape(len(MATERIALS), len(CLASSES))
    return {mat: CLASSES[int(np.argmax(z[m]))] for m, mat in enumerate(MATERIALS)}


# -------------------------- Data loading -------------------------------#

def _parse_label(file):
    parts = file.stem.split("_")
    return {
        "foam":      parts[parts.index("FOAM") + 1],
        "bitumen":   parts[parts.index("BITUMEN") + 1],
        "aluminium": parts[parts.index("AL") + 1],
        "eps":       parts[parts.index("EPS") + 1],
    }


def load_dataset(path):
    files = sorted(Path(path).glob("*.jpg"))
    images, labels = [], []
    for file in files:
        img = cv2.imread(str(file))
        img = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)  # (C, H, W)
        images.append(img)
        labels.append(_parse_label(file))
    data = np.asarray(images, dtype=np.float32)
    print(f"Loaded {len(data)} images from {path} into memory.")
    return data, labels


def load_batch(path, batch, batch_size):
    files = sorted(Path(path).glob("*.jpg"))[batch * batch_size:(batch + 1) * batch_size]
    images, labels = [], []
    for file in files:
        img = cv2.imread(str(file))
        img = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        images.append(img)
        labels.append(_parse_label(file))
    return np.asarray(images, dtype=np.float32), labels


def count_batches(path):
    imgnum = sum(1 for f in os.listdir(path) if f.endswith(".jpg"))
    batchnum = math.ceil(imgnum / BATCHSIZE)
    return batchnum


# ----------------- Train / Test / Validate --------------------------------#

def macro_recall(cms):
    """cms: list of (nc x nc) confusion matrices, cm[true, pred].
    Returns (per_material_macro (nm,), overall_macro float, per_class list)."""
    per_material, per_class_all = [], []
    for cm in cms:
        support = cm.sum(axis=1)                       # true count per class (row sums)
        with np.errstate(invalid="ignore", divide="ignore"):
            recall = np.diag(cm) / support             # nan where support == 0
        present = support > 0
        per_class_all.append(recall)
        per_material.append(recall[present].mean() if present.any() else np.nan)
    per_material = np.array(per_material)
    overall = np.nanmean(per_material)                 # mean over heads, skipping empty ones
    return per_material, overall, per_class_all

def augment(batch):                       
    out = batch.copy()
    N = out.shape[0]
    flip = np.random.rand(N) < 0.5        
    out[flip] = out[flip, :, :, ::-1]
    out *= np.random.uniform(0.9, 1.1, size=(N, 1, 1, 1)).astype(out.dtype) 
    return np.clip(out, 0.0, 1.0)


def show_random_image(network, path, set_name):
    n_batches = count_batches(path)
    batch = random.randrange(n_batches)
    data, labels = load_batch(path, batch, BATCHSIZE)

    i = random.randrange(len(data))
    image, true = data[i], labels[i]

    output = predict(network, image[None])          # add batch dim -> (16, 1)
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
    return correct, loss

def test_image(network, frame):

    output = predict(network, frame)
    probs = output.reshape(len(MATERIALS), len(CLASSES))
    pred = decode_prediction(output)

    pred = decode_prediction(predict(network, frame))
    print("-"*terminal_char_len)
    print("Prediction: " + "   ".join(f"{mat}={pred[mat]}" for mat in MATERIALS))
    
    return pred
    

def train(training_path, validating_path, network, epochs=25, learning_rate=0.0005, batch_size=BATCHSIZE):
    val_data, val_labels = load_dataset(validating_path)
    data, labels = load_dataset(training_path)
    N = len(data)
    nm, nc = len(MATERIALS), len(CLASSES)
    idx = np.arange(N)
    best_val = -1.0
    head_acc_list = []
    avg_loss_list = []
    val_correct_list = []
    val_loss_list = []
    timediagnostic = 0

    query = input("Do you wish to train on a single batch? (y/n)")
    if query == 'y' or query == 'yes':
        sel = np.random.choice(N, batch_size, replace=False)
        data = data[sel]
        labels = [labels[i] for i in sel]
        N = len(data)
        idx = np.arange(N)
        print(f"Single-batch mode: training on {N} images.")

    # from TRAINING labels, count per (material, class):
    counts = encode_labels(labels).reshape(nm, nc, -1).sum(axis=2)               # (nm, nc)
    weights = counts.sum(axis=1, keepdims=True) / (nc * np.maximum(counts, 1))   # inverse-freq, (nm, nc)

    print(Fore.YELLOW + "Beginning training...\n" + Style.RESET_ALL)

    for epoch in range(int(epochs)):
        np.random.shuffle(idx)
        running_loss = 0.0
        correct_heads = 0
        seen = 0

        for start in range(0, N, batch_size):
            sel = idx[start:start + batch_size]
            xb = data[sel]
            xb = augment(xb)
            yb = encode_labels([labels[i] for i in sel])

            timediagnostic += 1
            diag = (timediagnostic == 10)

            if diag:
                fwd_t, bwd_t = defaultdict(float), defaultdict(float)
                out = xb
                for i, layer in enumerate(network):
                    t = time.perf_counter()
                    out = layer.forward(out)
                    fwd_t[f"{i}:{type(layer).__name__}"] += time.perf_counter() - t
            else:
                out = predict(network, xb)

            running_loss += cce(yb, out)
            pred_idx = out.reshape(nm, nc, -1).argmax(axis=1)
            true_idx = yb.reshape(nm, nc, -1).argmax(axis=1)
            correct_heads += int((pred_idx == true_idx).sum())
            seen += nm * xb.shape[0]

            grad = cce_prime(yb, out, np.clip(weights, 0.3, 4.0))
            if diag:
                for layer in reversed(network):
                    t = time.perf_counter()
                    grad = layer.backward(grad, learning_rate)
                    bwd_t[type(layer).__name__] += time.perf_counter() - t
                print("Warm-batch (10th) diagnostics:")
                print("forward :", dict(fwd_t))
                print("backward:", dict(bwd_t))
            else:
                for layer in reversed(network):
                    grad = layer.backward(grad, learning_rate)

        avg_loss = running_loss / max(N, 1)
        head_acc = correct_heads / max(seen, 1)

        print(f"Epoch {epoch + 1}/{epochs}  Avg Loss {avg_loss:.3f}  Head Accuracy {head_acc:.3f}")
        head_acc_list.append(head_acc)
        avg_loss_list.append(avg_loss)

        val_macro, val_loss, val_head_acc = validate(network, val_data, val_labels)
        val_correct_list.append(val_macro)          # plot macro recall instead of head acc
        val_loss_list.append(val_loss)

        if val_macro > best_val:
            best_val = val_macro
            save_network(network, "best_model.pkl")
            print(Fore.GREEN + f"  new best macro recall {val_macro:.3f} -> saved" + Style.RESET_ALL)

    query = input("Do you wish to see the training diagnostics? (y/n)")
    if query == 'y' or query == 'yes':
        output.show_training(epochs, head_acc_list, avg_loss_list, val_correct_list, val_loss_list)

    network = load_network(network, "best_model.pkl")   # restore best-macro weights before returning
    return network

def test(testing_path, network, batch_size=BATCHSIZE):

    data, labels = load_dataset(testing_path)

    query = input("Do you wish to compute confusion matrices? (y/n): ")

    if query == 'y' or query == 'yes':


        nm, nc = len(MATERIALS), len(CLASSES)
        cms = [np.zeros((nc, nc), dtype=int) for _ in range(nm)]
        N = len(data)

        #confusion matrix of all materials (split up later in the code for plotting)

        """ example of perfect confusion matrix: 
            Foam   | HIGH MEDIUM LOW NONE (true labels)
            ---------------------------
            HIGH   |  x     0     0   0
            MEDIUM |  0     x     0   0
            LOW    |  0     0     x   0 
            NONE   |  0     0     0   x


            going to do: MATERIAL1 MATERIAL2
                         MATERIAL3 MATERIAL4
        """

        for s in range(0, N, batch_size):
            xb = data[s:s + batch_size]
            yb = encode_labels(labels[s:s + batch_size])
            out = predict(network, xb)                       # whole batch at once                   
            predict_idx = out.reshape(nm, nc, -1).argmax(axis=1)  
            true_idx = yb.reshape(nm, nc, -1).argmax(axis=1) 
            for m in range(nm):
                np.add.at(cms[m], (true_idx[m], predict_idx[m]), 1)       # cms[m][t, p] += 1

        for m, mat in enumerate(MATERIALS):
            print(f"\n{mat}  (rows = true, cols = predicted)")
            print("        " + "".join(f"{c:>8}" for c in CLASSES))
            for t in range(nc):
                print(f"{CLASSES[t]:>8}" + "".join(f"{cms[m][t, p]:>8}" for p in range(nc)))

        per_mat, macro, _ = macro_recall(cms)
        print("\nMacro recall: " + "  ".join(f"{m}={r:.3f}" for m, r in zip(MATERIALS, per_mat)))
        print(f"Overall macro recall: {macro:.3f}")

        output.confusion_matrices(cms, MATERIALS, CLASSES)

def validate(network, val_data, val_labels, batch_size=BATCHSIZE):
    nm, nc = len(MATERIALS), len(CLASSES)
    N = len(val_data)
    total_loss = 0.0
    cms = [np.zeros((nc, nc), dtype=np.int64) for _ in range(nm)]
    for s in range(0, N, batch_size):
        xb = val_data[s:s + batch_size]
        yb = encode_labels(val_labels[s:s + batch_size])
        out = predict(network, xb)
        total_loss += cce(yb, out)
        pred_idx = out.reshape(nm, nc, -1).argmax(axis=1)
        true_idx = yb.reshape(nm, nc, -1).argmax(axis=1)
        for m in range(nm):
            np.add.at(cms[m], (true_idx[m], pred_idx[m]), 1)

    avg_loss  = total_loss / N
    head_acc  = sum(np.trace(cm) for cm in cms) / (nm * N)   # your old "avg correct heads" / nm
    per_mat, macro, _ = macro_recall(cms)
    print(f"Avg val Loss: {avg_loss:.3f}   |   head acc: {head_acc:.3f}   |   macro recall: {macro:.3f}")
    print("  per-material recall: " + "  ".join(f"{m}={r:.2f}" for m, r in zip(MATERIALS, per_mat)))
    print("." * terminal_char_len)
    return macro, avg_loss, head_acc

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
   


def load_network(network, model="model.pkl"):

    matches = sorted(Path(".").glob(model))
    if not matches:
        print(Fore.RED + "Failed to find model file. Model unchanged!" + Style.RESET_ALL)
        return network
    else:
        print(Fore.GREEN + f"Found model file {model}" + Style.RESET_ALL)

    network = build_network()
    path = matches[-1]   
    with open(path, "rb") as f:
        params = pickle.load(f)
    for layer, p in zip(network, params):
        if isinstance(layer, Convolutional):
            layer.kernels, layer.biases = p["kernels"], p["biases"]
        elif isinstance(layer, Dense):
            layer.weights, layer.bias = p["weights"], p["bias"]
    
    return network


