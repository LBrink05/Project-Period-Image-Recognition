import numpy as np
from scipy import signal 
from pathlib import Path 


# 1. CNN class(forward+backward)
# ============================================================================
class CNN: 
    def __init__(self, input_shape, kernel_size, depth) : 
        self.input_depth, self.input_height, self.input_width = input_shape  #the image 
        self.kernel_size = kernel_size
        self.depth = depth

        self.kernels_shape = (depth, self.input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(depth, self.input_depth, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros((depth, 1,1)) 


        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1
        self.output_shape = (depth, self.output_height, self.output_width) 
 

    def forward(self, data): 
        self.input = data
        self.output = np.zeros(self.output_shape)

        for filters in range(self.depth):
         for colours in range (self.input_depth):

            cross_correlation = signal.correlate2d(data[colours], self.kernels[filters,colours], "valid")
            self.output += cross_correlation 
        self.output += self.biases
        


        return self.output




    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)  
        input_gradient = np.zeros(self.input_shape)
        
    
        for filters in range(self.depth):
            for colours in range(self.input_depth):
                kernels_gradient[filters, colours] = signal.correlate2d(
                    self.input[colours],
                    output_gradient[filters],
                    mode="valid"
                )
                input_gradient[colours] += signal.convolve2d(
                    output_gradient[filters],
                    self.kernels[filters, colours],
                    mode="full"
                )
        #this is where we update the weigths aka the modle learns
        self.kernels -= learning_rate * kernels_gradient
        bias_gradient = np.sum(output_gradient, axis=(1, 2), keepdims=True)   
        self.biases -= learning_rate * bias_gradient.reshape(-1, 1)
        
        return input_gradient






# 2. Dense class(forward+backward)
# ============================================================================

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1  # (out, in)
        self.bias = np.random.randn(output_size, 1) * 0.1              # (out, 1)
    
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


# 3. Layer class
# ============================================================================
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass




# 4. Reshape class
# ============================================================================

class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


# 5. Activation class
# ============================================================================

class Activation:
    
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None
    
    def forward(self, data):
        self.input = data
        self.output = self.activation(self.input)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * self.activation_prime(self.input)






# 6. Sigmoid class
# ============================================================================

class Sigmoid:
    def __init__(self):
        def sigmoid(x): return 1/(1+np.exp(-np.clip(x)))
        def sigmoid_prime(x): s=sigmoid(x); return s*(1-s)
        super().__init__(sigmoid, sigmoid_prime)

test = CNN(input_shape=(4, 384, 384), kernel_size=5, depth=16)




# 7. ReLu class (instead of Sigmoid cause it is more fast and efficent for big sizes like ours????)
# ============================================================================

class ReLU:
    def forward(self, data):
        self.input = data
        return np.maximum(0, data)
    
    def backward(self, grad, lr=None):
        return grad * (self.input > 0)


# 8. MaxPooling class (also extremly useful for bigsizes like ours because its gonna take the max from a 2x2 square basscily its gonna decrease the sie of the image for faster processing)
# ============================================================================

class MaxPooling:
    
    def __init__(self, pool_size=2, stride=2):
        super().__init__()
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
                    window = data[
                        c,
                        start_h:start_h + self.pool_size,
                        start_w:start_w + self.pool_size
                    ]
                    
            
                    max_val = np.max(window)
                    max_idx = np.argmax(window)
                    max_pos_h = max_idx // self.pool_size
                    max_pos_w = max_idx % self.pool_size
                
                    self.max_positions[
                        c,
                        start_h + max_pos_h,
                        start_w + max_pos_w
                    ] = True
                    
                    self.output[c, i, j] = max_val
        
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        input_gradient = np.zeros_like(self.input)
        channels, out_h, out_w = output_gradient.shape
        
        for c in range(channels):
            for i in range(out_h):
                for j in range(out_w):
                    start_h = i * self.stride
                    start_w = j * self.stride
                    
                    mask = self.max_positions[
                        c,
                        start_h:start_h + self.pool_size,
                        start_w:start_w + self.pool_size
                    ]
                    
                    input_gradient[
                        c,
                        start_h:start_h + self.pool_size,
                        start_w:start_w + self.pool_size
                    ][mask] = output_gradient[c, i, j]
        
        return input_gradient
    


def load_dataset(folder: Path):
    images, labels = [], []
    for file in sorted(folder.glob("*.jpg")):
        img = cv2.imread(str(file))
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img = img.transpose(2, 0, 1)  # Change to (C, H, W) format
        images.append(img)

        parts = file.stem.split("_")
        label = {
            "foam":      parts[parts.index("FOAM") + 1],
            "bitumin":   parts[parts.index("BITUMEN") + 1],
            "aluminium": parts[parts.index("AL") + 1],
            "eps":       parts[parts.index("EPS") + 1],
        }
        labels.append(label)

    data = np.array(images)
    return data, labels

def train(path):
    pass

def test(path):
    pass

def val(path):
    pass