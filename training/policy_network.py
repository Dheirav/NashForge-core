"""
Policy network implementation for poker AI.

A simple feedforward neural network that maps game state to action logits.
No PyTorch/TensorFlow dependency - pure NumPy for evolution compatibility.
Optimized with potential Numba JIT compilation for 2-3× speedup.
"""
import numpy as np
from typing import List, Optional, Tuple
from .config import NetworkConfig

# Try to import Numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True, fastmath=True)
def relu_jit(x: np.ndarray) -> np.ndarray:
    """ReLU activation - JIT compiled."""
    return np.maximum(0, x)


@jit(nopython=True, cache=True, fastmath=True)
def tanh_jit(x: np.ndarray) -> np.ndarray:
    """Tanh activation - JIT compiled."""
    return np.tanh(x)


@jit(nopython=True, cache=True, fastmath=True)
def softmax_jit(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax with temperature - JIT compiled."""
    if temperature <= 0:
        temperature = 1e-8
    scaled = x / temperature
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals) + 1e-10)


@jit(nopython=True, cache=True, fastmath=True)
def forward_pass_jit(x, weights_tuple, biases_tuple):
    """
    JIT-compiled forward pass through network.
    
    Args:
        x: Input features (1D array)
        weights_tuple: Tuple of weight matrices
        biases_tuple: Tuple of bias vectors
        
    Returns:
        Output logits
    """
    # Hidden layers with ReLU activation
    for i in range(len(weights_tuple) - 1):
        x = x @ weights_tuple[i] + biases_tuple[i]
        x = np.maximum(0, x)  # ReLU
    
    # Output layer (no activation)
    x = x @ weights_tuple[-1] + biases_tuple[-1]
    
    return x


@jit(nopython=True, cache=True, fastmath=True, parallel=False)
def forward_batch_jit(x_batch, weights_tuple, biases_tuple):
    """
    JIT-compiled batched forward pass.
    
    Args:
        x_batch: Input features (batch_size, input_size)
        weights_tuple: Tuple of weight matrices
        biases_tuple: Tuple of bias vectors
        
    Returns:
        Output logits (batch_size, output_size)
    """
    # Hidden layers with ReLU activation
    for i in range(len(weights_tuple) - 1):
        x_batch = x_batch @ weights_tuple[i] + biases_tuple[i]
        x_batch = np.maximum(0, x_batch)  # ReLU
    
    # Output layer (no activation)
    x_batch = x_batch @ weights_tuple[-1] + biases_tuple[-1]
    
    return x_batch


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    if HAS_NUMBA:
        return relu_jit(x)
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    if HAS_NUMBA:
        return tanh_jit(x)
    return np.tanh(x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Softmax with temperature scaling.
    
    Args:
        x: Input logits
        temperature: Temperature parameter (lower = more deterministic)
        
    Returns:
        Probability distribution
    """
    if HAS_NUMBA:
        return softmax_jit(x, temperature)
    
    if temperature <= 0:
        temperature = 1e-8
    
    scaled = x / temperature
    # Numerical stability
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals) + 1e-10)


class PolicyNetwork:
    """
    Feedforward policy network.
    
    Architecture:
        Input (features) -> Hidden layers (ReLU) -> Output (action logits)
    
    The network outputs raw logits. Action selection applies:
        1. Mask illegal actions (set to -inf)
        2. Apply softmax with temperature
        3. Sample from distribution
    
    Attributes:
        config: Network configuration
        layer_sizes: List of all layer sizes [input, hidden..., output]
        weights: List of weight matrices
        biases: List of bias vectors
        genome_size: Total number of parameters
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        """
        Initialize network with given configuration.
        
        Args:
            config: Network config (uses defaults if None)
        """
        self.config = config or NetworkConfig()
        
        # Build layer sizes
        self.layer_sizes = (
            [self.config.input_size] +
            list(self.config.hidden_sizes) +
            [self.config.output_size]
        )
        
        # Select activation function
        activations = {
            'relu': relu,
            'tanh': tanh,
            'sigmoid': sigmoid,
        }
        self.activation = activations.get(self.config.activation, relu)
        
        # Initialize weights and biases
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        
        # Expose architecture attributes once
        self.input_size = self.config.input_size
        self.hidden_sizes = self.config.hidden_sizes
        self.output_size = self.config.output_size
        for i in range(len(self.layer_sizes) - 1):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            # Xavier/He initialization - use float32 for Numba compatibility
            std = np.sqrt(2.0 / in_size)
            self.weights.append(np.zeros((in_size, out_size), dtype=np.float32))
            self.biases.append(np.zeros(out_size, dtype=np.float32))
        
        # Calculate total genome size
        self._genome_size = sum(
            w.size + b.size for w, b in zip(self.weights, self.biases)
        )
    
    @property
    def genome_size(self) -> int:
        """Total number of trainable parameters."""
        return self._genome_size
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            features: Input feature vector (shape: input_size,)
            
        Returns:
            Action logits (shape: output_size,)
        """
        x = features.astype(np.float32)
        
        if HAS_NUMBA:
            # Use JIT-compiled version (2-3× faster)
            weights_tuple = tuple(self.weights)
            biases_tuple = tuple(self.biases)
            return forward_pass_jit(x, weights_tuple, biases_tuple)
        else:
            # Fallback: standard numpy implementation
            # Hidden layers with activation
            for i in range(len(self.weights) - 1):
                x = x @ self.weights[i] + self.biases[i]
                x = self.activation(x)
            
            # Output layer (no activation - raw logits)
            x = x @ self.weights[-1] + self.biases[-1]
            
            return x
    
    def forward_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Vectorized forward pass for batch of states.
        Processes multiple states at once for better performance.
        
        Args:
            features_batch: Input features (shape: batch_size, input_size)
            
        Returns:
            Action logits (shape: batch_size, output_size)
        """
        x = features_batch.astype(np.float32)
        
        if HAS_NUMBA:
            # Use JIT-compiled batched version (2-3× faster with parallelization)
            weights_tuple = tuple(self.weights)
            biases_tuple = tuple(self.biases)
            return forward_batch_jit(x, weights_tuple, biases_tuple)
        else:
            # Fallback: standard numpy implementation
            # Hidden layers with activation
            for i in range(len(self.weights) - 1):
                x = x @ self.weights[i] + self.biases[i]
                x = self.activation(x)
            
            # Output layer (no activation - raw logits)
            x = x @ self.weights[-1] + self.biases[-1]
            
            return x
    
    def select_action_batch(self, features_batch: np.ndarray, mask_batch: np.ndarray,
                           rng: np.random.Generator,
                           temperature: float = 1.0) -> np.ndarray:
        """
        Select actions for a batch of states (1.3-1.5× speedup via vectorization).
        
        Args:
            features_batch: State features (shape: batch_size, input_size)
            mask_batch: Binary masks (shape: batch_size, output_size)
            rng: Random number generator
            temperature: Sampling temperature
            
        Returns:
            Selected action indices (shape: batch_size,)
        """
        # Vectorized forward pass for entire batch
        logits_batch = self.forward_batch(features_batch)
        
        # Apply masks and temperature
        logits_batch = logits_batch - 1e9 * (1 - mask_batch)
        logits_batch = logits_batch / temperature
        
        # Compute probabilities for entire batch
        logits_batch = logits_batch - np.max(logits_batch, axis=1, keepdims=True)
        exp_logits = np.exp(logits_batch)
        probs_batch = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Sample actions for the whole batch by inverse-CDF rather than calling
        # rng.choice(p=...) once per row. That call re-validates the probability
        # vector and rebuilds a cumulative distribution every time, costing
        # ~19 us each; it dominated the training profile at 4.7 s per generation.
        # This draws from the identical distribution in a single vectorised step.
        cdf = np.cumsum(probs_batch, axis=1)
        cdf[:, -1] = 1.0                       # guard against float drift < 1.0
        u = rng.random((probs_batch.shape[0], 1))
        return (cdf < u).sum(axis=1).astype(np.int32)
    
    def select_action(self, features: np.ndarray, mask: np.ndarray,
                      rng: np.random.Generator,
                      temperature: float = 1.0) -> int:
        """
        Select an action given features and legal action mask.
        
        Args:
            features: State feature vector
            mask: Binary mask of legal actions (1=legal, 0=illegal)
            rng: Random number generator
            temperature: Sampling temperature (lower = more greedy)
            
        Returns:
            Selected action index
        """
        logits = self.forward(features)
        
        # Mask illegal actions
        masked_logits = np.where(mask > 0.5, logits, -1e9)
        
        # Check if any action is legal
        if not np.any(mask > 0.5):
            return 0  # Default to fold if nothing legal
        
        # Apply softmax with temperature
        probs = softmax(masked_logits, temperature)
        
        # Sample action
        action = rng.choice(len(probs), p=probs)
        return int(action)
    
    def get_action_probs(self, features: np.ndarray, mask: np.ndarray,
                         temperature: float = 1.0) -> np.ndarray:
        """
        Get action probability distribution.
        
        Args:
            features: State feature vector
            mask: Binary mask of legal actions
            temperature: Softmax temperature
            
        Returns:
            Probability distribution over actions
        """
        logits = self.forward(features)
        masked_logits = np.where(mask > 0.5, logits, -1e9)
        return softmax(masked_logits, temperature)
    
    def get_greedy_action(self, features: np.ndarray, mask: np.ndarray) -> int:
        """
        Get the greedy (highest probability) action.
        
        Args:
            features: State feature vector
            mask: Binary mask of legal actions
            
        Returns:
            Action index with highest probability
        """
        logits = self.forward(features)
        masked_logits = np.where(mask > 0.5, logits, -1e9)
        return int(np.argmax(masked_logits))
    
    def set_weights_from_genome(self, genome: np.ndarray):
        """
        Set network weights from a flat genome array.
        
        Args:
            genome: 1D array of all parameters
        """
        if len(genome) != self.genome_size:
            raise ValueError(
                f"Genome size mismatch: expected {self.genome_size}, "
                f"got {len(genome)}"
            )
        
        offset = 0
        for i in range(len(self.weights)):
            # Extract weight matrix
            w_size = self.weights[i].size
            w_shape = self.weights[i].shape
            self.weights[i] = genome[offset:offset + w_size].reshape(w_shape)
            offset += w_size
            
            # Extract bias vector
            b_size = self.biases[i].size
            self.biases[i] = genome[offset:offset + b_size]
            offset += b_size
    
    def get_genome(self) -> np.ndarray:
        """
        Get flat genome array from current weights.
        
        Returns:
            1D array of all parameters
        """
        parts = []
        for w, b in zip(self.weights, self.biases):
            parts.append(w.flatten())
            parts.append(b)
        return np.concatenate(parts)

    def set_weights(self, data):
        """
        Flexible setter for network parameters.

        Accepts:
          - 1D numpy array (flat genome) matching `genome_size`
          - list/tuple of numpy arrays [w0, b0, w1, b1, ...] matching shapes
          - dict with keys 'weights' and 'biases' containing lists of arrays
        """
        # 1) Flat genome array
        if isinstance(data, np.ndarray) and data.ndim == 1:
            return self.set_weights_from_genome(data)

        # 2) List/tuple of arrays as alternating [w0, b0, w1, b1, ...]
        #
        # A shape mismatch here is a real error and is reported as one.  This
        # used to be wrapped in `except Exception: pass`, so passing weights of
        # the wrong shape fell through to the TypeError at the bottom and was
        # reported as an unsupported *format* — sending you looking for the
        # wrong problem entirely.
        if isinstance(data, (list, tuple)):
            seq = list(data)
            if len(seq) != 2 * len(self.weights):
                raise ValueError(
                    f"expected {2 * len(self.weights)} arrays "
                    f"([w0, b0, ...] for {len(self.weights)} layers), got {len(seq)}"
                )
            for i in range(len(self.weights)):
                w = np.asarray(seq[2 * i], dtype=np.float32)
                b = np.asarray(seq[2 * i + 1], dtype=np.float32)
                if w.shape != self.weights[i].shape or b.shape != self.biases[i].shape:
                    raise ValueError(
                        f"layer {i} shape mismatch: got weights {w.shape} biases {b.shape}, "
                        f"expected {self.weights[i].shape} and {self.biases[i].shape}"
                    )
                self.weights[i] = w.copy()
                self.biases[i] = b.copy()
            return

        # 3) Dict with 'weights'/'biases'
        if isinstance(data, dict):
            ws, bs = data.get('weights'), data.get('biases')
            if ws is None or bs is None:
                raise ValueError("dict form requires both 'weights' and 'biases' keys")
            if len(ws) != len(self.weights) or len(bs) != len(self.biases):
                raise ValueError(
                    f"expected {len(self.weights)} weight and bias arrays, "
                    f"got {len(ws)} and {len(bs)}"
                )
            for i in range(len(self.weights)):
                w = np.asarray(ws[i], dtype=np.float32)
                b = np.asarray(bs[i], dtype=np.float32)
                if w.shape != self.weights[i].shape or b.shape != self.biases[i].shape:
                    raise ValueError(
                        f"layer {i} shape mismatch: got weights {w.shape} biases {b.shape}, "
                        f"expected {self.weights[i].shape} and {self.biases[i].shape}"
                    )
                self.weights[i] = w.copy()
                self.biases[i] = b.copy()
            return

        raise TypeError(
            f"unsupported weight format {type(data).__name__}; expected a flat "
            f"1-D genome array, a [w0, b0, ...] sequence, or a "
            f"{{'weights': [...], 'biases': [...]}} dict"
        )
    
    def copy(self) -> 'PolicyNetwork':
        """Create a deep copy of this network."""
        new_net = PolicyNetwork(self.config)
        new_net.set_weights_from_genome(self.get_genome())
        return new_net
    
    def __repr__(self) -> str:
        return f"PolicyNetwork({self.layer_sizes}, params={self.genome_size})"

# Abstract action indices
ABSTRACT_ACTIONS = {
    0: 'fold',
    1: 'check_call',
    2: 'raise_half_pot',
    3: 'raise_pot',
    4: 'raise_2x_pot',
    5: 'all_in',
}

def create_action_mask(game, player_id: int) -> np.ndarray:
    """
    Create abstract action mask from game state.
    
    Maps the engine's legal actions to our 6 abstract actions:
        0: fold
        1: check/call
        2: raise 0.5x pot
        3: raise 1.0x pot
        4: raise 2.0x pot
        5: all-in
    
    Args:
        game: PokerGame instance
        player_id: Player to create mask for

    Returns:
        Binary mask array of shape (6,)

    Note:
        Kept as a thin alias so training code reads naturally.  The mask itself
        is defined once in `engine.features.get_abstract_action_mask` — there
        used to be three separate implementations of this that disagreed.
    """
    from engine.features import get_abstract_action_mask
    return get_abstract_action_mask(game, player_id)
