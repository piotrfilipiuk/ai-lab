"""Minimal transformer."""
import jax
import jax.numpy as jnp
from jax import random
from jax.nn import initializers

# A stable GELU activation function implementation
# Gaussian Error Linear Unit is a smooth activation often used in transformers.
def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + 0.044715 * x**3)))

#---------------------------------------------------#
## 1. Positional Encoding
#---------------------------------------------------#
# Since the model contains no recurrence or convolution, we must inject information
# about the relative or absolute position of the tokens in the sequence.
def positional_encoding(seq_len, d_model):
    """
    Generates sinusoidal positional encodings.

    Args:
        seq_len (int): The length of the sequence.
        d_model (int): The dimensionality of the model's embeddings.

    Returns:
        jnp.ndarray: A (1, seq_len, d_model) array of positional encodings.
    """
    # Create an array of position indices [0, 1, ..., seq_len-1]
    position = jnp.arange(seq_len)[:, jnp.newaxis]
    # Create an array for the division term in the formula
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))

    # Calculate the positional encodings
    pe = jnp.zeros((seq_len, d_model))
    # Apply sin to even indices in the array; 2i
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    # Apply cos to odd indices in the array; 2i+1
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))

    # Add a batch dimension so it can be easily added to the input embeddings
    return pe[jnp.newaxis, :, :]

#---------------------------------------------------#
## 2. Scaled Dot-Product Attention
#---------------------------------------------------#
# This is the core attention mechanism. It computes a weighted sum of the values,
# where the weight of each value is determined by the dot-product of the query
# with the corresponding key.
def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Computes Scaled Dot-Product Attention.

    Args:
        q (jnp.ndarray): Queries, shape (..., seq_len_q, d_k)
        k (jnp.ndarray): Keys, shape (..., seq_len_k, d_k)
        v (jnp.ndarray): Values, shape (..., seq_len_k, d_v)
        mask (jnp.ndarray, optional): Mask to apply to the attention scores.

    Returns:
        tuple: A tuple containing the output of the attention and the attention weights.
    """
    # Matmul: (..., seq_len_q, d_k) @ (..., d_k, seq_len_k) -> (..., seq_len_q, seq_len_k)
    matmul_qk = jnp.matmul(q, jnp.swapaxes(k, -2, -1))

    # Scale matmul_qk by the square root of the key dimension (d_k)
    d_k = q.shape[-1]
    scaled_attention_logits = matmul_qk / jnp.sqrt(d_k)

    # Apply the mask (if provided). The mask is used to prevent attention
    # to certain positions (e.g., padding tokens or future tokens in a decoder).
    if mask is not None:
        # We add a very large negative number to the masked positions.
        # When softmax is applied, these positions will have a weight of ~0.
        scaled_attention_logits += (mask * -1e9)

    # Softmax is applied to the last axis (seq_len_k) to obtain the weights.
    attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)

    # The weights are multiplied by the values to get the final output.
    # Output shape: (..., seq_len_q, d_v)
    output = jnp.matmul(attention_weights, v)
    return output, attention_weights

#---------------------------------------------------#
## 3. Multi-Head Attention
#---------------------------------------------------#
# Multi-head attention allows the model to jointly attend to information from
# different representation subspaces at different positions.
def multi_head_attention_init_params(key, d_model, num_heads):
    """Initializes parameters for the Multi-Head Attention layer."""
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    d_k = d_model // num_heads
    keys = random.split(key, 4)

    # Initialize weights for Q, K, V, and the final output projection
    # glorot_uniform is a standard way to initialize neural network weights.
    wq = initializers.glorot_uniform()(keys[0], (d_model, d_model))
    wk = initializers.glorot_uniform()(keys[1], (d_model, d_model))
    wv = initializers.glorot_uniform()(keys[2], (d_model, d_model))
    wo = initializers.glorot_uniform()(keys[3], (d_model, d_model))

    return {"wq": wq, "wk": wk, "wv": wv, "wo": wo}

def multi_head_attention_apply(params, x_q, x_k, x_v, mask, d_model, num_heads):
    """Applies the Multi-Head Attention layer."""
    batch_size = x_q.shape[0]
    d_k = d_model // num_heads

    # 1. Linearly project Q, K, V
    # x shape: (batch_size, seq_len, d_model)
    # W shape: (d_model, d_model)
    # Q, K, V shape: (batch_size, seq_len, d_model)
    q = jnp.matmul(x_q, params["wq"])
    k = jnp.matmul(x_k, params["wk"])
    v = jnp.matmul(x_v, params["wv"])

    # 2. Reshape and transpose for multi-head computation
    # This splits the d_model dimension into (num_heads, d_k)
    # New shape: (batch_size, num_heads, seq_len, d_k)
    q = q.reshape(batch_size, -1, num_heads, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, -1, num_heads, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, -1, num_heads, d_k).transpose(0, 2, 1, 3)

    # 3. Apply scaled dot-product attention
    # attention shape: (batch_size, num_heads, seq_len_q, d_k)
    attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    # 4. Concatenate heads and apply final linear layer
    # Transpose back to (batch_size, seq_len_q, num_heads, d_k)
    attention = attention.transpose(0, 2, 1, 3)
    # Reshape to (batch_size, seq_len_q, d_model) to concatenate heads
    concat_attention = attention.reshape(batch_size, -1, d_model)

    # Final linear projection
    # output shape: (batch_size, seq_len_q, d_model)
    output = jnp.matmul(concat_attention, params["wo"])
    return output, attention_weights

#---------------------------------------------------#
## 4. Position-wise Feed-Forward Network
#---------------------------------------------------#
# This is applied to each position independently. It consists of two linear
# transformations with a GELU activation in between.
def feed_forward_init_params(key, d_model, d_ff):
    """Initializes parameters for the Feed-Forward Network."""
    keys = random.split(key, 2)
    # Weights for the two linear layers
    w1 = initializers.glorot_uniform()(keys[0], (d_model, d_ff))
    w2 = initializers.glorot_uniform()(keys[1], (d_ff, d_model))
    # Biases for the two linear layers
    b1 = jnp.zeros(d_ff)
    b2 = jnp.zeros(d_model)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

def feed_forward_apply(params, x):
    """Applies the Feed-Forward Network."""
    # First linear transformation with GELU activation
    x = gelu(jnp.matmul(x, params["w1"]) + params["b1"])
    # Second linear transformation
    x = jnp.matmul(x, params["w2"]) + params["b2"]
    return x

#---------------------------------------------------#
## Mixture of Experts
#---------------------------------------------------#
def moe_init_params(key, d_model, d_ff, num_experts):
    """
    Initializes all parameters for the MoE Transformer model.
    """
    keys = random.split(key, num_experts+1)
    return {
       # Gating network parameters
       "gate": {
          "wg": random.normal(keys[-1], (d_model, num_experts))
       },
       # Parameters for each expert
       "experts": {
           f"expert_{j}": {
               "w1": random.normal(keys[j], (d_model, d_ff)),
               "b1": jnp.zeros((d_ff,)),
               "w2": random.normal(keys[j], (d_ff, d_model)),
               "b2": jnp.zeros((d_model,)),
           } for j in range(num_experts)
       }
    }

def moe_apply(params, x, num_experts, top_k=3):
    """
    A Mixture of Experts (MoE) layer supporting Top-K routing.
    """
    batch_size, seq_len, d_model = x.shape

    # 1. Gating Network
    # Shape: (batch, seq, num_experts)
    gate_logits = jnp.matmul(x, params["gate"]["wg"])
    
    # 2. Get Top-K Indices and Values
    # We use jax.lax.top_k to find the k highest scores and their indices.
    # gate_values shape: (batch, seq, top_k) - The logits or probs of the best experts
    # expert_indices shape: (batch, seq, top_k) - The IDs of the best experts
    gate_logits_top_k, expert_indices = jax.lax.top_k(gate_logits, k=top_k)

    # 3. Softmax & Normalization
    # We typically apply softmax only over the top-k selected experts to ensure
    # the weights we use for aggregation sum to 1.
    weights = jax.nn.softmax(gate_logits_top_k, axis=-1)
    
    # 4. Create Mask for "Scatter/Gather" operations
    # We need a way to send the right tokens to the right experts.
    # Shape: (batch, seq, top_k, num_experts)
    # This one-hot encodes the expert ID chosen at each of the k positions.
    expert_mask = jax.nn.one_hot(expert_indices, num_classes=num_experts)

    # 5. Process Experts
    # Flatten batch and seq dimensions for processing
    flat_x = x.reshape(-1, d_model) # (batch*seq, d_model)
    
    # Run all experts. (In production, you'd use parallel computation/sharding here)
    # We calculate the output of EVERY expert on EVERY token (naive implementation)
    # Note: Optimizing this to only run selected experts requires 'dispatch/combine' logic 
    # which is complex in pure JAX without library helpers like flax/haiku.
    # For this educational 'minimal' version, we compute all and mask the results.
    
    all_expert_outputs = []
    for i in range(num_experts):
        out = feed_forward_apply(params["experts"][f"expert_{i}"], flat_x)
        all_expert_outputs.append(out)
    
    # Stack: (num_experts, batch*seq, d_model)
    stacked_expert_outputs = jnp.stack(all_expert_outputs)
    
    # Reshape to (batch, seq, num_experts, d_model) to match our mask
    # We move num_experts to the 2nd dim to align with the mask logic
    stacked_expert_outputs = stacked_expert_outputs.transpose(1, 0, 2).reshape(batch_size, seq_len, num_experts, d_model)

    # 6. Aggregate (Weighted Sum)
    # We now have:
    #   weights:      (batch, seq, top_k)
    #   expert_mask:  (batch, seq, top_k, num_experts)
    #   outputs:      (batch, seq, num_experts, d_model)
    
    # First, select the specific outputs for our top_k experts.
    # We sum over num_experts (e) using the mask to pick the right one.
    # Result: (batch, seq, top_k, d_model)
    selected_outputs = jnp.einsum("bst e, bse d -> bst d", expert_mask, stacked_expert_outputs)

    # Finally, weight them by the gate probabilities and sum over top_k (t).
    # Result: (batch, seq, d_model)
    final_output = jnp.einsum("bst, bst d -> bsd", weights, selected_outputs)

    return final_output

#---------------------------------------------------#
## 5. Layer Normalization
#---------------------------------------------------#
# Used to stabilize training by normalizing the inputs to each sub-layer.
def layer_norm_init_params(d_model, eps=1e-5):
    """Initializes parameters for Layer Normalization."""
    # These are learnable parameters, sometimes called gain (gamma) and bias (beta)
    gamma = jnp.ones(d_model)
    beta = jnp.zeros(d_model)
    return {"gamma": gamma, "beta": beta, "eps": eps}

def layer_norm_apply(params, x):
    """Applies Layer Normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return params["gamma"] * (x - mean) / (std + params["eps"]) + params["beta"]

#---------------------------------------------------#
## 6. Encoder Layer
#---------------------------------------------------#
# An encoder layer consists of a multi-head self-attention mechanism followed by
# a position-wise feed-forward network. Residual connections and layer normalization
# are applied around each of the two sub-layers.
def encoder_layer_init_params(key, d_model, num_heads, d_ff, num_experts):
    """Initializes parameters for a single Encoder Layer."""
    mha_key, ffn_key = random.split(key)
    params = {
        "mha": multi_head_attention_init_params(mha_key, d_model, num_heads),
        "norm1": layer_norm_init_params(d_model),
        "norm2": layer_norm_init_params(d_model)
    }
    if num_experts == 1:
        params["ffn"] = feed_forward_init_params(ffn_key, d_model, d_ff)
    else:
        assert num_experts > 1
        params["moe"] = moe_init_params(ffn_key, d_model, d_ff, num_experts)
    return params

def encoder_layer_apply(params, x, mask, d_model, num_heads, num_experts):
    """Applies a single Encoder Layer."""
    # 1. Multi-Head Attention sub-layer
    attn_output, _ = multi_head_attention_apply(
        params["mha"], x, x, x, mask, d_model, num_heads
    )
    # Residual connection and layer normalization
    x = layer_norm_apply(params["norm1"], x + attn_output)

    if num_experts == 1:
        # 2. Feed-Forward sub-layer
        ffn_output = feed_forward_apply(params["ffn"], x)
    else:
        assert num_experts > 1
        ffn_output = moe_apply(params["moe"], x, num_experts)
    # Residual connection and layer normalization
    x = layer_norm_apply(params["norm2"], x + ffn_output)
    return x

#---------------------------------------------------#
## 7. Decoder Layer
#---------------------------------------------------#
# A decoder layer has three sub-layers: masked self-attention, encoder-decoder
# attention, and a feed-forward network.
def decoder_layer_init_params(key, d_model, num_heads, d_ff, num_experts):
    """Initializes parameters for a single Decoder Layer."""
    mha1_key, mha2_key, ffn_key = random.split(key, 3)
    params = {
        "mha1": multi_head_attention_init_params(mha1_key, d_model, num_heads),
        "mha2": multi_head_attention_init_params(mha2_key, d_model, num_heads),
        "norm1": layer_norm_init_params(d_model),
        "norm2": layer_norm_init_params(d_model),
        "norm3": layer_norm_init_params(d_model)
    }
    if num_experts == 1:
        params["ffn"] = feed_forward_init_params(ffn_key, d_model, d_ff)
    else:
        assert num_experts > 1
        params["moe"] = moe_init_params(ffn_key, d_model, d_ff, num_experts)
    return params

def decoder_layer_apply(params, x, enc_output, look_ahead_mask, padding_mask, d_model, num_heads, num_experts):
    """Applies a single Decoder Layer."""
    # 1. Masked Multi-Head Attention (self-attention)
    attn1, _ = multi_head_attention_apply(
        params["mha1"], x, x, x, look_ahead_mask, d_model, num_heads
    )
    x = layer_norm_apply(params["norm1"], x + attn1)

    # 2. Encoder-Decoder Attention
    # Query is from the previous decoder layer (x), Key and Value are from the encoder output.
    attn2, _ = multi_head_attention_apply(
        params["mha2"], x, enc_output, enc_output, padding_mask, d_model, num_heads
    )
    x = layer_norm_apply(params["norm2"], x + attn2)

    if num_experts == 1:
        # 3. Feed-Forward Network
        ffn_output = feed_forward_apply(params["ffn"], x)
    else:
        assert num_experts > 1
        ffn_output = moe_apply(params["moe"], x, num_experts)
    x = layer_norm_apply(params["norm3"], x + ffn_output)
    return x

#---------------------------------------------------#
## 8. Full Transformer Model
#---------------------------------------------------#
# The Transformer model stacks N encoder layers and N decoder layers.
def transformer_init_params(key, num_layers, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_experts):
    """Initializes parameters for the entire Transformer model."""
    keys = random.split(key, num_layers * 2 + 2)
    return {
        "encoder_embedding": initializers.glorot_uniform()(keys[0], (src_vocab_size, d_model)),
        "decoder_embedding": initializers.glorot_uniform()(keys[1], (tgt_vocab_size, d_model)),
        "encoder_layers": [encoder_layer_init_params(keys[i+2], d_model, num_heads, d_ff, num_experts) for i in range(num_layers)],
        "decoder_layers": [decoder_layer_init_params(keys[i+2+num_layers], d_model, num_heads, d_ff, num_experts) for i in range(num_layers)],
        "final_linear": initializers.glorot_uniform()(keys[-1], (d_model, tgt_vocab_size))
    }

def transformer_apply(params, src, tgt, src_mask, tgt_mask, num_layers, d_model, num_heads, num_experts):
    """Applies the full Transformer model (forward pass)."""
    # 1. Get sequence lengths and apply embeddings
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    # Encoder input embedding and positional encoding
    enc_output = params["encoder_embedding"][src, :] * jnp.sqrt(d_model)
    enc_output += positional_encoding(src_seq_len, d_model)
    
    # Decoder input embedding and positional encoding
    dec_output = params["decoder_embedding"][tgt, :] * jnp.sqrt(d_model)
    dec_output += positional_encoding(tgt_seq_len, d_model)

    # 2. Encoder Stack
    for i in range(num_layers):
        enc_output = encoder_layer_apply(params["encoder_layers"][i], enc_output, src_mask, d_model, num_heads, num_experts)

    # 3. Decoder Stack
    for i in range(num_layers):
        dec_output = decoder_layer_apply(params["decoder_layers"][i], dec_output, enc_output, tgt_mask, src_mask, d_model, num_heads, num_experts)
        
    # 4. Final Linear layer and Softmax
    final_output = jnp.matmul(dec_output, params["final_linear"])
    
    # We return the logits. Softmax would be applied in the loss function for training.
    return final_output

#---------------------------------------------------#
## 9. Utility: Mask Creation
#---------------------------------------------------#
def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask for the decoder's self-attention.
    This prevents positions from attending to subsequent positions.
    Returns a mask of shape (1, 1, size, size)
    """
    mask = 1 - jnp.tril(jnp.ones((size, size)))
    return mask[jnp.newaxis, jnp.newaxis, :, :] # Add batch and head dimensions

#---------------------------------------------------#
## 10. Example Usage
#---------------------------------------------------#
if __name__ == "__main__":
    # Hyperparameters
    key = random.PRNGKey(0)
    batch_size = 2
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    src_seq_len = 10
    tgt_seq_len = 12
    num_layers = 4      # Number of encoder/decoder layers
    d_model = 128       # Embedding dimension
    num_heads = 8       # Number of attention heads
    d_ff = 512          # Hidden layer size in FFN
    # If num_experts == 1 we use ffn, otherwise if num_experts > 1 we use moe
    num_experts = 8     # Number of experts

    # 1. Initialize model parameters
    print("Initializing model parameters...")
    model_params = transformer_init_params(
        key, num_layers, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_experts
    )
    print("Done.\n")
    
    # 2. Generate dummy input data
    key, src_key, tgt_key = random.split(key, 3)
    dummy_src = random.randint(src_key, (batch_size, src_seq_len), 0, src_vocab_size)
    dummy_tgt = random.randint(tgt_key, (batch_size, tgt_seq_len), 0, tgt_vocab_size)

    print(f"Source input shape: {dummy_src.shape}")
    print(f"Target input shape: {dummy_tgt.shape}\n")

    # 3. Create masks
    # For this minimal example, we'll use a simple look-ahead mask for the target
    # and no padding mask for the source. In a real application, you would also
    # create a padding mask for the source input.
    src_padding_mask = None # Not implemented for simplicity
    look_ahead_mask = create_look_ahead_mask(tgt_seq_len)
    print(f"Look-ahead mask shape: {look_ahead_mask.shape} (for target self-attention)\n")
    
    # 4. JIT compile the apply function for performance
    # JAX's Just-In-Time (JIT) compilation will dramatically speed up execution.
    # The first run will be slow due to compilation, but subsequent runs will be fast.
    print("JIT compiling the model's forward pass...")
    fast_transformer_apply = jax.jit(transformer_apply, static_argnums=(5, 6, 7, 8))
    print("Done.\n")

    # 5. Run the forward pass
    print("Running a forward pass...")
    output_logits = fast_transformer_apply(
        model_params, dummy_src, dummy_tgt, src_padding_mask, look_ahead_mask,
        num_layers, d_model, num_heads, num_experts
    )
    print("Done.\n")
    
    # 6. Check the output
    # The output should have the shape (batch_size, target_seq_len, target_vocab_size)
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")

    # Verify that the output shape is as expected
    assert output_logits.shape == (batch_size, tgt_seq_len, tgt_vocab_size)

    print("\n✅ Minimal Transformer implementation in JAX ran successfully!")
