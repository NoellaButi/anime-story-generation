# Author: Yuki Rivera
# This python file defines the basic transformer layers used in a separate 
# notebook that trains the transformer model

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

# Positional Encoding layer that adds position info using sine and cosine function
class PositionalEncoding(layers.Layer):
    def __init__(self, max_len, d_model, **kwargs):
        
        super().__init__(**kwargs)
        # column vector containing the positions
        pos = np.arange(max_len)[:, np.newaxis]
        # dimension indices
        i = np.arange(d_model)[np.newaxis, :]

        # Calculates the angle rates using the formula from the paper(attention is all you need)
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

        # multiplies each position by its corresponding rate to make it full angle matrix
        angle_rads = pos * angle_rates
        # Applies sin to even indices in the embedding dimension
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) 
        # Applies cos to odd indices in the embedding dimension
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) 

        # Adds the batch dimension by casting with np.newaxis
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    # Called during the forward pass
    def call(self, x):
        # Adds positional encoding to the input token embeddings
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]

# A single block for the Transformer Encoder. It contains self-attention and a feed-forward network.
class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        # Layer Normalization to stabilize training.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout for regularization.
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False, mask=None):

        attention_mask = None
        if mask is not None:

            # Expands 2D padding mask to a 4D shape that Keras expects
            attention_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")

        attn_output = self.att(inputs, inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# A single block for the Transformer Decoder. It has two attention layers.
class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        # Masked self-attention
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        # Cross-attention
        self.cross_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        # A simple two-layer feed-forward network.
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        # Three layer normalization layers for each sub-layer's output.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        # Three dropout layers for regularization.
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, inputs, encoder_outputs, training=False, decoder_mask=None, encoder_mask=None):

        # Gets the length of the current sequence dynamically as the length might change from batch to batch.
        seq_len = tf.shape(inputs)[1]

        # Creates a causal_mask(look-ahead mask) to prevent attention to future tokens
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

        # Reshapes the causal mask to 4D for broadcasting.
        causal_mask = causal_mask[tf.newaxis, tf.newaxis, :, :]

        # Combines the causal mask and padding mask to prevent attending to future or padded tokens
        self_attention_mask = causal_mask
        if decoder_mask is not None:
            # Reshapes the 2D padding mask to a broadcastable 4D shape.
            padding_mask = tf.cast(decoder_mask[:, tf.newaxis, tf.newaxis, :], dtype=causal_mask.dtype)
            # Combines the two masks
            self_attention_mask = tf.minimum(causal_mask, padding_mask)

        # Performs masked self-attention.
        self_attn_output = self.self_att(
            query=inputs, value=inputs, key=inputs,
            attention_mask=self_attention_mask,
            use_causal_mask=False
        )
        self_attn_output = self.dropout1(self_attn_output, training=training)
        out1 = self.layernorm1(inputs + self_attn_output) 


        # Prepares the mask for the second attention layer.
        cross_attention_mask = None
        if encoder_mask is not None:
            cross_attention_mask = tf.cast(encoder_mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)

        # Performs cross-attention.
        cross_attn_output = self.cross_att(
            query=out1, value=encoder_outputs, key=encoder_outputs,
            attention_mask=cross_attention_mask
        )
        cross_attn_output = self.dropout2(cross_attn_output, training=training)
        out2 = self.layernorm2(out1 + cross_attn_output) 

        # The final sub-layer
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output) 
