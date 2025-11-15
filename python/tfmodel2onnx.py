import tensorflow as tf
import onnx
import tf2onnx

def masked_spectral_distance(y_true, y_pred):
    """Custom loss function for spectral distance with masking."""
    # This is a placeholder implementation - the exact implementation would depend on the original code
    # For loading purposes, we just need something that can be deserialized
    return tf.reduce_mean(tf.square(y_true - y_pred))

class Attention(tf.keras.layers.Layer):
    def __init__(self, context=False, bias=True, W_regularizer=None, u_regularizer=None, 
                 b_regularizer=None, W_constraint=None, u_constraint=None, b_constraint=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        # Store the parameters even though most aren't used, for config compatibility
        self.context = context
        self.bias = bias
        self.W_regularizer = W_regularizer
        self.u_regularizer = u_regularizer
        self.b_regularizer = b_regularizer
        self.W_constraint = W_constraint
        self.u_constraint = u_constraint
        self.b_constraint = b_constraint
        
    def build(self, input_shape):
        # Based on the actual saved weights:
        # encoder_att_W:0 shape: (512,) - weight vector for features
        # encoder_att_b:0 shape: (30,) - bias vector for time steps
        
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1],),  # (512,) - feature dimension
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.b = self.add_weight(
            name='b',
            shape=(input_shape[1],),  # (30,) - time steps dimension  
            initializer='zeros',
            trainable=True
        )
        
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        # x shape: (batch_size, time_steps, features) = (?, 30, 512)
        # From original error: uit = tf.matmul(x, self.W) where W is (512,)
        # This suggests element-wise multiplication followed by sum
        
        # Apply W to each time step: x * W (broadcasting)
        uit = tf.reduce_sum(x * self.W, axis=-1)  # (?, 30, 512) * (512,) -> (?, 30)
        
        # Add bias
        uit = uit + self.b  # (?, 30) + (30,) = (?, 30)
        
        # Apply activation (commonly used in attention)
        uit = tf.tanh(uit)
        
        # Apply softmax to get attention weights
        ait = tf.nn.softmax(uit, axis=1)  # (?, 30)
        ait = tf.expand_dims(ait, axis=-1)  # (?, 30, 1)
        
        # Apply attention weights to input
        weighted_input = x * ait  # (?, 30, 512) * (?, 30, 1) = (?, 30, 512)
        output = tf.reduce_sum(weighted_input, axis=1)  # (?, 512)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({
            'context': self.context,
            'bias': self.bias,
            'W_regularizer': self.W_regularizer,
            'u_regularizer': self.u_regularizer,
            'b_regularizer': self.b_regularizer,
            'W_constraint': self.W_constraint,
            'u_constraint': self.u_constraint,
            'b_constraint': self.b_constraint
        })
        return config

if __name__ == "__main__":
    
    print("Loading model...")
    # Load model with custom objects
    model = tf.keras.models.load_model(
        "models/prosit/hla_cid/weight_192_0.16253.hdf5",
        custom_objects={
            'Attention': Attention,
            'masked_spectral_distance': masked_spectral_distance
        }
    )
    
    print("Model loaded successfully!")
    print(f"Model summary:")
    model.summary()

    # Define input specifications for the model
    spec = (tf.TensorSpec((None, 30), tf.float32, name="peptide_sequences"),
            tf.TensorSpec((None, 6), tf.float32, name="precursor_charge"),
            tf.TensorSpec((None, 1), tf.float32, name="normalized_collision_energy")
           )

    output_path = "models/prosit/hla_cid/weight_192_0.16253.onnx"
    print(f"Converting to ONNX format...")
    print(f"Output path: {output_path}")
    
    try:
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
        onnx.save_model(model_proto, output_path)
        print(f"ONNX model saved to: {output_path}")

        # Check the model
        print("Validating ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("This might be due to unsupported operations in tf2onnx.")