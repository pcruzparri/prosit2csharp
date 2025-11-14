import tensorflow as tf
import onnx
import tf2onnx
import numpy as np

def masked_spectral_distance(y_true, y_pred):
    """Custom loss function for spectral distance with masking."""
    return tf.reduce_mean(tf.square(y_true - y_pred))

class Attention(tf.keras.layers.Layer):
    def __init__(self, context=False, bias=True, W_regularizer=None, u_regularizer=None, 
                 b_regularizer=None, W_constraint=None, u_constraint=None, b_constraint=None, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.context = context
        self.bias = bias
        self.W_regularizer = W_regularizer
        self.u_regularizer = u_regularizer
        self.b_regularizer = b_regularizer
        self.W_constraint = W_constraint
        self.u_constraint = u_constraint
        self.b_constraint = b_constraint
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(input_shape[-1],),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.b = self.add_weight(
            name='b',
            shape=(input_shape[1],),
            initializer='zeros',
            trainable=True
        )
        
        super(Attention, self).build(input_shape)
    
    def call(self, x):
        uit = tf.reduce_sum(x * self.W, axis=-1)
        uit = uit + self.b
        uit = tf.tanh(uit)
        ait = tf.nn.softmax(uit, axis=1)
        ait = tf.expand_dims(ait, axis=-1)
        weighted_input = x * ait
        output = tf.reduce_sum(weighted_input, axis=1)
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

def create_compatible_model():
    """Create a model with regular GRU layers instead of CuDNN layers."""
    
    # Input layers
    peptides_in = tf.keras.layers.Input(shape=(30,), dtype='int32', name='peptides_in')
    collision_energy_in = tf.keras.layers.Input(shape=(1,), dtype='float32', name='collision_energy_in')
    precursor_charge_in = tf.keras.layers.Input(shape=(6,), dtype='float32', name='precursor_charge_in')
    
    # Embedding
    embedding = tf.keras.layers.Embedding(22, 32, name='embedding')(peptides_in)
    
    # Encoder 1 - Bidirectional GRU (replace CuDNN)
    encoder1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256, return_sequences=True, name='encoder1_gru'),
        name='encoder1'
    )(embedding)
    dropout_1 = tf.keras.layers.Dropout(0.3, name='dropout_1')(encoder1)
    
    # Encoder 2 - GRU (replace CuDNN)
    encoder2 = tf.keras.layers.GRU(512, return_sequences=True, name='encoder2')(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.3, name='dropout_2')(encoder2)
    
    # Meta inputs
    meta_in = tf.keras.layers.Concatenate(axis=-1, name='meta_in')([collision_energy_in, precursor_charge_in])
    meta_dense = tf.keras.layers.Dense(512, name='meta_dense')(meta_in)
    meta_dense_do = tf.keras.layers.Dropout(0.3, name='meta_dense_do')(meta_dense)
    
    # Attention
    encoder_att = Attention(name='encoder_att')(dropout_2)
    
    # Combine with meta
    add_meta = tf.keras.layers.Multiply(name='add_meta')([encoder_att, meta_dense_do])
    
    # Decoder
    repeat = tf.keras.layers.RepeatVector(29, name='repeat')(add_meta)
    decoder = tf.keras.layers.GRU(512, return_sequences=True, name='decoder')(repeat)
    dropout_3 = tf.keras.layers.Dropout(0.3, name='dropout_3')(decoder)
    
    # Output processing
    permute_1 = tf.keras.layers.Permute((2, 1), name='permute_1')(dropout_3)
    dense_1 = tf.keras.layers.Dense(29, activation='softmax', name='dense_1')(permute_1)
    permute_2 = tf.keras.layers.Permute((2, 1), name='permute_2')(dense_1)
    multiply_1 = tf.keras.layers.Multiply(name='multiply_1')([dropout_3, permute_2])
    timedense = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(6, name='dense_2'), 
        name='timedense'
    )(multiply_1)
    activation = tf.keras.layers.LeakyReLU(alpha=0.3, name='activation')(timedense)
    out = tf.keras.layers.Flatten(name='out')(activation)
    
    # Create model
    model = tf.keras.Model(
        inputs=[peptides_in, precursor_charge_in, collision_energy_in],
        outputs=out,
        name='model_1'
    )
    
    return model

def transfer_weights(original_model, new_model):
    """Transfer weights from original model to new compatible model."""
    
    # Map of layer names that need weight transfer
    weight_mapping = {
        'embedding': 'embedding',
        'encoder1': 'encoder1',  # Will need special handling for bidirectional
        'encoder2': 'encoder2',
        'encoder_att': 'encoder_att',
        'meta_dense': 'meta_dense', 
        'decoder': 'decoder',
        'dense_1': 'dense_1',
        'timedense': 'timedense'
    }
    
    for orig_name, new_name in weight_mapping.items():
        try:
            orig_layer = original_model.get_layer(orig_name)
            new_layer = new_model.get_layer(new_name)
            
            if orig_name == 'encoder1':  # Special handling for bidirectional
                # CuDNN bidirectional weights need to be handled carefully
                # For now, we'll try to copy what we can
                if hasattr(orig_layer, 'get_weights') and hasattr(new_layer, 'get_weights'):
                    orig_weights = orig_layer.get_weights()
                    new_weights = new_layer.get_weights()
                    if len(orig_weights) == len(new_weights):
                        new_layer.set_weights(orig_weights)
                        print(f"Transferred weights for {orig_name}")
                    else:
                        print(f"Weight shape mismatch for {orig_name}, skipping")
            else:
                # Standard weight transfer
                if hasattr(orig_layer, 'get_weights') and hasattr(new_layer, 'get_weights'):
                    weights = orig_layer.get_weights()
                    if weights:  # Only set if there are weights
                        new_layer.set_weights(weights)
                        print(f"Transferred weights for {orig_name}")
                        
        except Exception as e:
            print(f"Could not transfer weights for {orig_name}: {e}")

if __name__ == "__main__":
    
    print("Loading original model...")
    # Load original model with CuDNN layers
    original_model = tf.keras.models.load_model(
        "models/prosit/hla_cid/weight_192_0.16253.hdf5",
        custom_objects={
            'Attention': Attention,
            'masked_spectral_distance': masked_spectral_distance
        }
    )
    
    print("Creating compatible model...")
    # Create new model with regular GRU layers
    compatible_model = create_compatible_model()
    
    print("Transferring weights...")
    # Transfer weights from original to compatible model
    transfer_weights(original_model, compatible_model)
    
    print("Model summary:")
    compatible_model.summary()

    # Define input specifications
    spec = (tf.TensorSpec((None, 30), tf.float32, name="peptide_sequences"),
            tf.TensorSpec((None, 6), tf.float32, name="precursor_charges"),
            tf.TensorSpec((None, 1), tf.float32, name="normalized_collision_energies")
           )

    output_path = "models/prosit/hla_cid/weight_192_0.16253_compatible.onnx"
    print(f"Converting to ONNX format...")
    print(f"Output path: {output_path}")
    
    try:
        model_proto, _ = tf2onnx.convert.from_keras(compatible_model, input_signature=spec, opset=13, output_path=output_path)
        onnx.save_model(model_proto, output_path)
        print(f"ONNX model saved to: {output_path}")

        # Check the model
        print("Validating ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("This might be due to other unsupported operations.")