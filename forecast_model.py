from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import keras_tuner as kt

class MCDropout(layers.Dropout):
    """Monte Carlo Dropout layer that remains active during inference."""
    def call(self, inputs, training=None):
        # Force training=True to keep dropout active during inference
        return super().call(inputs, training=True)

class BahdanauAttention(layers.Layer):
    """Bahdanau-style (additive) attention mechanism."""
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, features, hidden):
        # features (encoder output) shape: (batch_size, time_steps, features_dim)
        # hidden (decoder state) shape: (batch_size, hidden_dim)
        
        # Expand hidden dimension to match features sequence length
        # hidden_with_time_axis shape: (batch_size, 1, hidden_dim)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        
        # Calculate attention score
        # score shape: (batch_size, time_steps, units)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        
        # attention_weights shape: (batch_size, time_steps, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        
        # context_vector shape: (batch_size, time_steps, features_dim)
        context_vector = attention_weights * features
        
        # Sum along time_steps axis
        # context_vector shape after sum: (batch_size, features_dim)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector
        
    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({'units': self.units})
        return config

def build_hypermodel(hp, input_shape=(None, 48), output_steps=30):
    """
    Builds a Bidirectional LSTM HyperModel with Bahdanau attention and MC Dropout
    for multi-step crop price forecasting using Keras Tuner.
    
    Args:
        hp: Keras Tuner HyperParameters object.
        input_shape: Tuple representing (lookback_window, num_features). Notice None is used for variable length.
        output_steps: Number of future days to forecast.
        
    Returns:
        Compiled Keras model.
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")
    
    # Tuned hyperparameters
    lstm_units = hp.Choice('lstm_units', values=[64, 128, 256])
    dropout_rate = hp.Choice('dropout', values=[0.1, 0.2, 0.3])
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-2])
    
    # 1. First BiLSTM Layer
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True), 
        name="bilstm_layer_1"
    )(inputs)
    x = MCDropout(dropout_rate, name="mc_dropout_1")(x)
    
    # 2. Second BiLSTM Layer (Current implementation uses 64 statically or lstm_units // 2)
    lstm_out, forward_h, forward_c, backward_h, backward_c = layers.Bidirectional(
        layers.LSTM(lstm_units // 2, return_sequences=True, return_state=True),
        name="bilstm_layer_2"
    )(x)
    lstm_out = MCDropout(dropout_rate, name="mc_dropout_2")(lstm_out)
    
    # Concatenate final hidden states from both directions
    state_h = layers.Concatenate(name="concat_hidden_states")([forward_h, backward_h])
    
    # 3. Bahdanau-style Attention
    context_vector = BahdanauAttention(lstm_units // 2, name="bahdanau_attention")(lstm_out, state_h)
    
    # Combine context vector with the hidden state for dense layers
    attention_output = layers.Concatenate(name="concat_attention_hidden")([context_vector, state_h])
    
    # 4. Two Dense Layers
    x = layers.Dense(lstm_units, activation='relu', name="dense_1")(attention_output)
    x = MCDropout(dropout_rate, name="mc_dropout_3")(x)
    
    x = layers.Dense(lstm_units // 2, activation='relu', name="dense_2")(x)
    x = MCDropout(dropout_rate, name="mc_dropout_4")(x)
    
    # 5. Output Layer
    outputs = layers.Dense(output_steps, activation='linear', name="output_layer")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CropPrice_BiLSTM_Attention")
    
    # 6. Optimizer with Cosine Annealing Learning Rate Schedule
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=1000, 
        alpha=0.01
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # 7. Loss: Huber loss (delta=1.0)
    loss = tf.keras.losses.Huber(delta=1.0)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    
    return model

def get_mc_dropout_predictions(model, X, n_iter=100, confidence_level=0.95):
    """
    Implements Monte Carlo Dropout for generating predictions with 95% confidence intervals.
    
    Args:
        model: Trained Keras model with MCDropout layers.
        X: Input data for prediction - shape (batch_size, 60, 48).
        n_iter: Number of stochastic forward passes (default 100).
        confidence_level: Desired confidence interval (e.g., 0.95 for 95%).
        
    Returns:
        mean_pred: Mean predicted values.
        lower_bound: Lower bound of the confidence interval.
        upper_bound: Upper bound of the confidence interval.
    """
    import scipy.stats as stats
    
    predictions = []
    # Perform n_iter stochastic forward passes
    for _ in range(n_iter):
        # By calling the model directly, we trigger a forward pass where training=True in our custom MCDropout layers
        predictions.append(model(X, training=True).numpy())
        
    predictions = np.array(predictions) # Shape: (n_iter, batch_size, output_steps)
    
    # Calculate mean and standard deviation across iterations
    mean_pred = np.mean(predictions, axis=0) # Shape: (batch_size, output_steps)
    std_pred = np.std(predictions, axis=0)   # Shape: (batch_size, output_steps)
    
    # Calculate z-score for the desired confidence level (1.96 for 95% CI)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    lower_bound = mean_pred - z_score * std_pred
    upper_bound = mean_pred + z_score * std_pred
    
    return mean_pred, lower_bound, upper_bound

if __name__ == "__main__":
    # Test hypermodel builder
    hp = kt.HyperParameters()
    hp.Choice('lstm_units', [64, 128, 256])
    hp.Choice('dropout', [0.1, 0.2, 0.3])
    hp.Choice('learning_rate', [1e-4, 1e-3, 1e-2])
    
    model = build_hypermodel(hp)
    model.summary()
