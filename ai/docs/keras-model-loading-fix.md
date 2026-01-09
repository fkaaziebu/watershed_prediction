# Keras Model Loading Fix

## Problem

When loading a Keras model saved in `.h5` format, the following error occurs:

```
Could not deserialize 'keras.metrics.mse' because it is not a KerasSaveable subclass
```

This error happens in `source_sink_analysis.py` line 37 when calling:
```python
self.model = keras.models.load_model(model_path)
```

## Root Cause

When a Keras model is compiled with string-based metrics like `metrics=['mae', 'mse']`, these are saved as part of the model. In newer versions of TensorFlow/Keras (especially TF 2.13+), the deserialization of these string-based metrics can fail because:

1. The serialization format changed between Keras versions
2. String shortcuts like `'mse'` are converted to metric objects during compilation, and the saved representation may not match what the newer Keras version expects
3. The `.h5` format stores metrics in a way that can be incompatible across versions

## Solution

Load the model with `compile=False` to skip deserializing the optimizer and metrics, then manually recompile:

```python
# Before (causes error):
self.model = keras.models.load_model(model_path)

# After (fixed):
self.model = keras.models.load_model(model_path, compile=False)
self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
```

## Why This Works

- `compile=False` tells Keras to only load the model architecture and weights, ignoring the saved optimizer state and metrics
- Manually recompiling ensures the metrics are created fresh with the current Keras version
- The model weights are preserved, so predictions remain accurate

## Alternative Solutions

### 1. Use SavedModel format instead of .h5

```python
# Save
model.save('models/watershed_model')  # No .h5 extension = SavedModel format

# Load
model = keras.models.load_model('models/watershed_model')
```

### 2. Use custom_objects parameter

```python
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

model = keras.models.load_model(
    model_path,
    custom_objects={
        'mse': MeanSquaredError(),
        'mae': MeanAbsoluteError()
    }
)
```

### 3. Save only weights

```python
# Save
model.save_weights('models/watershed_model.weights.h5')

# Load (requires rebuilding architecture first)
model = build_model()
model.load_weights('models/watershed_model.weights.h5')
```

## Applied Fix Location

File: `source_sink_analysis.py`
Lines: 35-38

```python
# Load model (compile=False to avoid metric deserialization issues)
self.model = keras.models.load_model(model_path, compile=False)
self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
```

## References

- [TensorFlow Model Saving Documentation](https://www.tensorflow.org/guide/keras/serialization_and_saving)
- [Keras load_model API](https://keras.io/api/models/model_saving_apis/)
