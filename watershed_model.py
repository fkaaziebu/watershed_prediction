"""
watershed_model.py
==================
TensorFlow/Keras model for predicting water pollution in watershed networks.

This model combines:
- Spatial features (segment properties, upstream info)
- Temporal patterns (LSTM for time series)
- Graph-aware processing (upstream-downstream relationships)

Author: MedTrack
Date: January 2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

class WatershedDataPreprocessor:
    """Prepare data for model training."""
    
    def __init__(self, data_path='data/synthetic_watershed.csv'):
        """Initialize preprocessor."""
        self.data_path = data_path
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        
    def load_data(self):
        """Load watershed data."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"  Loaded {len(self.df)} records")
        print(f"  {self.df['segment_id'].nunique()} unique segments")
        print(f"  {self.df['day'].nunique()} time steps")
        
    def create_sequences(self, sequence_length=7):
        """Create time series sequences for LSTM."""
        print(f"\nCreating sequences (length={sequence_length})...")
        
        # Feature columns
        feature_cols = [
            'segment_index',
            'latitude',
            'longitude',
            'land_use_encoded',
            'upstream_count',
            'upstream_concentration',
            'flow_rate',
            'day'  # To capture seasonality
        ]
        
        target_col = 'concentration'
        
        X_sequences = []
        y_sequences = []
        segment_ids = []
        
        # Create sequences for each segment
        for segment in self.df['segment_id'].unique():
            segment_data = self.df[self.df['segment_id'] == segment].sort_values('day')
            
            features = segment_data[feature_cols].values
            targets = segment_data[target_col].values
            
            # Create sliding windows
            for i in range(len(segment_data) - sequence_length):
                X_seq = features[i:i+sequence_length]
                y_seq = targets[i+sequence_length]  # Predict next time step
                
                X_sequences.append(X_seq)
                y_sequences.append(y_seq)
                segment_ids.append(segment)
        
        self.X = np.array(X_sequences)
        self.y = np.array(y_sequences)
        self.segment_ids = np.array(segment_ids)
        
        print(f"  Created {len(self.X)} sequences")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")
        
    def normalize_data(self):
        """Normalize features and target."""
        print("\nNormalizing data...")
        
        # Reshape for scaling
        n_samples, n_timesteps, n_features = self.X.shape
        X_reshaped = self.X.reshape(-1, n_features)
        
        # Fit and transform features
        X_scaled = self.scaler_features.fit_transform(X_reshaped)
        self.X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
        
        # Scale target
        self.y_scaled = self.scaler_target.fit_transform(self.y.reshape(-1, 1)).flatten()
        
        print("  Features normalized")
        print("  Target normalized")
        
    def split_data(self, test_size=0.2, val_size=0.1):
        """Split into train/val/test sets."""
        print(f"\nSplitting data (test={test_size}, val={val_size})...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
            self.X_scaled, self.y_scaled, self.segment_ids,
            test_size=test_size, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_temp, y_temp, ids_temp,
            test_size=val_ratio, random_state=42
        )
        
        self.splits = {
            'X_train': X_train, 'y_train': y_train, 'ids_train': ids_train,
            'X_val': X_val, 'y_val': y_val, 'ids_val': ids_val,
            'X_test': X_test, 'y_test': y_test, 'ids_test': ids_test
        }
        
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        return self.splits
    
    def prepare(self, sequence_length=7):
        """Complete preprocessing pipeline."""
        self.load_data()
        self.create_sequences(sequence_length)
        self.normalize_data()
        return self.split_data()


class WatershedLSTMModel:
    """LSTM-based model for watershed pollution prediction."""
    
    def __init__(self, sequence_length=7, n_features=8):
        """Initialize model architecture."""
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units=64, dense_units=32, dropout_rate=0.2):
        """Build the model architecture."""
        print("\nBuilding model...")
        
        # Input layer
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers for temporal patterns
        x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.LSTM(lstm_units // 2, return_sequences=False)(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Dense layers for final prediction
        x = layers.Dense(dense_units, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(dense_units // 2, activation='relu')(x)
        
        # Output layer (single value: concentration)
        outputs = layers.Dense(1)(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='WatershedLSTM')
        
        print("  Model architecture:")
        self.model.summary()
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model."""
        print("\nCompiling model...")
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        print("  Compiled with Adam optimizer")
        print(f"  Learning rate: {learning_rate}")
        
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=50, batch_size=32, patience=10):
        """Train the model."""
        print("\nTraining model...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining complete!")
        
    def evaluate(self, X_test, y_test):
        """Evaluate model on test set."""
        print("\nEvaluating on test set...")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"  Test Loss (MSE): {results[0]:.4f}")
        print(f"  Test MAE: {results[1]:.4f}")
        print(f"  Test MSE: {results[2]:.4f}")
        
        return results
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def save_model(self, path='models/watershed_model.h5'):
        """Save trained model."""
        import os
        os.makedirs('models', exist_ok=True)
        self.model.save(path)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='models/watershed_model.h5'):
        """Load trained model."""
        self.model = keras.models.load_model(path)
        print(f"\nModel loaded from {path}")
    
    def plot_training_history(self, save_path='results/training_history.png'):
        """Plot training history."""
        import os
        os.makedirs('results', exist_ok=True)
        
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE', fontsize=12)
        ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
        plt.close()


def train_watershed_model():
    """Complete training pipeline."""
    print("="*60)
    print("WATERSHED MODEL TRAINING")
    print("="*60)
    
    # 1. Prepare data
    preprocessor = WatershedDataPreprocessor('data/synthetic_watershed.csv')
    splits = preprocessor.prepare(sequence_length=7)
    
    # 2. Build model
    model = WatershedLSTMModel(sequence_length=7, n_features=8)
    model.build_model(lstm_units=64, dense_units=32, dropout_rate=0.2)
    model.compile_model(learning_rate=0.001)
    
    # 3. Train
    model.train(
        splits['X_train'], splits['y_train'],
        splits['X_val'], splits['y_val'],
        epochs=50,
        batch_size=32,
        patience=10
    )
    
    # 4. Evaluate
    model.evaluate(splits['X_test'], splits['y_test'])
    
    # 5. Save
    model.save_model('models/watershed_model.h5')
    model.plot_training_history()
    
    # Save preprocessor scalers
    import pickle
    import os
    os.makedirs('models', exist_ok=True)
    with open('models/scalers.pkl', 'wb') as f:
        pickle.dump({
            'features': preprocessor.scaler_features,
            'target': preprocessor.scaler_target
        }, f)
    print("Scalers saved to models/scalers.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nSaved files:")
    print("  - models/watershed_model.h5 (trained model)")
    print("  - models/scalers.pkl (feature/target scalers)")
    print("  - results/training_history.png (training curves)")
    print("\nNext steps:")
    print("  1. Make predictions on test set")
    print("  2. Identify source/sink segments")
    print("  3. Visualize results")
    print("="*60)
    
    return model, preprocessor, splits


if __name__ == "__main__":
    # Run training
    model, preprocessor, splits = train_watershed_model()

