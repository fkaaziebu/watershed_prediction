"""
source_sink_analysis.py
=======================
Analyzes model predictions to identify sources and sinks in the watershed.

This script:
1. Loads trained model and makes predictions
2. Calculates upstream-downstream gradients
3. Identifies source and sink segments
4. Creates comprehensive visualizations
5. Compares with ground truth

Author: MedTrack
Date: January 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import pickle
from tensorflow import keras

class SourceSinkDetector:
    """Detect source and sink behavior from model predictions."""

    def __init__(self, model_path='models/watershed_model.h5',
                 scalers_path='models/scalers.pkl',
                 data_path='data/synthetic_watershed.csv',
                 network_path='data/network_structure.json'):
        """Initialize detector."""
        print("Initializing Source-Sink Detector...")

        # Load model (compile=False to avoid metric deserialization issues)
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
        print(f"  Loaded model from {model_path}")

        # Load scalers
        with open(scalers_path, 'rb') as f:
            scalers = pickle.load(f)
            self.scaler_features = scalers['features']
            self.scaler_target = scalers['target']
        print(f"  Loaded scalers from {scalers_path}")

        # Load data
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"  Loaded data: {len(self.df)} records")

        # Load network
        with open(network_path, 'r') as f:
            network_data = json.load(f)
            self.network = nx.node_link_graph(network_data)
        print(f"  Loaded network: {self.network.number_of_nodes()} nodes")

    def make_predictions(self, sequence_length=7):
        """Make predictions for all data."""
        print("\nMaking predictions...")

        # Feature columns (same as training)
        feature_cols = [
            'segment_index',
            'latitude',
            'longitude',
            'land_use_encoded',
            'upstream_count',
            'upstream_concentration',
            'flow_rate',
            'day'
        ]

        predictions = []
        actuals = []
        segments = []
        days = []

        # Predict for each segment
        for segment in self.df['segment_id'].unique():
            segment_data = self.df[self.df['segment_id'] == segment].sort_values('day')

            features = segment_data[feature_cols].values
            targets = segment_data['concentration'].values

            # Create sequences
            for i in range(len(segment_data) - sequence_length):
                X_seq = features[i:i+sequence_length]
                y_actual = targets[i+sequence_length]

                # Normalize
                X_scaled = self.scaler_features.transform(X_seq.reshape(-1, len(feature_cols)))
                X_scaled = X_scaled.reshape(1, sequence_length, len(feature_cols))

                # Predict
                y_pred_scaled = self.model.predict(X_scaled, verbose=0)[0, 0]
                y_pred = self.scaler_target.inverse_transform([[y_pred_scaled]])[0, 0]

                predictions.append(y_pred)
                actuals.append(y_actual)
                segments.append(segment)
                days.append(segment_data.iloc[i+sequence_length]['day'])

        # Create prediction dataframe
        self.predictions_df = pd.DataFrame({
            'segment_id': segments,
            'day': days,
            'actual': actuals,
            'predicted': predictions
        })

        print(f"  Made {len(predictions)} predictions")

        # Calculate metrics
        mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
        rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions))**2))
        print(f"  Overall MAE: {mae:.3f} mg/L")
        print(f"  Overall RMSE: {rmse:.3f} mg/L")

    def calculate_source_sink_indicators(self):
        """Calculate source-sink indicators for each segment."""
        print("\nCalculating source-sink indicators...")

        indicators = []

        for segment in self.predictions_df['segment_id'].unique():
            seg_data = self.predictions_df[self.predictions_df['segment_id'] == segment]

            # Get upstream segments
            predecessors = list(self.network.predecessors(segment))

            if len(predecessors) == 0:
                # Headwater - no upstream comparison
                continue

            # Calculate average gradient (downstream - upstream)
            gradients = []

            for day in seg_data['day'].unique():
                downstream_conc = seg_data[seg_data['day'] == day]['predicted'].values
                if len(downstream_conc) == 0:
                    continue
                downstream_conc = downstream_conc[0]

                # Get upstream concentrations
                upstream_concs = []
                for pred in predecessors:
                    pred_data = self.predictions_df[
                        (self.predictions_df['segment_id'] == pred) &
                        (self.predictions_df['day'] == day)
                    ]
                    if len(pred_data) > 0:
                        upstream_concs.append(pred_data['predicted'].values[0])

                if len(upstream_concs) > 0:
                    upstream_avg = np.mean(upstream_concs)
                    gradient = downstream_conc - upstream_avg
                    gradients.append(gradient)

            if len(gradients) > 0:
                avg_gradient = np.mean(gradients)
                std_gradient = np.std(gradients)

                # Classify
                if avg_gradient > 1.0:  # Threshold for source
                    classification = 'source'
                elif avg_gradient < -1.0:  # Threshold for sink
                    classification = 'sink'
                else:
                    classification = 'neutral'

                indicators.append({
                    'segment_id': segment,
                    'avg_gradient': avg_gradient,
                    'std_gradient': std_gradient,
                    'classification': classification,
                    'true_role': self.network.nodes[segment].get('role', 'unknown'),
                    'true_source_sink': self.network.nodes[segment].get('source_sink', 0)
                })

        self.indicators_df = pd.DataFrame(indicators)
        print(f"  Calculated indicators for {len(indicators)} segments")

        # Print summary
        print("\nClassification Summary:")
        print(self.indicators_df['classification'].value_counts())

        # Accuracy compared to ground truth
        correct = (self.indicators_df['classification'] ==
                  self.indicators_df['true_role']).sum()
        total = len(self.indicators_df)
        accuracy = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")

        return self.indicators_df

    def save_results(self, output_dir='results'):
        """Save analysis results."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving results to {output_dir}/...")

        self.predictions_df.to_csv(f'{output_dir}/predictions.csv', index=False)
        self.indicators_df.to_csv(f'{output_dir}/source_sink_indicators.csv', index=False)

        print("  Saved predictions.csv")
        print("  Saved source_sink_indicators.csv")


class SourceSinkVisualizer:
    """Create comprehensive visualizations."""

    def __init__(self, detector):
        """Initialize with detector results."""
        self.detector = detector
        self.df = detector.df
        self.predictions_df = detector.predictions_df
        self.indicators_df = detector.indicators_df
        self.network = detector.network

    def plot_network_map(self, save_path='results/network_map.png'):
        """Plot network with source/sink classifications."""
        print("\nCreating network map...")

        fig, ax = plt.subplots(figsize=(14, 10))

        # Position nodes
        pos = {node: (data['lon'], data['lat'])
               for node, data in self.network.nodes(data=True)}

        # Color by predicted role
        color_map = {'source': 'red', 'sink': 'blue', 'neutral': 'lightgray'}
        node_colors = []
        node_sizes = []

        for node in self.network.nodes():
            if node in self.indicators_df['segment_id'].values:
                role = self.indicators_df[
                    self.indicators_df['segment_id'] == node
                ]['classification'].values[0]
                node_colors.append(color_map[role])

                # Size by magnitude of effect
                gradient = self.indicators_df[
                    self.indicators_df['segment_id'] == node
                ]['avg_gradient'].values[0]
                size = 500 + abs(gradient) * 100
                node_sizes.append(size)
            else:
                node_colors.append('lightgray')
                node_sizes.append(500)

        # Draw network
        nx.draw(self.network, pos, ax=ax,
                node_color=node_colors,
                node_size=node_sizes,
                with_labels=True,
                font_size=9,
                font_weight='bold',
                arrows=True,
                arrowsize=15,
                edge_color='gray',
                width=2,
                alpha=0.8)

        ax.set_title('Watershed Source-Sink Map\n(Predicted Classifications)',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Source (adds pollution)'),
            Patch(facecolor='blue', label='Sink (removes pollution)'),
            Patch(facecolor='lightgray', label='Neutral')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved network map to {save_path}")
        plt.close()

    def plot_comparison_matrix(self, save_path='results/comparison_matrix.png'):
        """Compare predicted vs true classifications."""
        print("\nCreating comparison matrix...")

        # Create confusion matrix
        from sklearn.metrics import confusion_matrix

        true_labels = self.indicators_df['true_role'].values
        pred_labels = self.indicators_df['classification'].values

        labels = ['neutral', 'sink', 'source']
        cm = confusion_matrix(true_labels, pred_labels, labels=labels)

        fig, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels,
                   ax=ax, cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Classification', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Classification', fontsize=12, fontweight='bold')
        ax.set_title('Source-Sink Classification Accuracy',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved comparison matrix to {save_path}")
        plt.close()

    def plot_time_series_comparison(self, save_path='results/time_series_comparison.png'):
        """Plot predictions vs actuals for selected segments."""
        print("\nCreating time series comparison...")

        # Select interesting segments (sources, sinks, neutral)
        sources = self.indicators_df[
            self.indicators_df['classification'] == 'source'
        ]['segment_id'].head(2).tolist()

        sinks = self.indicators_df[
            self.indicators_df['classification'] == 'sink'
        ]['segment_id'].head(2).tolist()

        neutral = self.indicators_df[
            self.indicators_df['classification'] == 'neutral'
        ]['segment_id'].head(1).tolist()

        selected = sources + sinks + neutral

        fig, axes = plt.subplots(len(selected), 1, figsize=(14, 10))

        if len(selected) == 1:
            axes = [axes]

        for idx, segment in enumerate(selected):
            data = self.predictions_df[
                self.predictions_df['segment_id'] == segment
            ].sort_values('day')

            role = self.indicators_df[
                self.indicators_df['segment_id'] == segment
            ]['classification'].values[0]

            color = {'source': 'red', 'sink': 'blue', 'neutral': 'gray'}[role]

            axes[idx].plot(data['day'], data['actual'],
                          label='Actual', alpha=0.7, linewidth=2)
            axes[idx].plot(data['day'], data['predicted'],
                          label='Predicted', linestyle='--', linewidth=2, color=color)

            axes[idx].set_ylabel('Conc. (mg/L)', fontsize=10)
            axes[idx].set_title(f'{segment} ({role.upper()})',
                              fontsize=11, fontweight='bold')
            axes[idx].legend(loc='upper right')
            axes[idx].grid(alpha=0.3)

            if idx == len(selected) - 1:
                axes[idx].set_xlabel('Day', fontsize=10)

        plt.suptitle('Predicted vs Actual Concentrations',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved time series comparison to {save_path}")
        plt.close()

    def plot_gradient_heatmap(self, save_path='results/gradient_heatmap.png'):
        """Plot heatmap of source-sink gradients over time."""
        print("\nCreating gradient heatmap...")

        # Calculate gradients for visualization
        segments = self.indicators_df['segment_id'].tolist()
        days = sorted(self.predictions_df['day'].unique())

        # Sample days for visualization (every 10 days)
        sampled_days = days[::10]

        gradient_matrix = np.zeros((len(segments), len(sampled_days)))

        for i, segment in enumerate(segments):
            predecessors = list(self.network.predecessors(segment))
            if len(predecessors) == 0:
                continue

            for j, day in enumerate(sampled_days):
                seg_data = self.predictions_df[
                    (self.predictions_df['segment_id'] == segment) &
                    (self.predictions_df['day'] == day)
                ]

                if len(seg_data) == 0:
                    continue

                downstream = seg_data['predicted'].values[0]

                upstream_vals = []
                for pred in predecessors:
                    pred_data = self.predictions_df[
                        (self.predictions_df['segment_id'] == pred) &
                        (self.predictions_df['day'] == day)
                    ]
                    if len(pred_data) > 0:
                        upstream_vals.append(pred_data['predicted'].values[0])

                if len(upstream_vals) > 0:
                    gradient = downstream - np.mean(upstream_vals)
                    gradient_matrix[i, j] = gradient

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(gradient_matrix,
                   xticklabels=[f'D{d}' for d in sampled_days],
                   yticklabels=segments,
                   cmap='RdBu_r',
                   center=0,
                   cbar_kws={'label': 'Gradient (mg/L)'},
                   ax=ax)

        ax.set_xlabel('Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Segment ID', fontsize=12, fontweight='bold')
        ax.set_title('Source-Sink Gradient Heatmap Over Time\n(Red=Source, Blue=Sink)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved gradient heatmap to {save_path}")
        plt.close()

    def create_all_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        self.plot_network_map()
        self.plot_comparison_matrix()
        self.plot_time_series_comparison()
        self.plot_gradient_heatmap()

        print("\n" + "="*60)
        print("VISUALIZATIONS COMPLETE!")
        print("="*60)


def main():
    """Main analysis pipeline."""
    print("="*60)
    print("SOURCE-SINK ANALYSIS")
    print("="*60)

    # Initialize detector
    detector = SourceSinkDetector(
        model_path='models/watershed_model.h5',
        scalers_path='models/scalers.pkl',
        data_path='data/synthetic_watershed.csv',
        network_path='data/network_structure.json'
    )

    print("SOURCE_SINK_DETECTOR:")
    print(detector)

    # Make predictions
    detector.make_predictions(sequence_length=7)

    # Calculate source-sink indicators
    detector.calculate_source_sink_indicators()

    # Save results
    detector.save_results()

    # Create visualizations
    visualizer = SourceSinkVisualizer(detector)
    visualizer.create_all_visualizations()

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/predictions.csv")
    print("  - results/source_sink_indicators.csv")
    print("  - results/network_map.png")
    print("  - results/comparison_matrix.png")
    print("  - results/time_series_comparison.png")
    print("  - results/gradient_heatmap.png")
    print("\nNext steps:")
    print("  1. Review visualizations")
    print("  2. Write paper using these results")
    print("  3. Prepare presentation")
    print("="*60)


if __name__ == "__main__":
    main()
