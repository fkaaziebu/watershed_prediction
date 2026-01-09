"""
generate_synthetic_data.py
===========================
Creates synthetic watershed data with known sources and sinks.

This script generates:
1. River network structure (15 segments)
2. Time series of pollution measurements (365 days)
3. Known source and sink locations
4. Spatial and temporal features

Author: MedTrack
Date: January 2026
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)

class WatershedGenerator:
    """Generate synthetic watershed with pollution dynamics."""
    
    def __init__(self, n_segments=15, n_days=365):
        self.n_segments = n_segments
        self.n_days = n_days
        self.segments = []
        self.network = nx.DiGraph()
        
    def create_network(self):
        """Create river network structure."""
        print("Creating river network...")
        
        # Define network structure (upstream -> downstream)
        # Simple dendritic (tree-like) pattern
        edges = [
            # Headwaters -> tributaries
            ('S1', 'S4'),  # Headwater 1
            ('S2', 'S4'),  # Headwater 2
            ('S3', 'S5'),  # Headwater 3
            
            # Tributaries -> main stem
            ('S4', 'S7'),  # Tributary confluence
            ('S5', 'S8'),  # Tributary confluence
            ('S6', 'S8'),  # Another tributary
            
            # Main stem
            ('S7', 'S9'),
            ('S8', 'S9'),
            ('S9', 'S11'),
            ('S10', 'S11'),  # Side tributary
            
            # Lower reach
            ('S11', 'S13'),
            ('S12', 'S13'),  # Last tributary
            ('S13', 'S14'),
            ('S14', 'S15'),  # Outlet
        ]
        
        self.network.add_edges_from(edges)
        
        # Assign segment properties
        segment_properties = {
            'S1': {'type': 'headwater', 'land_use': 'forest', 'lat': 40.0, 'lon': -75.0},
            'S2': {'type': 'headwater', 'land_use': 'forest', 'lat': 40.1, 'lon': -75.1},
            'S3': {'type': 'headwater', 'land_use': 'agriculture', 'lat': 40.0, 'lon': -74.9},
            'S4': {'type': 'tributary', 'land_use': 'forest', 'lat': 39.9, 'lon': -75.0},
            'S5': {'type': 'tributary', 'land_use': 'agriculture', 'lat': 39.9, 'lon': -74.9},
            'S6': {'type': 'tributary', 'land_use': 'urban', 'lat': 39.8, 'lon': -74.8},
            'S7': {'type': 'main_stem', 'land_use': 'forest', 'lat': 39.8, 'lon': -75.0},
            'S8': {'type': 'main_stem', 'land_use': 'wetland', 'lat': 39.7, 'lon': -74.9},
            'S9': {'type': 'main_stem', 'land_use': 'agriculture', 'lat': 39.6, 'lon': -74.9},
            'S10': {'type': 'tributary', 'land_use': 'urban', 'lat': 39.6, 'lon': -74.8},
            'S11': {'type': 'main_stem', 'land_use': 'urban', 'lat': 39.5, 'lon': -74.9},
            'S12': {'type': 'tributary', 'land_use': 'forest', 'lat': 39.5, 'lon': -75.0},
            'S13': {'type': 'lower_reach', 'land_use': 'urban', 'lat': 39.4, 'lon': -74.9},
            'S14': {'type': 'lower_reach', 'land_use': 'urban', 'lat': 39.3, 'lon': -74.9},
            'S15': {'type': 'outlet', 'land_use': 'urban', 'lat': 39.2, 'lon': -74.9},
        }
        
        nx.set_node_attributes(self.network, segment_properties)
        
        print(f"  Created network with {self.network.number_of_nodes()} segments")
        print(f"  and {self.network.number_of_edges()} connections")
        
    def define_sources_sinks(self):
        """Define known pollution sources and sinks."""
        print("\nDefining sources and sinks...")
        
        # SOURCES (add pollution)
        self.sources = {
            'S3': {
                'magnitude': 8.0,  # mg/L added
                'type': 'agricultural_runoff',
                'description': 'Farm runoff, fertilizer'
            },
            'S6': {
                'magnitude': 6.0,
                'type': 'urban_stormwater',
                'description': 'Urban runoff after rain'
            },
            'S10': {
                'magnitude': 5.0,
                'type': 'wastewater',
                'description': 'Treated wastewater discharge'
            }
        }
        
        # SINKS (remove pollution)
        self.sinks = {
            'S8': {
                'magnitude': -4.5,  # mg/L removed
                'type': 'wetland',
                'description': 'Wetland natural filtering'
            },
            'S12': {
                'magnitude': -3.0,
                'type': 'forest_buffer',
                'description': 'Forest riparian zone'
            }
        }
        
        print(f"  Defined {len(self.sources)} sources")
        print(f"  Defined {len(self.sinks)} sinks")
        
        # Store in network
        for node in self.network.nodes():
            if node in self.sources:
                self.network.nodes[node]['source_sink'] = self.sources[node]['magnitude']
                self.network.nodes[node]['role'] = 'source'
            elif node in self.sinks:
                self.network.nodes[node]['source_sink'] = self.sinks[node]['magnitude']
                self.network.nodes[node]['role'] = 'sink'
            else:
                self.network.nodes[node]['source_sink'] = 0.0
                self.network.nodes[node]['role'] = 'neutral'
    
    def simulate_pollution(self):
        """Simulate pollution propagation through network over time."""
        print("\nSimulating pollution dynamics...")
        
        # Initialize storage
        self.concentrations = np.zeros((self.n_segments, self.n_days))
        self.flow_rates = np.zeros((self.n_segments, self.n_days))
        
        # Base flow rates (cubic meters per second)
        base_flows = {
            'headwater': 2.0,
            'tributary': 5.0,
            'main_stem': 15.0,
            'lower_reach': 25.0,
            'outlet': 30.0
        }
        
        # Simulate for each day
        for day in range(self.n_days):
            # Seasonal pattern (higher flow in spring)
            seasonal_factor = 1.0 + 0.5 * np.sin(2 * np.pi * (day - 80) / 365)
            
            # Rain events (increase flow and pollution)
            is_rain = np.random.random() < 0.15  # 15% chance of rain
            rain_factor = 1.8 if is_rain else 1.0
            
            # Topological sort to process upstream -> downstream
            topo_order = list(nx.topological_sort(self.network))
            
            for idx, node in enumerate(topo_order):
                node_data = self.network.nodes[node]
                
                # Base concentration (background level)
                base_conc = 3.0  # mg/L background nitrogen
                
                # Get upstream concentrations
                predecessors = list(self.network.predecessors(node))
                
                if len(predecessors) == 0:
                    # Headwater - starts with base concentration
                    conc = base_conc
                    flow = base_flows[node_data['type']] * seasonal_factor * rain_factor
                else:
                    # Mix upstream sources
                    upstream_concs = []
                    upstream_flows = []
                    for pred in predecessors:
                        pred_idx = topo_order.index(pred)
                        upstream_concs.append(self.concentrations[pred_idx, day])
                        upstream_flows.append(self.flow_rates[pred_idx, day])
                    
                    # Flow mixing
                    total_flow = sum(upstream_flows)
                    if total_flow > 0:
                        conc = sum(c * f for c, f in zip(upstream_concs, upstream_flows)) / total_flow
                    else:
                        conc = base_conc
                    
                    flow = total_flow + base_flows.get(node_data['type'], 0) * 0.2
                
                # Add source/sink effect
                source_sink_effect = node_data['source_sink']
                
                # Sources amplified by rain
                if source_sink_effect > 0 and is_rain:
                    source_sink_effect *= 1.5
                
                # Sinks less effective at high flow
                if source_sink_effect < 0 and is_rain:
                    source_sink_effect *= 0.7
                
                conc += source_sink_effect
                
                # Add measurement noise
                conc += np.random.normal(0, 0.3)
                
                # Ensure positive
                conc = max(0, conc)
                
                # Store
                self.concentrations[idx, day] = conc
                self.flow_rates[idx, day] = flow
        
        print(f"  Simulated {self.n_days} days of data")
        print(f"  Concentration range: {self.concentrations.min():.2f} - {self.concentrations.max():.2f} mg/L")
    
    def create_dataframe(self):
        """Create pandas DataFrame with all data."""
        print("\nCreating dataset...")
        
        data_records = []
        topo_order = list(nx.topological_sort(self.network))
        
        # Land use encoding
        land_use_map = {
            'forest': 0,
            'agriculture': 1,
            'urban': 2,
            'wetland': 3
        }
        
        for day in range(self.n_days):
            date = datetime(2024, 1, 1) + timedelta(days=day)
            
            for idx, node in enumerate(topo_order):
                node_data = self.network.nodes[node]
                
                # Get upstream information
                predecessors = list(self.network.predecessors(node))
                upstream_conc = 0.0
                upstream_count = len(predecessors)
                
                if upstream_count > 0:
                    upstream_indices = [topo_order.index(p) for p in predecessors]
                    upstream_conc = np.mean([self.concentrations[i, day] for i in upstream_indices])
                
                record = {
                    'date': date,
                    'day': day,
                    'segment_id': node,
                    'segment_index': idx,
                    'latitude': node_data['lat'],
                    'longitude': node_data['lon'],
                    'land_use': node_data['land_use'],
                    'land_use_encoded': land_use_map[node_data['land_use']],
                    'segment_type': node_data['type'],
                    'upstream_count': upstream_count,
                    'upstream_concentration': upstream_conc,
                    'flow_rate': self.flow_rates[idx, day],
                    'concentration': self.concentrations[idx, day],  # Target variable
                    'true_source_sink': node_data['source_sink'],
                    'true_role': node_data['role'],
                }
                
                data_records.append(record)
        
        self.df = pd.DataFrame(data_records)
        print(f"  Created dataset with {len(self.df)} records")
        print(f"  Shape: {self.df.shape}")
        
    def save_data(self, output_dir='data'):
        """Save data to files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nSaving data to {output_dir}/...")
        
        # Save main dataset
        self.df.to_csv(f'{output_dir}/synthetic_watershed.csv', index=False)
        print(f"  Saved synthetic_watershed.csv")
        
        # Save network structure
        network_data = nx.node_link_data(self.network)
        with open(f'{output_dir}/network_structure.json', 'w') as f:
            json.dump(network_data, f, indent=2)
        print(f"  Saved network_structure.json")
        
        # Save source/sink definitions
        source_sink_info = {
            'sources': self.sources,
            'sinks': self.sinks
        }
        with open(f'{output_dir}/source_sink_definitions.json', 'w') as f:
            json.dump(source_sink_info, f, indent=2)
        print(f"  Saved source_sink_definitions.json")
        
        # Save summary statistics
        summary = self.df.groupby('segment_id')['concentration'].agg(['mean', 'std', 'min', 'max'])
        summary.to_csv(f'{output_dir}/summary_statistics.csv')
        print(f"  Saved summary_statistics.csv")
    
    def visualize_network(self, save_path='data/network_visualization.png'):
        """Create network visualization."""
        print("\nCreating network visualization...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Network structure with source/sink colors
        pos = {node: (data['lon'], data['lat']) 
               for node, data in self.network.nodes(data=True)}
        
        # Color by role
        color_map = {
            'source': 'red',
            'sink': 'blue',
            'neutral': 'lightgray'
        }
        node_colors = [color_map[data['role']] for _, data in self.network.nodes(data=True)]
        
        nx.draw(self.network, pos, ax=ax1,
                node_color=node_colors,
                node_size=800,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=20,
                edge_color='gray',
                width=2)
        
        ax1.set_title('River Network Structure\n(Red=Source, Blue=Sink, Gray=Neutral)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Source (adds pollution)'),
            Patch(facecolor='blue', label='Sink (removes pollution)'),
            Patch(facecolor='lightgray', label='Neutral')
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        
        # Plot 2: Average concentration by segment
        avg_conc = self.df.groupby('segment_id')['concentration'].mean()
        topo_order = list(nx.topological_sort(self.network))
        
        ax2.bar(range(len(topo_order)), 
                [avg_conc[node] for node in topo_order],
                color=[color_map[self.network.nodes[node]['role']] for node in topo_order],
                alpha=0.7)
        ax2.set_xticks(range(len(topo_order)))
        ax2.set_xticklabels(topo_order, rotation=45)
        ax2.set_xlabel('Segment ID')
        ax2.set_ylabel('Average Concentration (mg/L)')
        ax2.set_title('Average Nitrogen Concentration by Segment', 
                     fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved visualization to {save_path}")
        plt.close()
    
    def visualize_time_series(self, save_path='data/time_series_examples.png'):
        """Create time series examples."""
        print("\nCreating time series visualization...")
        
        # Select interesting segments
        examples = ['S1', 'S3', 'S8', 'S10', 'S15']  # Headwater, source, sink, source, outlet
        
        fig, axes = plt.subplots(len(examples), 1, figsize=(14, 10))
        
        for idx, segment in enumerate(examples):
            segment_data = self.df[self.df['segment_id'] == segment]
            role = self.network.nodes[segment]['role']
            
            axes[idx].plot(segment_data['day'], 
                          segment_data['concentration'],
                          linewidth=1.5,
                          color={'source': 'red', 'sink': 'blue', 'neutral': 'gray'}[role])
            
            axes[idx].set_ylabel('Conc. (mg/L)', fontsize=10)
            axes[idx].set_title(f'{segment} ({role.upper()})', 
                              fontsize=11, fontweight='bold')
            axes[idx].grid(alpha=0.3)
            
            if idx == len(examples) - 1:
                axes[idx].set_xlabel('Day', fontsize=10)
        
        plt.suptitle('Time Series of Nitrogen Concentration', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved time series to {save_path}")
        plt.close()

def main():
    """Main execution function."""
    print("="*60)
    print("WATERSHED SYNTHETIC DATA GENERATOR")
    print("="*60)
    
    # Create generator
    generator = WatershedGenerator(n_segments=15, n_days=365)
    
    # Generate data
    generator.create_network()
    generator.define_sources_sinks()
    generator.simulate_pollution()
    generator.create_dataframe()
    
    # Save outputs
    generator.save_data()
    generator.visualize_network()
    generator.visualize_time_series()
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/synthetic_watershed.csv (main dataset)")
    print("  - data/network_structure.json (network topology)")
    print("  - data/source_sink_definitions.json (ground truth)")
    print("  - data/summary_statistics.csv (segment summaries)")
    print("  - data/network_visualization.png (network map)")
    print("  - data/time_series_examples.png (time series plots)")
    print("\nNext steps:")
    print("  1. Explore the data in data/synthetic_watershed.csv")
    print("  2. Build your TensorFlow model")
    print("  3. Train and predict")
    print("  4. Compare predictions with ground truth!")
    print("="*60)

if __name__ == "__main__":
    main()

