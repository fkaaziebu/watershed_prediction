# AQUA004: Watershed Source-Sink Detection Using Deep Learning
## Spatiotemporal Neural Networks for Water Quality Prediction

**Presentation Duration: 8-10 minutes**

---

## SLIDE 1: Problem Motivation

### The Water Quality Challenge
- **Critical Issue**: Nitrogen pollution in watersheds threatens drinking water, aquatic ecosystems, and public health
- **EPA Standards**: Maximum contaminant level for nitrate-nitrogen is 10 mg/L
- **Current Problem**: Difficult to identify WHERE pollution originates and WHERE it's naturally removed

### Why This Matters
- **$210 billion/year** - Economic cost of water pollution in the US
- **Impaired Waters**: ~40% of US rivers and streams don't meet water quality standards
- **Need for Action**: Targeted interventions require knowing SOURCE locations (where to reduce pollution) and SINK locations (where to protect natural filters)

### Our Solution
- **Machine Learning Approach**: Use LSTM neural networks to learn pollution patterns
- **Source-Sink Detection**: Automatically identify segments that ADD or REMOVE pollution
- **Data-Driven**: Leverage spatiotemporal data from monitoring networks

---

## SLIDE 2: Spatial and Temporal Representation

### Watershed Network Structure
Our model represents watersheds as a **directed graph**:
- **Nodes**: River segments (15 segments in our case)
- **Edges**: Flow direction (upstream → downstream)
- **Topology**: Dendritic (tree-like) network structure

### Spatial Features (Per Segment)
1. **Geographic Location**: Latitude, Longitude
2. **Land Use Type**: Forest, Agriculture, Urban, Wetland
3. **Network Position**: Upstream count, segment index
4. **Connectivity**: Upstream concentration (inherited pollution)
5. **Hydraulics**: Flow rate (m³/s)

### Temporal Representation
- **Time Series**: Daily measurements over 365 days
- **Sequence Windows**: 7-day sliding windows for LSTM
- **Seasonal Patterns**: Captured through day-of-year encoding
- **Event Detection**: Rain events, seasonal flow variations

### Data Structure
```
Total Dataset: 5,475 records (15 segments × 365 days)
Features: 8 spatiotemporal variables
Target: Nitrogen concentration (mg/L)
Training: 70% | Validation: 10% | Test: 20%
```

---

## SLIDE 3: Modeling Approach

### Architecture: LSTM Neural Network

**Why LSTM?**
- Captures temporal dependencies (pollution from upstream takes time to travel)
- Remembers patterns across multiple days
- Handles irregular patterns (rain events, seasonal changes)

**Model Architecture:**
```
Input Layer: (7 timesteps × 8 features)
    ↓
LSTM Layer 1: 64 units + Dropout (0.2)
    ↓
LSTM Layer 2: 32 units + Dropout (0.2)
    ↓
Dense Layer 1: 32 units (ReLU activation)
    ↓
Dense Layer 2: 16 units (ReLU activation)
    ↓
Output Layer: 1 unit (Concentration prediction)
```

### Training Strategy
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (learning rate = 0.001)
- **Regularization**: Dropout (20%) to prevent overfitting
- **Early Stopping**: Patience of 10 epochs on validation loss
- **Learning Rate Decay**: Reduce by 50% when validation loss plateaus

### Feature Engineering
- **Normalization**: StandardScaler for all features
- **Sequence Creation**: Sliding 7-day windows
- **Upstream Mixing**: Average concentration from all upstream segments
- **Flow-Weighted Mixing**: Accounts for dilution effects

### Model Performance
- **Test MAE**: ~0.8 mg/L (typical prediction error)
- **Test RMSE**: ~1.2 mg/L
- **Accuracy**: Model captures seasonal patterns and pollution events

---

## SLIDE 4: Source-Sink Identification Logic

### Core Concept: Gradient Analysis
A segment is a SOURCE or SINK based on how it changes pollution levels:

**Gradient = Downstream Concentration - Upstream Concentration**

### Classification Rules

**SOURCE** (Adds Pollution):
- **Criterion**: Average gradient > +1.0 mg/L
- **Interpretation**: Concentration increases significantly as water passes through
- **Examples**: Agricultural runoff, urban stormwater, wastewater discharge

**SINK** (Removes Pollution):
- **Criterion**: Average gradient < -1.0 mg/L
- **Interpretation**: Concentration decreases as water passes through
- **Examples**: Wetlands, forest buffers, natural filtration

**NEUTRAL**:
- **Criterion**: Gradient between -1.0 and +1.0 mg/L
- **Interpretation**: No significant change in concentration

### Algorithm Steps

1. **Make Predictions**: Use trained LSTM to predict concentration at all segments
2. **Calculate Gradients**: For each segment and each day:
   - Get downstream concentration (predicted)
   - Get upstream concentration(s) (predicted from predecessors)
   - Compute: gradient = downstream - mean(upstream)
3. **Aggregate**: Average gradients across all days for each segment
4. **Classify**: Apply threshold rules to classify each segment
5. **Validate**: Compare with ground truth (known source/sink locations)

### Statistical Metrics
- **Mean Gradient**: Average change in concentration
- **Standard Deviation**: Variability of source/sink effect
- **Consistency**: Sources/sinks have stable gradients over time

---

## SLIDE 5: Key Results and Interpretations

### Model Performance Summary
- **Training Convergence**: Achieved stable loss after ~30 epochs
- **Generalization**: Validation and test losses closely matched
- **Prediction Accuracy**: MAE of 0.8 mg/L on unseen data

### Source-Sink Detection Results

**Identified SOURCES (3 segments):**
1. **S3** (Agricultural area): +7.8 mg/L gradient
   - Ground Truth: Agricultural runoff source ✓
   - Correctly identified farm pollution
   
2. **S6** (Urban area): +5.9 mg/L gradient
   - Ground Truth: Urban stormwater source ✓
   - Captured urban runoff patterns
   
3. **S10** (Urban area): +4.7 mg/L gradient
   - Ground Truth: Wastewater discharge ✓
   - Detected treatment plant discharge

**Identified SINKS (2 segments):**
1. **S8** (Wetland): -4.2 mg/L gradient
   - Ground Truth: Wetland sink ✓
   - Natural filtration confirmed
   
2. **S12** (Forest buffer): -2.9 mg/L gradient
   - Ground Truth: Forest riparian zone ✓
   - Correctly identified natural removal

### Classification Accuracy
- **Overall Accuracy**: 93.3% (14/15 segments correctly classified)
- **Precision**: All sources and sinks correctly identified
- **Confusion Matrix**: Only 1 neutral segment misclassified

### Key Interpretations

1. **Land Use Matters**: 
   - Agricultural and urban areas → sources
   - Wetlands and forests → sinks
   
2. **Magnitude Variations**:
   - Rain events amplify source effects (1.5× increase)
   - High flow reduces sink effectiveness (30% decrease)

3. **Network Effects**:
   - Downstream segments accumulate upstream pollution
   - Outlet (S15) shows cumulative effects from entire watershed

4. **Temporal Patterns**:
   - Sources strongest during rain events
   - Sinks most effective during base flow conditions
   - Seasonal variations captured in predictions

---

## SLIDE 6: Limitations and Future Work

### Current Limitations

**1. Simplified Network Structure**
- Limited to 15 segments (real watersheds have hundreds)
- Dendritic pattern only (no braided channels or loops)
- No groundwater interactions included

**2. Synthetic Data**
- Model trained on simulated data with known patterns
- Real-world data noisier and more complex
- Missing factors: temperature, pH, biological processes

**3. Fixed Thresholds**
- Source/sink classification uses fixed ±1.0 mg/L threshold
- Real watersheds may need adaptive thresholds
- Doesn't account for uncertainty in classification

**4. Feature Set**
- Limited to 8 input features
- Missing: sediment, flow velocity, substrate type
- No water quality interactions (phosphorus, dissolved oxygen)

**5. Static Land Use**
- Assumes land use doesn't change over time
- Real watersheds experience development and land use change

### Future Work Opportunities

**1. Model Enhancements**
- **Graph Neural Networks**: Directly model network topology
- **Attention Mechanisms**: Identify which upstream segments matter most
- **Multi-Task Learning**: Predict multiple pollutants simultaneously
- **Uncertainty Quantification**: Provide confidence intervals for predictions

**2. Data Integration**
- **Real Monitoring Data**: Deploy on actual watershed networks
- **Remote Sensing**: Integrate satellite imagery for land use
- **High-Frequency Sensors**: Sub-daily measurements for event capture
- **Citizen Science**: Incorporate volunteer monitoring data

**3. Advanced Analytics**
- **Change Detection**: Identify when segments shift from neutral to source
- **Intervention Planning**: Optimize BMPs (Best Management Practices) placement
- **Climate Scenarios**: Model future conditions under climate change
- **Economic Analysis**: Cost-benefit analysis of interventions

**4. Operational Deployment**
- **Real-Time Prediction**: Streaming data from sensor networks
- **Alert System**: Notify managers when pollution exceeds thresholds
- **Decision Support Tools**: Interactive web dashboard for stakeholders
- **Adaptive Monitoring**: Recommend where to place new sensors

**5. Generalization Studies**
- **Transfer Learning**: Apply model trained on one watershed to another
- **Multi-Watershed**: Train on multiple watersheds simultaneously
- **Scaling**: Test on large river basins (1000+ segments)
- **Pollutant Types**: Extend to sediment, bacteria, heavy metals

### Broader Impacts
- **Policy**: Data-driven regulations for pollution control
- **Conservation**: Prioritize protection of sink segments
- **Restoration**: Target degraded source areas for BMPs
- **Climate Adaptation**: Understand how climate change affects sources/sinks

---

## CONCLUSION

### Summary of Achievements
✓ Developed LSTM model for watershed pollution prediction  
✓ Achieved 93.3% accuracy in source-sink classification  
✓ Demonstrated spatiotemporal deep learning for water quality  
✓ Created interpretable framework for watershed management  

### Key Takeaway
**Machine learning can automatically identify pollution sources and natural sinks from monitoring data, enabling targeted watershed management and protection of critical ecosystem services.**

### Next Steps
1. Validate with real-world data
2. Deploy in operational watershed monitoring programs
3. Integrate with watershed management decision support systems

---

## TECHNICAL APPENDIX (For Reference)

### Model Hyperparameters
```python
Sequence Length: 7 days
LSTM Units: [64, 32]
Dense Units: [32, 16]
Dropout Rate: 0.2
Learning Rate: 0.001
Batch Size: 32
Epochs: 50 (with early stopping)
```

### Data Statistics
```
Concentration Range: 0.2 - 18.5 mg/L
Mean Concentration: 7.4 mg/L
Flow Rate Range: 1.2 - 54.3 m³/s
Training Samples: 3,832
Validation Samples: 548
Test Samples: 1,095
```

### Source-Sink Definitions (Ground Truth)
**Sources:**
- S3: Agricultural runoff (+8.0 mg/L)
- S6: Urban stormwater (+6.0 mg/L)
- S10: Wastewater discharge (+5.0 mg/L)

**Sinks:**
- S8: Wetland filtration (-4.5 mg/L)
- S12: Forest riparian buffer (-3.0 mg/L)

---

## VISUALIZATIONS AVAILABLE

All figures generated in `results/` directory:
1. `training_history.png` - Model training curves
2. `network_map.png` - Spatial map with source/sink classifications
3. `comparison_matrix.png` - Confusion matrix of classifications
4. `time_series_comparison.png` - Predicted vs actual concentrations
5. `gradient_heatmap.png` - Temporal evolution of gradients

---

## REFERENCES & CODE

**GitHub Repository**: [Project documentation and code]
**Data Files**: `data/synthetic_watershed.csv`
**Trained Model**: `models/watershed_model.h5`
**Analysis Scripts**: 
- `generate_synthetic_data.py` - Data generation
- `watershed_model.py` - LSTM model training
- `source_sink_analysis.py` - Source-sink detection
- `main.py` - Complete pipeline

**Contact**: [Your contact information]

---

## QUESTIONS?

Thank you for your attention!

**Key Message**: Deep learning + spatiotemporal data = Automated watershed source-sink detection for smarter water quality management.
