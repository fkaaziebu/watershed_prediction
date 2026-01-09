"""
main.py
=======
Master script to run the complete AQUA004 pipeline.

This script orchestrates the entire workflow:
1. Generate synthetic watershed data
2. Train TensorFlow model
3. Analyze source-sink dynamics
4. Generate all visualizations

Author: MedTrack
Date: January 2026
"""

import sys
import time
from datetime import datetime

def print_header(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_step(step_num, step_name):
    """Print step information."""
    print(f"\n{'─'*70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'─'*70}\n")

def run_step(step_func, step_name):
    """Run a step and track time."""
    print_step(step_func.__name__.split('_')[1], step_name)
    start_time = time.time()

    try:
        step_func()
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.1f} seconds")
        return True
    except Exception as e:
        print(f"\n✗ Error in {step_name}: {str(e)}")
        print(f"  Please check the error message above and try again.")
        return False

def step_1_generate_data():
    """Step 1: Generate synthetic watershed data."""
    print("Generating synthetic watershed data...")
    print("This creates the river network and simulates pollution dynamics.")

    from generate_synthetic_data import main as generate_main
    generate_main()

def step_2_train_model():
    """Step 2: Train the TensorFlow model."""
    print("Training TensorFlow LSTM model...")
    print("This learns to predict pollution from spatiotemporal patterns.")

    from watershed_model import train_watershed_model
    train_watershed_model()

def step_3_analyze_results():
    """Step 3: Analyze source-sink dynamics."""
    print("Analyzing source-sink dynamics...")
    print("This identifies pollution sources and sinks in the watershed.")

    from source_sink_analysis import main as analysis_main
    analysis_main()

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    required = [
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'networkx',
        'sklearn'
    ]

    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing.append(package)
            print(f"  ✗ {package} (missing)")

    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nPlease install with:")
        print("  pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies installed")
    return True

def summarize_results():
    """Print summary of generated files."""
    import os

    print_header("PIPELINE COMPLETE!")

    print("Generated files:\n")

    # Data files
    print("📁 DATA (data/)")
    data_files = [
        "synthetic_watershed.csv",
        "network_structure.json",
        "source_sink_definitions.json",
        "network_visualization.png",
        "time_series_examples.png"
    ]
    for f in data_files:
        path = f"data/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {f} ({size/1024:.1f} KB)")
        else:
            print(f"  ✗ {f} (not found)")

    # Model files
    print("\n🤖 MODELS (models/)")
    model_files = [
        "watershed_model.h5",
        "scalers.pkl"
    ]
    for f in model_files:
        path = f"models/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {f} ({size/1024:.1f} KB)")
        else:
            print(f"  ✗ {f} (not found)")

    # Results files
    print("\n📊 RESULTS (results/)")
    result_files = [
        "predictions.csv",
        "source_sink_indicators.csv",
        "training_history.png",
        "network_map.png",
        "comparison_matrix.png",
        "time_series_comparison.png",
        "gradient_heatmap.png"
    ]
    for f in result_files:
        path = f"results/{f}"
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  ✓ {f} ({size/1024:.1f} KB)")
        else:
            print(f"  ✗ {f} (not found)")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. 📊 Review visualizations in results/ directory")
    print("2. 📈 Check model performance metrics")
    print("3. 🗺️  Examine source-sink classifications")
    print("4. 📝 Use results for your paper and presentation")
    print("5. 🎥 Record your 8-10 minute presentation")
    print("6. ✍️  Write your 3-6 page scientific paper")
    print("\n" + "="*70)

def main():
    """Run the complete pipeline."""
    start_time = time.time()

    # Print welcome message
    print_header("AQUA004 - WATERSHED SOURCE-SINK DETECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis script will run the complete analysis pipeline:")
    print("  1. Generate synthetic watershed data (15 segments, 365 days)")
    print("  2. Train TensorFlow LSTM model")
    print("  3. Analyze source-sink dynamics")
    print("  4. Generate visualizations")
    print("\nEstimated time: ~10-15 minutes")

    # Check dependencies
    print_header("CHECKING DEPENDENCIES")
    if not check_dependencies():
        print("\n⚠ Please install missing packages before continuing.")
        sys.exit(1)

    # Run pipeline steps
    steps = [
        (step_1_generate_data, "Generate Synthetic Data"),
        (step_2_train_model, "Train TensorFlow Model"),
        (step_3_analyze_results, "Analyze Source-Sink Dynamics")
    ]

    for step_func, step_name in steps:
        success = run_step(step_func, step_name)
        if not success:
            print(f"\n⚠ Pipeline stopped due to error in: {step_name}")
            sys.exit(1)

    # Calculate total time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    # Print summary
    print_header("SUMMARY")
    print(f"Total execution time: {minutes} min {seconds} sec")
    summarize_results()

    print("\n🎉 SUCCESS! All steps completed.")
    print("\nFor questions, refer to:")
    print("  - README.md (general overview)")
    print("  - AQUA004_Implementation_Guide.md (detailed guide)")
    print("  - Python files (code documentation)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
