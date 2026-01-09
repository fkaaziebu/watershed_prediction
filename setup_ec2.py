#!/usr/bin/env python3
"""
setup_ec2.py
============
Script to set up the watershed prediction project on an EC2 instance.

This script:
1. Downloads project files from the GitHub repository
2. Creates necessary directories
3. Installs Python dependencies

Usage:
    python3 setup_ec2.py

Author: MedTrack
Date: January 2026
"""

import os
import subprocess
import sys
import urllib.request
import urllib.error

# GitHub raw file URLs
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/fkaaziebu/watershed_prediction/main"

FILES_TO_DOWNLOAD = [
    "generate_synthetic_data.py",
    "main.py",
    "requirements.txt",
    "source_sink_analysis.py",
    "watershed_model.py",
]

# Directories to create
DIRECTORIES = [
    "data",
    "models",
    "results",
]


def print_header(message):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {message}")
    print("=" * 60)


def print_status(success, message):
    """Print status message."""
    symbol = "[OK]" if success else "[FAIL]"
    print(f"  {symbol} {message}")


def download_file(filename, dest_dir="."):
    """Download a file from the GitHub repository."""
    url = f"{GITHUB_RAW_BASE}/{filename}"
    dest_path = os.path.join(dest_dir, filename)

    try:
        urllib.request.urlretrieve(url, dest_path)
        print_status(True, f"Downloaded {filename}")
        return True
    except urllib.error.URLError as e:
        print_status(False, f"Failed to download {filename}: {e}")
        return False
    except Exception as e:
        print_status(False, f"Error downloading {filename}: {e}")
        return False


def create_directories():
    """Create necessary project directories."""
    print_header("Creating Directories")

    for directory in DIRECTORIES:
        try:
            os.makedirs(directory, exist_ok=True)
            print_status(True, f"Created directory: {directory}/")
        except Exception as e:
            print_status(False, f"Failed to create {directory}/: {e}")
            return False

    return True


def download_project_files():
    """Download all project files from GitHub."""
    print_header("Downloading Project Files")

    success_count = 0
    for filename in FILES_TO_DOWNLOAD:
        if download_file(filename):
            success_count += 1

    print(f"\n  Downloaded {success_count}/{len(FILES_TO_DOWNLOAD)} files")
    return success_count == len(FILES_TO_DOWNLOAD)


def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print_header("Installing Dependencies")

    if not os.path.exists("requirements.txt"):
        print_status(False, "requirements.txt not found")
        return False

    try:
        print("  Running: pip install -r requirements.txt")
        print("  This may take a few minutes...\n")

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print_status(True, "Dependencies installed successfully")
            return True
        else:
            print_status(False, "Failed to install dependencies")
            print(f"  Error: {result.stderr}")
            return False

    except Exception as e:
        print_status(False, f"Error installing dependencies: {e}")
        return False


def verify_installation():
    """Verify that all files are in place."""
    print_header("Verifying Installation")

    all_good = True

    # Check files
    for filename in FILES_TO_DOWNLOAD:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print_status(True, f"{filename} ({size} bytes)")
        else:
            print_status(False, f"{filename} missing")
            all_good = False

    # Check directories
    for directory in DIRECTORIES:
        if os.path.isdir(directory):
            print_status(True, f"{directory}/ directory exists")
        else:
            print_status(False, f"{directory}/ directory missing")
            all_good = False

    return all_good


def print_next_steps():
    """Print instructions for next steps."""
    print_header("Setup Complete!")

    print("""
  Next steps:

  1. Run the complete pipeline:
     python3 main.py

  2. Or run individual steps:
     python3 generate_synthetic_data.py   # Generate data
     python3 watershed_model.py           # Train model
     python3 source_sink_analysis.py      # Analyze results

  3. Check results in:
     - data/      : Generated watershed data
     - models/    : Trained model files
     - results/   : Analysis outputs and visualizations

  Note: The full pipeline takes approximately 10-15 minutes.
""")


def main():
    """Main setup function."""
    print_header("EC2 Setup for Watershed Prediction")
    print(f"\n  Working directory: {os.getcwd()}")
    print(f"  Python version: {sys.version.split()[0]}")

    # Step 1: Create directories
    if not create_directories():
        print("\nSetup failed at directory creation step.")
        sys.exit(1)

    # Step 2: Download files
    if not download_project_files():
        print("\nWarning: Some files failed to download.")
        print("Please check your internet connection and try again.")

    # Step 3: Install dependencies
    install_deps = input("\n  Install Python dependencies? [Y/n]: ").strip().lower()
    if install_deps != 'n':
        if not install_dependencies():
            print("\nWarning: Dependency installation had issues.")
            print("You may need to install packages manually.")

    # Step 4: Verify
    verify_installation()

    # Step 5: Next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
