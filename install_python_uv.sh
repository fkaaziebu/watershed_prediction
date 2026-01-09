#!/bin/bash
#
# install_python_uv.sh
# ====================
# Script to install Python and uv package manager on an EC2 instance.
#
# Supports:
#   - Amazon Linux 2 / Amazon Linux 2023
#   - Ubuntu / Debian
#   - RHEL / CentOS
#
# Usage:
#   chmod +x install_python_uv.sh
#   ./install_python_uv.sh
#
# Author: MedTrack
# Date: January 2026

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

print_status() {
    if [ "$1" = "ok" ]; then
        echo -e "  ${GREEN}[OK]${NC} $2"
    elif [ "$1" = "fail" ]; then
        echo -e "  ${RED}[FAIL]${NC} $2"
    elif [ "$1" = "info" ]; then
        echo -e "  ${YELLOW}[INFO]${NC} $2"
    fi
}

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
    elif [ -f /etc/redhat-release ]; then
        OS="rhel"
    else
        OS="unknown"
    fi
    echo "$OS"
}

install_python_amazon_linux() {
    print_header "Installing Python on Amazon Linux"

    # Update system
    print_status "info" "Updating system packages..."
    sudo yum update -y

    # Install Python 3.11 (or latest available)
    print_status "info" "Installing Python..."
    if sudo yum list python3.11 &>/dev/null; then
        sudo yum install -y python3.11 python3.11-pip python3.11-devel
        sudo alternatives --set python3 /usr/bin/python3.11 2>/dev/null || true
    else
        sudo yum install -y python3 python3-pip python3-devel
    fi

    # Install development tools (needed for some Python packages)
    sudo yum groupinstall -y "Development Tools" 2>/dev/null || \
        sudo yum install -y gcc gcc-c++ make

    print_status "ok" "Python installed"
}

install_python_ubuntu() {
    print_header "Installing Python on Ubuntu/Debian"

    # Update system
    print_status "info" "Updating system packages..."
    sudo apt-get update -y

    # Install Python and dependencies
    print_status "info" "Installing Python..."
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev

    # Install build essentials
    sudo apt-get install -y build-essential

    print_status "ok" "Python installed"
}

install_python_rhel() {
    print_header "Installing Python on RHEL/CentOS"

    # Update system
    print_status "info" "Updating system packages..."
    sudo yum update -y

    # Enable EPEL repository
    sudo yum install -y epel-release 2>/dev/null || true

    # Install Python
    print_status "info" "Installing Python..."
    sudo yum install -y python3 python3-pip python3-devel

    # Install development tools
    sudo yum groupinstall -y "Development Tools" 2>/dev/null || \
        sudo yum install -y gcc gcc-c++ make

    print_status "ok" "Python installed"
}

install_uv() {
    print_header "Installing uv Package Manager"

    print_status "info" "Downloading and installing uv..."

    # Install uv using the official installer
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"

    # Add to shell profile for future sessions
    SHELL_PROFILE=""
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_PROFILE="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    elif [ -f "$HOME/.zshrc" ]; then
        SHELL_PROFILE="$HOME/.zshrc"
    fi

    if [ -n "$SHELL_PROFILE" ]; then
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_PROFILE"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_PROFILE"
            print_status "ok" "Added uv to PATH in $SHELL_PROFILE"
        fi
    fi

    print_status "ok" "uv installed"
}

verify_installation() {
    print_header "Verifying Installation"

    # Check Python
    if command -v python3 &>/dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        print_status "ok" "Python: $PYTHON_VERSION"
    else
        print_status "fail" "Python not found"
    fi

    # Check pip
    if command -v pip3 &>/dev/null; then
        PIP_VERSION=$(pip3 --version 2>&1 | cut -d' ' -f1-2)
        print_status "ok" "pip: $PIP_VERSION"
    else
        print_status "fail" "pip not found"
    fi

    # Check uv
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        UV_VERSION=$(uv --version 2>&1)
        print_status "ok" "uv: $UV_VERSION"
    else
        print_status "fail" "uv not found"
    fi
}

print_next_steps() {
    print_header "Installation Complete!"

    echo ""
    echo "  Next steps:"
    echo ""
    echo "  1. Reload your shell or run:"
    echo "     source ~/.bashrc"
    echo ""
    echo "  2. Create a virtual environment with uv:"
    echo "     uv venv"
    echo "     source .venv/bin/activate"
    echo ""
    echo "  3. Install project dependencies:"
    echo "     uv pip install -r requirements.txt"
    echo ""
    echo "  4. Or use pip directly:"
    echo "     pip3 install -r requirements.txt"
    echo ""
    echo "  5. Run the setup script:"
    echo "     python3 setup_ec2.py"
    echo ""
}

main() {
    print_header "EC2 Python & uv Installation Script"

    # Detect OS
    OS=$(detect_os)
    print_status "info" "Detected OS: $OS"

    # Install Python based on OS
    case $OS in
        amzn|amazon)
            install_python_amazon_linux
            ;;
        ubuntu|debian)
            install_python_ubuntu
            ;;
        rhel|centos|fedora)
            install_python_rhel
            ;;
        *)
            print_status "fail" "Unsupported OS: $OS"
            echo "  Please install Python 3.9+ manually, then run this script again."
            exit 1
            ;;
    esac

    # Install uv
    install_uv

    # Verify installation
    verify_installation

    # Print next steps
    print_next_steps
}

# Run main function
main
