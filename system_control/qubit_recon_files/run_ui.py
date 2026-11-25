#!/usr/bin/env python3
"""
Launcher script for the Entangle Sequence UI
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []

    # Check PyQt5
    try:
        import PyQt5
        print("✓ PyQt5 available")
    except ImportError:
        missing_deps.append("PyQt5")
        print("✗ PyQt5 not available")

    # Check NumPy
    try:
        import numpy
        print("✓ NumPy available")
    except ImportError:
        missing_deps.append("numpy")
        print("✗ NumPy not available")

    # Check Matplotlib
    try:
        import matplotlib
        print("✓ Matplotlib available")
    except ImportError:
        missing_deps.append("matplotlib")
        print("✗ Matplotlib not available")

    # Check local modules
    try:
        from entangle_sequence import EntangleConfig, generate_entangle_sequence
        print("✓ entangle_sequence available")
    except ImportError as e:
        print(f"✗ entangle_sequence not available: {e}")
        return False

    if missing_deps:
        print(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False

    return True

def main():
    print("Entangle Sequence UI Launcher")
    print("=" * 30)

    # Check dependencies
    if not check_dependencies():
        print("\nCannot run UI due to missing dependencies.")
        return 1

    # Try to import and run the UI
    try:
        from ui_entangle import main as ui_main
        print("\nStarting UI...")
        ui_main()
    except Exception as e:
        print(f"Failed to start UI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())