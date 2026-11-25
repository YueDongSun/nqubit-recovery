#!/usr/bin/env python3
"""
Script to run the 4-qubit entanglement UI directly.
"""

import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        # Import and run the 4-qubit UI
        from ui_entangle_4qubit import main as ui_main
        ui_main()
    except ImportError as e:
        print(f"Error importing UI module: {e}")
        print("Make sure all required files are in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error running 4-qubit UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()