#!/usr/bin/env python3
import sys
import os

# Add the parent directory to sys.path so 'polyfuse' can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from polyfuse.cli import main

if __name__ == '__main__':
    main()
