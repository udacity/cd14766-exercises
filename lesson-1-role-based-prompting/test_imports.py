#!/usr/bin/env python3
"""Test script to verify all imports work correctly for Lesson 1."""

import sys
import os

def test_imports():
    """Test all required imports for the lesson."""
    
    print("Testing Lesson 1 imports...")
    print("-" * 50)
    
    # Test 1: Basic imports
    try:
        import os
        import json
        import time
        from typing import Dict, List, Tuple
        print("✅ Standard library imports: OK")
    except Exception as e:
        print(f"❌ Standard library imports failed: {e}")
        return False
    
    # Test 2: Google GenAI imports
    try:
        from google import genai
        print("✅ google.genai import: OK")
    except Exception as e:
        print(f"❌ google.genai import failed: {e}")
        return False
    
    # Test 3: GenerateContentConfig import
    try:
        from google.genai.types import GenerateContentConfig
        print("✅ GenerateContentConfig import: OK")
    except Exception as e:
        print(f"❌ GenerateContentConfig import failed: {e}")
        return False
    
    # Test 4: Google genai types import
    try:
        from google.genai import types
        print("✅ google.genai.types import: OK")
    except Exception as e:
        print(f"❌ google.genai.types import failed: {e}")
        return False
    
    # Test 5: Test client creation
    try:
        from google.genai import Client
        print("✅ google.genai.Client import: OK")
    except Exception as e:
        print(f"❌ google.genai.Client import failed: {e}")
        return False
    
    # Test 6: Test the starter file
    print("\n" + "-" * 50)
    print("Testing starter/personas_with_ai.py...")
    sys.path.insert(0, 'exercises/starter')
    try:
        import personas_with_ai
        print("✅ starter/personas_with_ai.py imports: OK")
    except Exception as e:
        print(f"❌ starter/personas_with_ai.py import failed: {e}")
        return False
    
    # Test 7: Test the solution file
    print("\n" + "-" * 50)
    print("Testing solution/personas_with_ai.py...")
    sys.path[0] = 'exercises/solution'
    try:
        import personas_with_ai as solution_personas
        print("✅ solution/personas_with_ai.py imports: OK")
    except Exception as e:
        print(f"❌ solution/personas_with_ai.py import failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All imports working correctly!")
    print("\nIf you're seeing import errors in your IDE:")
    print("1. Make sure you've installed requirements:")
    print("   pip install -r exercises/requirements.txt")
    print("2. Configure your IDE's Python interpreter")
    print("3. Restart your IDE/language server")
    
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)