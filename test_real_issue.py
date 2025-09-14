#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Find the REAL issue with display
"""

import torch
import sys
from pathlib import Path

# Simulate exact Gradio app behavior
text = "The quick brown fox jumps over the lazy dog"
print(f"Testing: {text}")

# Simulate ByteTokenizer encode
byte_seq = list(text.encode('utf-8'))
chunk = [257] + byte_seq + [258]  # Add BOS/EOS
print(f"Encoded chunk: {chunk[:20]}...")

# Simulate model predictions
# Based on screenshot: [233, 103, 58, 109, 66, 11, 216, 58, 66, 210]
predictions = [233, 103, 58, 109, 66, 11, 216, 58, 66, 210]
print(f"Model predictions: {predictions}")

# Try to decode predictions (this is what's failing)
def decode_test(ids):
    """Test decode function"""
    filtered = []
    for id in ids:
        if isinstance(id, (int, torch.Tensor)):
            if torch.is_tensor(id):
                id = id.item()
            if 0 <= id < 256:
                filtered.append(id)
    
    print(f"Filtered bytes: {filtered}")
    
    if not filtered:
        return "[Empty result]"
    
    try:
        # Try decode
        result = bytes(filtered).decode('utf-8', errors='replace')
        print(f"Decoded result: '{result}'")
        print(f"Result repr: {repr(result)}")
        
        # Check for replacement chars
        if '\ufffd' in result:
            count = result.count('\ufffd')
            print(f"Contains {count} replacement characters")
            # Replace with visible marker
            result = result.replace('\ufffd', '<?>')
        
        return result
    except Exception as e:
        return f"[Error: {e}]"

# Test decode
result = decode_test(predictions)
print(f"\nFinal display: '{result}'")

# Check what's wrong
print("\n" + "="*60)
print("PROBLEM IDENTIFIED:")
print("1. Model is predicting WRONG token IDs")
print("2. Original: [84, 104, 101, 32, ...] = 'The '")
print("3. Predicted: [233, 103, 58, ...] = garbage bytes")
print("4. 233 is > 127, not valid ASCII")
print("5. These bytes create invalid UTF-8 sequences")
print("\nThe model is NOT properly trained for English either!")