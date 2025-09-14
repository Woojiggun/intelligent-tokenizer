#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test display issue - simulate what happens in Gradio
"""

import torch

# Simulate the ByteTokenizer decode function from app.py
def decode_method1(ids):
    """Current method in app.py"""
    filtered = []
    for id in ids:
        if isinstance(id, (int, torch.Tensor)):
            if torch.is_tensor(id):
                id = id.item()
            if 0 <= id < 256:
                filtered.append(id)
    
    if not filtered:
        return "[Empty result]"
        
    try:
        result = bytes(filtered).decode('utf-8', errors='ignore')
        return result if result else "[Decoding failed]"
    except Exception as e:
        return f"[Error: {str(e)}]"

def decode_method2(ids):
    """Alternative method with replace"""
    filtered = []
    for id in ids:
        if isinstance(id, (int, torch.Tensor)):
            if torch.is_tensor(id):
                id = id.item()
            if 0 <= id < 256:
                filtered.append(id)
    
    if not filtered:
        return "[Empty result]"
        
    try:
        result = bytes(filtered).decode('utf-8', errors='replace')
        return result
    except Exception as e:
        return f"[Error: {str(e)}]"

# Test with different scenarios
test_cases = [
    # English (should work)
    ([84, 104, 101, 32, 113, 117, 105, 99, 107], "The quick"),
    
    # Korean with some errors
    ([236, 136, 136, 235, 133, 149], "Korean with errors"),
    
    # Mixed with special tokens
    ([72, 101, 108, 108, 111, 258, 259], "Hello + special tokens"),
    
    # Invalid UTF-8 sequence
    ([239, 191, 189, 239, 191, 189], "Replacement chars"),
]

print("Testing different decode methods:")
print("="*60)

for bytes_list, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Bytes: {bytes_list}")
    
    # Convert to tensor for testing
    tensor_ids = torch.tensor(bytes_list)
    
    # Test both methods
    result1 = decode_method1(tensor_ids)
    result2 = decode_method2(tensor_ids)
    
    print(f"Method 1 (ignore): '{result1}'")
    print(f"Method 2 (replace): '{result2}'")
    
    # Show hex representation
    valid_bytes = [b for b in bytes_list if 0 <= b < 256]
    print(f"Hex: {' '.join(f'{b:02x}' for b in valid_bytes[:10])}")
    
    # Check if it contains replacement character
    if '\ufffd' in result2:
        count = result2.count('\ufffd')
        print(f"Contains {count} replacement characters")

print("\n" + "="*60)
print("ISSUE FOUND:")
print("The problem is that Korean bytes [236, 136, 136] create invalid UTF-8")
print("This produces � (replacement character) which may not display properly")
print("in Gradio's markdown renderer, especially on Windows/CP949 systems")