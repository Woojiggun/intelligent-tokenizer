#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test decoding issue
"""

# Test the decoding issue
test_bytes = [239, 191, 189, 239, 191, 189]  # The problematic bytes from the screenshot

try:
    result = bytes(test_bytes).decode('utf-8', errors='replace')
    print(f"Decoded: {result}")
    print(f"Decoded repr: {repr(result)}")
except Exception as e:
    print(f"Error: {e}")

# Test with Korean text
korean_text = "안녕하세요"
encoded = list(korean_text.encode('utf-8'))
print(f"\nKorean text: {korean_text}")
print(f"Encoded bytes: {encoded}")

# Try to decode
decoded = bytes(encoded).decode('utf-8')
print(f"Decoded back: {decoded}")

# Test what happens with model predictions
# Simulating wrong predictions
wrong_predictions = [239, 191, 189] * 10  # All replacement characters
decoded_wrong = bytes(wrong_predictions).decode('utf-8', errors='replace')
print(f"\nWrong predictions decoded: {repr(decoded_wrong)}")
print(f"Display: {decoded_wrong}")