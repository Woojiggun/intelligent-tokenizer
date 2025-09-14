#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test reconstruction issue locally
"""

import torch
import sys
from pathlib import Path

# Add parent directory
sys.path.append(str(Path(__file__).parent.parent / "intelligent-tokenizer_v6.0"))

from src.core.byte_tokenizer_v6 import ByteTokenizerV6
from core.boundary_aware_model import BoundaryAwareTokenizerModel

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = Path("../intelligent-tokenizer_v6.0/checkpoints/unified/latest_checkpoint.pt")

print(f"Loading model from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

model = BoundaryAwareTokenizerModel(**checkpoint['model_config']).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded - Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.4f}")

# Test tokenizer
tokenizer = ByteTokenizerV6()

# Test texts
test_texts = [
    "Hello world",
    "안녕하세요",
    "The quick brown fox"
]

for text in test_texts:
    print(f"\n{'='*60}")
    print(f"Testing: {text}")
    print(f"{'='*60}")
    
    # Encode
    encoded = tokenizer.encode(text)
    input_ids = encoded['input_ids']
    print(f"Original bytes: {input_ids[:20]}...")
    
    # Model prediction
    input_tensor = torch.tensor([input_ids], device=device)
    
    with torch.no_grad():
        if len(input_ids) > 1:
            # Prepare decoder input
            decoder_input = input_tensor[:, :-1]
            labels = input_tensor[:, 1:]
            
            # Get model predictions
            outputs = model(
                input_ids=input_tensor,
                attention_mask=torch.ones_like(input_tensor),
                decoder_input_ids=decoder_input,
                labels=labels,
                use_cross_attention=True
            )
            
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            # Get predicted bytes
            predicted_bytes = predictions[0].cpu().tolist()
            print(f"Predicted bytes: {predicted_bytes[:20]}...")
            
            # Compare
            original_text_bytes = input_ids[1:]  # Remove BOS
            accuracy = (predictions == labels).float().mean().item()
            print(f"Token accuracy: {accuracy:.1%}")
            
            # Try to decode
            print("\nDecoding attempts:")
            
            # 1. Direct decode
            try:
                # Filter special tokens
                valid_bytes = [b for b in predicted_bytes if 0 <= b < 256]
                if valid_bytes:
                    decoded = bytes(valid_bytes).decode('utf-8', errors='replace')
                    print(f"1. Direct decode: '{decoded[:50]}'")
                else:
                    print(f"1. No valid bytes in predictions")
            except Exception as e:
                print(f"1. Direct decode failed: {e}")
            
            # 2. Check what tokens are being predicted
            print(f"\n2. Token distribution:")
            unique_tokens = set(predicted_bytes)
            print(f"   Unique tokens predicted: {len(unique_tokens)}")
            print(f"   Token range: {min(predicted_bytes)} - {max(predicted_bytes)}")
            print(f"   Special tokens (>255): {[t for t in unique_tokens if t > 255]}")
            
            # 3. Check if model is predicting mostly padding or special tokens
            special_count = sum(1 for t in predicted_bytes if t >= 256)
            print(f"   Special token ratio: {special_count}/{len(predicted_bytes)} = {special_count/len(predicted_bytes):.1%}")