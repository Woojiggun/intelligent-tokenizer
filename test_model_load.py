#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test if model loads correctly
"""

import torch
import sys
from pathlib import Path

# Test 1: Load local model
sys.path.append(str(Path(__file__).parent.parent / "intelligent-tokenizer_v6.0"))
from src.core.byte_tokenizer_v6 import ByteTokenizerV6
from core.boundary_aware_model import BoundaryAwareTokenizerModel

checkpoint_path = Path("../intelligent-tokenizer_v6.0/checkpoints/unified/latest_checkpoint.pt")
print(f"Loading LOCAL model...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

model = BoundaryAwareTokenizerModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Test with English
text = "The quick"
tokenizer = ByteTokenizerV6()
encoded = tokenizer.encode(text)
input_ids = torch.tensor([encoded['input_ids']])

with torch.no_grad():
    decoder_input = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        decoder_input_ids=decoder_input,
        labels=labels,
        use_cross_attention=True
    )
    
    predictions = torch.argmax(outputs['logits'], dim=-1)
    pred_bytes = predictions[0].tolist()
    
print(f"Text: {text}")
print(f"Original bytes: {encoded['input_ids']}")
print(f"Predicted bytes: {pred_bytes}")
print(f"Match: {pred_bytes == list(labels[0].tolist())}")

# Test 2: Check uploaded model file
print("\n" + "="*60)
print("Checking uploaded model...")
uploaded_model = Path("pytorch_model.bin")
if uploaded_model.exists():
    state_dict = torch.load(uploaded_model, map_location='cpu')
    print(f"Uploaded model keys: {list(state_dict.keys())[:5]}...")
    
    # Try to load it
    model2 = BoundaryAwareTokenizerModel(**checkpoint['model_config'])
    model2.load_state_dict(state_dict)
    model2.eval()
    
    # Test again
    with torch.no_grad():
        outputs2 = model2(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            decoder_input_ids=decoder_input,
            labels=labels,
            use_cross_attention=True
        )
        
        predictions2 = torch.argmax(outputs2['logits'], dim=-1)
        pred_bytes2 = predictions2[0].tolist()
    
    print(f"Uploaded model predictions: {pred_bytes2}")
    print(f"Match with local: {pred_bytes2 == pred_bytes}")
else:
    print("No pytorch_model.bin found locally")