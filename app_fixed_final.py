#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent Tokenizer v6.0 - Embedding Communication Layer for LLMs (Fixed)
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from huggingface_hub import hf_hub_download

# ByteTokenizer class
class ByteTokenizerV6:
    def __init__(self, max_seq_len=256):
        self.vocab_size = 260
        self.max_seq_len = max_seq_len
        self.PAD = 256
        self.BOS = 257
        self.EOS = 258  
        self.MASK = 259
    