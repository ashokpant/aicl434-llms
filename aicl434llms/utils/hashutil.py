"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 18/05/2025
"""

import hashlib


def generate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
