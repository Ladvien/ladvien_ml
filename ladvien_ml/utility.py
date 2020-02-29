#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 07:05:13 2019

@author: ladvien
"""

import pandas as pd
from datetime import datetime

class Utility:

    def __init__(self):
        pass

    def make_dir(self, path):
        """
            If the directory doesn't already exist, let's make it.
            https://stackoverflow.com/a/273227
        """
        import os
        if not os.path.exists(path):
            os.makedirs(path)