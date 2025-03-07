"""
Script: hotpot-to-tiny.py
Description: Downloads and converts the Hotpot QA to a simple a tiny format.

---

We only need a few hundred samples at most. Any more than that is overkill.
Ideally, we just grab 100 samples random. The output file is formatted for
input-target pairs where the question becomes the input and the answer becomes
the target.
"""

import json
import os

import requests

source = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json"
destination = "data/tiny.json"
