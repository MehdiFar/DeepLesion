import os
import sys

cur_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(cur_dir, '..'))
sys.path.append(os.path.join(parent_dir,'PreProcessing'))