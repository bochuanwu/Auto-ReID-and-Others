from collections import namedtuple
import torch
import torch.nn as nn

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


BASELINE = Genotype(
  normal=[
    ('dil_conv_3x3', 1, 0.45), 
    ('dil_conv_3x3', 0, 0.24), 
    ('dil_conv_3x3', 1, 0.68), 
    ('sep_conv_3x3', 2, 0.66), 
    ('dil_conv_3x3', 2, 0.93), 
    ('sep_conv_3x3', 1, 0.66), 
    ('sep_conv_3x3', 2, 0.60), 
    ('sep_conv_3x3', 3, 0.61)
  ], 
  normal_concat=range(2, 6), 
  reduce=[
    ('sep_conv_3x3', 1, 0.46), 
    ('skip_connect', 0, 0.02), 
    ('max_pool_3x3', 2, 0.54), 
    ('dil_conv_3x3', 1, 0.52), 
    ('max_pool_3x3', 0, 0.40), 
    ('dil_conv_3x3', 2, 0.40), 
    ('skip_connect', 0, 0.88), 
    ('dil_conv_3x3', 4, 0.76)
  ], 
  reduce_concat=range(2, 6)
)

BASELINE_REID_SEARCH_SPACE = Genotype(
  normal=[
    ('dil_conv_3x3', 0, 0.29), 
    ('skip_connect', 1, 0.00), 
    ('max_pool_3x3', 2, 0.30), 
    ('sep_conv_3x3', 0, 0.13), 
    ('max_pool_3x3', 3, 0.22), 
    ('sep_conv_3x3', 1, 0.01), 
    ('sep_conv_3x3', 2, 0.98), 
    ('dil_conv_3x3', 3, 0.95)
  ], 
  normal_concat=range(2, 6), 
  reduce=[
    ('skip_connect', 1, 0.60), 
    ('part_aware',   0, 0.43), 
    ('avg_pool_3x3', 2, 0.44), 
    ('part_aware',   0, 0.43), 
    ('sep_conv_3x3', 1, 0.72), 
    ('avg_pool_3x3', 2, 0.47), 
    ('skip_connect', 4, 0.62), 
    ('max_pool_3x3', 3, 0.43)
  ], 
  reduce_concat=range(2, 6)
)


RETRIEVAL_REID_SEARCH_SPACE = Genotype(
  normal=[
    ('sep_conv_3x3', 0, 0.41), 
    ('skip_connect', 1, 0.00), 
    ('dil_conv_3x3', 2, 0.46), 
    ('dil_conv_3x3', 0, 0.03), 
    ('dil_conv_3x3', 3, 0.41), 
    ('sep_conv_3x3', 2, 0.21), 
    ('dil_conv_3x3', 4, 0.55), 
    ('dil_conv_3x3', 3, 0.18)
  ], 
  normal_concat=range(2, 6), 
  reduce=[
    ('max_pool_3x3', 0, 0.43), 
    ('part_aware',   1, 0.42), 
    ('sep_conv_3x3', 0, 0.83), 
    ('dil_conv_3x3', 2, 0.60), 
    ('dil_conv_3x3', 2, 0.46), 
    ('max_pool_3x3', 0, 0.40), 
    ('dil_conv_3x3', 4, 0.76), 
    ('dil_conv_3x3', 3, 0.63)
  ], 
  reduce_concat=range(2, 6)
)
