#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 06:11:38 2025

@author: christian
"""

from compare_annotations import match_annotations

p_flavia = '/home/christian/Downloads/labels_annotation_comparison_flavia_0g_g1_t3_2025-10-16-07-01-21.json'
p_matteo = '/home/christian/Downloads/C_0G_G1_T3_290825rep1_20251007_084954_4.json'
p_kentaro = '/home/christian/Downloads/labels_my-project-name_2025-10-22-03-14-20.json'

match_annotations(p_flavia,p_matteo,cutoff=200)

match_annotations(p_flavia,p_kentaro,cutoff=200)
