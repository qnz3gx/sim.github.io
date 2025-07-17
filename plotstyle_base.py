import numpy as np

def offset(x_array, trace_index, total_traces, offset_fraction=0.1):
    x_array = np.array(x_array)
    x_sorted = np.sort(np.unique(x_array))
    
    if len(x_sorted) > 1:
        min_spacing = np.min(np.diff(x_sorted))
    else:
        min_spacing = x_sorted[0] * offset_fraction  # fallback
    
    total_offset_range = (total_traces - 1) * min_spacing * offset_fraction
    offset = (trace_index - (total_traces - 1) / 2) * min_spacing * offset_fraction

    return x_array + offset