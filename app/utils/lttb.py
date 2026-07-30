import math
import numpy as np

def downsample(data, n_out):
    """
    Pure Python implementation of Largest Triangle Three Buckets (LTTB)
    data: list of [x, y] coordinates
    n_out: number of output points
    """
    if n_out >= len(data) or n_out == 0:
        return data

    sampled = []
    
    # Bucket size. Leave room for start and end data points
    every = (len(data) - 2) / (n_out - 2)
    
    a = 0  # Initially a is the first point in the triangle
    sampled.append(data[a])
    
    for i in range(0, n_out - 2):
        # Calculate point average for next bucket (containing c)
        avg_x = 0
        avg_y = 0
        avg_range_start = int(math.floor((i + 1) * every) + 1)
        avg_range_end = int(math.floor((i + 2) * every) + 1)
        
        if avg_range_end > len(data):
            avg_range_end = len(data)
            
        avg_range_length = avg_range_end - avg_range_start
        
        while avg_range_start < avg_range_end:
            avg_x += data[avg_range_start][0]
            avg_y += data[avg_range_start][1]
            avg_range_start += 1
            
        avg_x /= avg_range_length
        avg_y /= avg_range_length
        
        # Get the range for this bucket
        range_offs = int(math.floor((i + 0) * every) + 1)
        range_to = int(math.floor((i + 1) * every) + 1)
        
        # Point a
        point_a_x = data[a][0]
        point_a_y = data[a][1]
        
        max_area = -1
        max_area_point = -1
        
        while range_offs < range_to:
            # Calculate triangle area over three buckets
            area = math.fabs(
                (point_a_x - avg_x) * (data[range_offs][1] - point_a_y) -
                (point_a_x - data[range_offs][0]) * (avg_y - point_a_y)
            ) * 0.5
            
            if area > max_area:
                max_area = area
                max_area_point = data[range_offs]
                next_a = range_offs
                
            range_offs += 1
            
        sampled.append(max_area_point)
        a = next_a
        
    sampled.append(data[-1])
    return sampled
