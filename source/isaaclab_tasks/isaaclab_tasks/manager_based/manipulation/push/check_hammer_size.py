#!/usr/bin/env python3
"""
Script to check hammer mesh dimensions and compare with different scales
"""

import numpy as np
import argparse
import os

def get_estimated_dimensions():
    """
    Estimated dimensions from the hammer USD file based on typical hammer sizes
    These are approximate values - you should measure your real hammer for accuracy
    """
    # These are estimated dimensions in meters for the base hammer mesh
    # You'll need to measure your actual hammer to get precise values
    return {
        'dimensions': np.array([0.32, 0.05, 0.18]),  # Estimated: 32cm x 5cm x 18cm
        'note': 'These are estimated dimensions. Please measure your real hammer for accuracy.'
    }

def analyze_hammer_scales():
    """Analyze hammer with different scales"""
    hammer_path = "/home/maggiewang/Workspace/sim-to-real-rl/assets/hammer/hammer_mass620g.usda"
    
    print("="*60)
    print("HAMMER MESH ANALYSIS")
    print("="*60)
    
    # Get estimated mesh dimensions
    bbox_data = get_estimated_dimensions()
    print(f"USD File: {hammer_path}")
    print(f"NOTE: {bbox_data['note']}")
    print()
    
    bbox = bbox_data
    print(f"Estimated mesh dimensions (meters):")
    print(f"  X (length): {bbox['dimensions'][0]:.6f} m")
    print(f"  Y (width):  {bbox['dimensions'][1]:.6f} m") 
    print(f"  Z (height): {bbox['dimensions'][2]:.6f} m")
    print()
    
    # Test different scales
    scales_to_test = [0.105, 0.115, 0.100, 0.120]
    
    print("SCALED DIMENSIONS:")
    print("-" * 60)
    print(f"{'Scale':<8} {'Length(cm)':<12} {'Width(cm)':<12} {'Height(cm)':<12}")
    print("-" * 60)
    
    for scale in scales_to_test:
        scaled_dims = bbox['dimensions'] * scale * 100  # Convert to cm
        print(f"{scale:<8.3f} {scaled_dims[0]:<12.2f} {scaled_dims[1]:<12.2f} {scaled_dims[2]:<12.2f}")
    
    print()
    print("COMPARISON WITH REAL HAMMER:")
    print("-" * 60)
    print("Please measure your real hammer and compare with the scaled dimensions above.")
    print("Typical hammer dimensions:")
    print("  - Length: 25-35 cm")
    print("  - Width: 3-5 cm")  
    print("  - Height: 15-20 cm")
    print()
    
    # Calculate what scale would give specific target dimensions
    print("SCALE CALCULATOR:")
    print("-" * 60)
    print("To find the right scale, measure your real hammer and use:")
    print(f"  Scale = target_length_cm / {bbox['dimensions'][0]*100:.2f}")
    print(f"  Scale = target_width_cm / {bbox['dimensions'][1]*100:.2f}")
    print(f"  Scale = target_height_cm / {bbox['dimensions'][2]*100:.2f}")
    print()
    
    # Show what the old vs new scales translate to
    old_scale = 0.105
    new_scale = 0.115
    
    print("CURRENT CONFIGURATION COMPARISON:")
    print("-" * 60)
    print("Old working config (IsaacLab April):")
    old_dims = bbox['dimensions'] * old_scale * 100
    print(f"  Scale: {old_scale}")
    print(f"  Dimensions: {old_dims[0]:.2f} x {old_dims[1]:.2f} x {old_dims[2]:.2f} cm")
    
    print("Current config (isaaclab-phys2real):")
    new_dims = bbox['dimensions'] * new_scale * 100
    print(f"  Scale: {new_scale}")
    print(f"  Dimensions: {new_dims[0]:.2f} x {new_dims[1]:.2f} x {new_dims[2]:.2f} cm")
    
    print("Difference:")
    diff = new_dims - old_dims
    print(f"  Length: +{diff[0]:.2f} cm")
    print(f"  Width:  +{diff[1]:.2f} cm") 
    print(f"  Height: +{diff[2]:.2f} cm")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hammer mesh dimensions")
    parser.add_argument("--real_length", type=float, help="Real hammer length in cm")
    parser.add_argument("--real_width", type=float, help="Real hammer width in cm") 
    parser.add_argument("--real_height", type=float, help="Real hammer height in cm")
    
    args = parser.parse_args()
    
    if args.real_length or args.real_width or args.real_height:
        # Calculate recommended scale from real measurements
        bbox_data = get_estimated_dimensions()
        dims = bbox_data['dimensions'] * 100  # Convert to cm
        
        print("SCALE CALCULATION FROM REAL MEASUREMENTS:")
        print("-" * 50)
        if args.real_length:
            scale_x = args.real_length / dims[0]
            print(f"Based on length: {args.real_length}cm / {dims[0]:.1f}cm = {scale_x:.3f}")
        if args.real_width:
            scale_y = args.real_width / dims[1] 
            print(f"Based on width: {args.real_width}cm / {dims[1]:.1f}cm = {scale_y:.3f}")
        if args.real_height:
            scale_z = args.real_height / dims[2]
            print(f"Based on height: {args.real_height}cm / {dims[2]:.1f}cm = {scale_z:.3f}")
        print()
    
    # Always run full analysis
    analyze_hammer_scales()