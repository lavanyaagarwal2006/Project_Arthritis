"""
Helper script to extract class names from your training data

Run this script to update the CLASS_LABELS in model_utils.py with actual
class names from your _classes.csv file.

Usage:
    python extract_class_names.py /path/to/multiclass_zip/train/_classes.csv
"""

import pandas as pd
import sys
import os

def extract_class_names(csv_path):
    """Extract class names from _classes.csv file"""
    try:
        df = pd.read_csv(csv_path)

        # Get all column names except 'filename'
        all_columns = [col for col in df.columns if col != 'filename']

        # Remove 'soft tissue calcination' as it was dropped in the final model
        class_columns = [col for col in all_columns if col.strip() != 'soft tissue calcination']

        print("=" * 60)
        print("Extracted Class Names (11 classes after dropping 'soft tissue calcination'):")
        print("=" * 60)

        for idx, class_name in enumerate(class_columns):
            print(f"{idx}: {class_name.strip()}")

        print("\n" + "=" * 60)
        print("Python list format (copy this to model_utils.py):")
        print("=" * 60)
        print("\nCLASS_LABELS = [")
        for class_name in class_columns:
            print(f'    "{class_name.strip()}",')
        print("]")
        print("\n" + "=" * 60)

        # Also save to a file
        with open('class_labels.txt', 'w') as f:
            f.write("CLASS_LABELS = [\n")
            for class_name in class_columns:
                f.write(f'    "{class_name.strip()}",\n')
            f.write("]\n")

        print("\nClass labels have been saved to 'class_labels.txt'")
        print("Copy the content and replace CLASS_LABELS in model_utils.py")

    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        print("\nUsage: python extract_class_names.py /path/to/_classes.csv")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_class_names.py /path/to/_classes.csv")
        print("\nExample:")
        print("  python extract_class_names.py C:/Users/Lavanya/Desktop/CreatED/multiclass_zip/train/_classes.csv")
    else:
        csv_path = sys.argv[1]
        extract_class_names(csv_path)
