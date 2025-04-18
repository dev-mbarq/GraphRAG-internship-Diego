import pandas as pd
import numpy as np


def standardize_article_number(number):
    """
    Standardizes article number codes to 8XXX.XX.XXX format
    Input can be:
        - NaN -> 80999.00.000
        - XXX/XXXX (e.g., 824 -> 8024)
        - XXXX (e.g., 8007)
        - XXXX.XX (e.g., 8052.03)
        - XXXX.XXX (e.g., 8008.003)
        - XXXX.XX.XXX (e.g., 8025.04.001)
    Output: 8XXX.XX.XXX
    """
    try:
        # Handle NaN cases
        if pd.isna(number):
            return "8999.00.000"

        # Convert to string if it's not
        number = str(number)

        # Split by dots
        parts = number.split(".")

        # Helper function to fix base number
        def fix_base_number(base_str):
            # Specific case for '824'
            if base_str == "824":
                return "8024"
            # For all other cases, ensure it starts with 8 and has 4 digits
            return base_str.zfill(4)

        if len(parts) == 1:
            # Case XXXX -> 8XXX.00.000
            base = fix_base_number(parts[0])
            return f"{base}.00.000"

        elif len(parts) == 2:
            if len(parts[1]) == 2:
                # Case XXXX.XX -> 8XXX.XX.000
                base = fix_base_number(parts[0])
                suffix = parts[1].zfill(2)
                return f"{base}.{suffix}.000"
            elif len(parts[1]) == 3:
                # Case XXXX.XXX -> 8XXX.00.XXX
                base = fix_base_number(parts[0])
                subdivision = parts[1].zfill(3)
                return f"{base}.00.{subdivision}"

        elif len(parts) == 3:
            # Case XXXX.XX.XXX -> 8XXX.XX.XXX
            base = fix_base_number(parts[0])
            suffix = parts[1].zfill(2)
            subdivision = parts[2].zfill(3)
            return f"{base}.{suffix}.{subdivision}"

        return "0000.00.000"  # Default value for invalid cases

    except Exception as e:
        print(f"Error processing number: {number}")
        return "0000.00.000"


def std_to_sort_tuple(std_number):
    """
    Converts a number in 8XXX.XX.XXX format to a tuple (base, suffix, subdivision)
    for sorting purposes
    """
    try:
        # Handle special cases
        if pd.isna(std_number) or std_number == "0nan.00.000":
            return (0, 0, 0)

        if std_number == "8999.00.000":  # Special case for original NaN values
            return (999, 0, 0)

        if std_number == "0000.00.000":  # Case for invalid values
            return (0, 0, 0)

        parts = std_number.split(".")
        base = int(parts[0]) - 8000  # 8002.00.000 -> 2
        suffix = int(parts[1])  # 8002.03.000 -> 3
        subdivision = int(parts[2])  # 8002.00.001 -> 1

        return (base, suffix, subdivision)

    except Exception as e:
        print(f"Error processing number: {std_number}")
        return (0, 0, 0)
