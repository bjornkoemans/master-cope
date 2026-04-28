import pandas as pd


def remove_short_cases(data: pd.DataFrame, min_length: int = 3) -> pd.DataFrame:
    """
    Remove cases with fewer than min_length activities from the DataFrame.
    """
    # Group by case_id and filter out cases with fewer than min_length activities
    filtered_data = data.groupby("case_id").filter(lambda x: len(x) >= min_length)

    return filtered_data.reset_index(drop=True)
