import pandas as pd


def remove_short_cases(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cases with less than 3 activities from the DataFrame.
    """
    # Group by case_id and filter out cases with less than 3 activities
    filtered_data = data.groupby("case_id").filter(lambda x: len(x) >= 3)

    return filtered_data.reset_index(drop=True)
