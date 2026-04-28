import pandas as pd  # type: ignore
import os


def load_data(config: dict[str, str]) -> pd.DataFrame:
    """
    Loads the data based on the provided config.
    Returns a DataFrame with the data in the following columns and types:
    - case_id: str
    - resource: str
    - activity_name: str
    - start_timestamp: pd.Timestamp
    - end_timestamp: pd.Timestamp
    """
    data_dir = config["input_filename"]
    df = pd.read_csv(data_dir)

    # Sort the data by case_id and start_timestamp
    df.sort_values(
        by=[config["case_id_col"], config["start_timestamp_col"]],
        inplace=True,
        ignore_index=True,
    )

    # Create new dataframe with only the necessary columns and standardized names
    processed_df = pd.DataFrame(
        columns=[
            "case_id",
            "resource",
            "activity_name",
            "start_timestamp",
            "end_timestamp",
        ]
    )
    processed_df["case_id"] = df[config["case_id_col"]]
    processed_df["resource"] = df[config["resource_id_col"]]
    processed_df["activity_name"] = df[config["activity_col"]]

    # Convert to datetime format with flexible parsing for different formats
    # Handle both formats: 2023-02-09T08:00:00.000 and 2019-03-25 08:00:00+00:00
    processed_df["start_timestamp"] = pd.to_datetime(
        df[config["start_timestamp_col"]],
        format="mixed",
        utc=False,  # Don't assume UTC to handle both timezone-aware and naive properly
    )
    processed_df["end_timestamp"] = pd.to_datetime(
        df[config["end_timestamp_col"]], format="mixed", utc=False
    )

    # Handle timezone conversion properly
    # If timestamps are timezone-naive, localize them to UTC first
    # If they're already timezone-aware, convert to UTC
    if processed_df["start_timestamp"].dt.tz is None:
        # All timestamps are timezone-naive - localize to UTC
        processed_df["start_timestamp"] = processed_df[
            "start_timestamp"
        ].dt.tz_localize("UTC")
        processed_df["end_timestamp"] = processed_df["end_timestamp"].dt.tz_localize(
            "UTC"
        )
    else:
        # All timestamps are timezone-aware - convert to UTC
        processed_df["start_timestamp"] = processed_df["start_timestamp"].dt.tz_convert(
            "UTC"
        )
        processed_df["end_timestamp"] = processed_df["end_timestamp"].dt.tz_convert(
            "UTC"
        )

    return processed_df


def split_data(df: pd.DataFrame, split: float = 0.8):
    # Split data into training and testing sets, taking into account the case_id and splitting such that no day is split between the sets
    # Get unique case_ids
    case_ids = df["case_id"].unique()
    case_dates = df.groupby("case_id")["start_timestamp"].min().dt.date

    # Shuffle case_ids

    # Split case_ids into training and testing sets
    train_case_ids = set()
    test_case_ids = set()

    # Date split, so that no day or case is split between the sets
    case_dates = sorted(case_dates)
    date_split_index = int(len(case_dates) * split)

    for i in range(len(case_ids)):
        if case_dates[i] in case_dates[:date_split_index]:
            train_case_ids.add(case_ids[i])
        else:
            test_case_ids.add(case_ids[i])

    # Create training and testing sets
    train_df = df[df["case_id"].isin(train_case_ids)]
    test_df = df[df["case_id"].isin(test_case_ids)]
    print(f"Train set size: {len(train_df)}, Test set size: {len(test_df)}")
    print(
        f"Proportion of train set: {len(train_df) / (len(test_df) + len(train_df)):.2f}"
    )
    return train_df, test_df


def load_and_preprocess_data(
    data_path: str,
    train_split: float = 0.8,
    min_case_length: int = 3,
    collaboration_config: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess event log data for training.

    Args:
        data_path: Path to CSV file
        train_split: Proportion of data for training (default 0.8)
        min_case_length: Minimum number of activities per case (default 3)
        collaboration_config: Optional collaboration config dict with 'tasks' list

    Returns:
        Tuple of (train_data, test_data) as pandas DataFrames
    """
    from .preprocessing import remove_short_cases

    print(f"Loading data from: {data_path}")

    # Load the CSV file directly with standardized column names
    df = pd.read_csv(data_path)

    # Detect column names (flexible to different naming conventions)
    col_mapping = {}

    # Try to find case_id column
    case_cols = ['case_id', 'case', 'caseid', 'Case', 'CaseID']
    for col in case_cols:
        if col in df.columns:
            col_mapping['case_id'] = col
            break

    # Try to find resource column
    resource_cols = ['resource', 'Resource', 'user', 'User', 'agent', 'Agent']
    for col in resource_cols:
        if col in df.columns:
            col_mapping['resource'] = col
            break

    # Try to find activity column
    activity_cols = ['activity_name', 'activity', 'Activity', 'task', 'Task']
    for col in activity_cols:
        if col in df.columns:
            col_mapping['activity_name'] = col
            break

    # Try to find timestamp columns
    start_cols = ['start_timestamp', 'start_time', 'start', 'Start', 'timestamp', 'Timestamp']
    for col in start_cols:
        if col in df.columns:
            col_mapping['start_timestamp'] = col
            break

    end_cols = ['end_timestamp', 'end_time', 'end', 'End', 'complete', 'Complete']
    for col in end_cols:
        if col in df.columns:
            col_mapping['end_timestamp'] = col
            break

    # Try to find assign_timestamp column (optional — used as case arrival time)
    assign_cols = ['assign_timestamp', 'assign', 'Assign']
    for col in assign_cols:
        if col in df.columns:
            col_mapping['assign_timestamp'] = col
            break

    # Rename columns to standardized names
    df_renamed = df.rename(columns={v: k for k, v in col_mapping.items()})

    # Ensure we have all required columns
    required_cols = ['case_id', 'resource', 'activity_name', 'start_timestamp']
    missing_cols = [col for col in required_cols if col not in df_renamed.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please ensure your CSV has columns for: case_id, resource, activity_name, start_timestamp"
        )

    # Convert timestamps
    df_renamed['start_timestamp'] = pd.to_datetime(
        df_renamed['start_timestamp'],
        format='mixed',
        utc=False
    )

    if 'end_timestamp' in df_renamed.columns:
        df_renamed['end_timestamp'] = pd.to_datetime(
            df_renamed['end_timestamp'],
            format='mixed',
            utc=False
        )
    else:
        # If no end_timestamp, create a dummy one (start + 1 hour)
        df_renamed['end_timestamp'] = df_renamed['start_timestamp'] + pd.Timedelta(hours=1)

    # Parse assign_timestamp if present
    if 'assign_timestamp' in df_renamed.columns:
        df_renamed['assign_timestamp'] = pd.to_datetime(
            df_renamed['assign_timestamp'],
            format='mixed',
            utc=False
        )

    # Handle timezones
    if df_renamed['start_timestamp'].dt.tz is None:
        df_renamed['start_timestamp'] = df_renamed['start_timestamp'].dt.tz_localize('UTC')
        df_renamed['end_timestamp'] = df_renamed['end_timestamp'].dt.tz_localize('UTC')
        if 'assign_timestamp' in df_renamed.columns:
            df_renamed['assign_timestamp'] = df_renamed['assign_timestamp'].dt.tz_localize('UTC')
    else:
        df_renamed['start_timestamp'] = df_renamed['start_timestamp'].dt.tz_convert('UTC')
        df_renamed['end_timestamp'] = df_renamed['end_timestamp'].dt.tz_convert('UTC')
        if 'assign_timestamp' in df_renamed.columns:
            df_renamed['assign_timestamp'] = df_renamed['assign_timestamp'].dt.tz_convert('UTC')

    # Sort by case_id and timestamp
    df_renamed = df_renamed.sort_values(
        by=['case_id', 'start_timestamp'],
        ignore_index=True
    )

    print(f"Loaded {len(df_renamed)} events from {df_renamed['case_id'].nunique()} cases")

    # Remove short cases
    df_cleaned = remove_short_cases(df_renamed, min_length=min_case_length)
    print(f"After removing cases with < {min_case_length} activities: {len(df_cleaned)} events")

    # Apply collaboration augmentation if configured
    if collaboration_config and collaboration_config.get('enabled', False):
        df_cleaned = _augment_collaboration(df_cleaned, collaboration_config)
    else:
        df_cleaned['required_roles'] = ''

    # Split into train and test
    train_data, test_data = split_data(df_cleaned, split=train_split)

    return train_data, test_data


def _augment_collaboration(
    df: pd.DataFrame,
    collaboration_config: dict,
) -> pd.DataFrame:
    """Add collaboration metadata columns to the DataFrame.

    For each task rule in the config, randomly mark a fraction of occurrences
    as collaborative by setting agents_required and required_roles.
    """
    import numpy as np

    df = df.copy()
    df['required_roles'] = ''

    tasks_rules = collaboration_config.get('tasks', [])
    if not tasks_rules:
        return df

    seed = collaboration_config.get('seed', 42)
    rng = np.random.RandomState(seed)

    total_collab = 0
    for rule in tasks_rules:
        activity = rule['activity']
        probability = rule.get('probability', 0.0)
        required_roles = rule.get('required_roles', [])
        n_agents = len(required_roles)

        mask = df['activity_name'] == activity
        n_events = mask.sum()
        if n_events == 0:
            print(f"  Warning: activity '{activity}' not found in data")
            continue

        n_collab = int(n_events * probability)
        if n_collab == 0:
            continue

        indices = df[mask].index.tolist()
        collab_indices = rng.choice(indices, size=n_collab, replace=False)

        df.loc[collab_indices, 'required_roles'] = ','.join(required_roles)

        total_collab += n_collab
        print(f"  Collaboration: {activity}: {n_collab}/{n_events} "
              f"({probability*100:.0f}%) require {n_agents} agents {required_roles}")

    print(f"  Total collaborative events: {total_collab}/{len(df)} "
          f"({total_collab/len(df)*100:.1f}%)")

    return df
