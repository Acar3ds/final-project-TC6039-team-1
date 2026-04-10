import json
import sys
import pathlib
from datetime import datetime, timedelta, timezone

import pytest

# Make sure repo root is on sys.path so we can import src.data_loader
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data_loader import load_data, clean_data, eda_summary, MissingColumnError


def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def test_normal_case(tmp_path):
    """
    Normal Case: Standard Weekday Extraction
    - Create 7 raw files (Monday through Sunday) with a single event each.
    - Expectation: after cleaning with drop_weekends=True, we get exactly 5 records
      corresponding to Monday-Friday and no timestamps are lost at day transitions.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Choose a known Monday date: 2026-04-06 (Monday) and create 7 consecutive days
    start = datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc)  # midday UTC
    expected_dates = []
    for i in range(7):
        dt = start + timedelta(days=i)
        expected_dates.append(dt.date().isoformat())
        epoch = int(dt.timestamp())
        _write_json(raw_dir / f"event_{i}.json", {"frame_timestamp": epoch})

    df = load_data(str(raw_dir))
    # we should have 7 records
    assert len(df) == 7

    cleaned = clean_data(df, output_file=None, drop_weekends=True)
    # only weekdays remain
    assert len(cleaned) == 5
    unique_dates = sorted(cleaned['date'].unique().tolist())
    # Expect the first five dates (Monday-Friday)
    assert unique_dates == expected_dates[:5]


def test_edge_case(tmp_path):
    """
    Edge Case: corrupted files and missing values + gap-filling (TC6039.1)
    - Create a few files: one valid, one corrupted (invalid JSON), one with null timestamp.
    - Use fill_gaps=True and gap frequency of 15 minutes. Expect that the filled
      DataFrame contains 15-minute intervals and zero volumes for missing slots.
    """
    raw_dir = tmp_path / "raw2"
    raw_dir.mkdir()

    # Valid event at 00:00 UTC and another at 01:00 UTC on the same day
    day = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)
    _write_json(raw_dir / "good1.json", {"frame_timestamp": int(day.timestamp())})
    _write_json(raw_dir / "good2.json", {"frame_timestamp": int((day + timedelta(hours=1)).timestamp())})

    # Corrupted file (write invalid JSON)
    with open(raw_dir / "corrupt.json", 'w') as f:
        f.write('{ this is not valid json')

    # File with null timestamp
    _write_json(raw_dir / "null.json", {"frame_timestamp": None})

    df = load_data(str(raw_dir))
    # load_data should have parsed the two valid files (corrupt should be skipped)
    assert len(df) >= 2

    # Now test fill_gaps behavior (disable drop_weekends to preserve all data for gap-filling)
    filled = clean_data(df, output_file=None, fill_gaps=True, gap_freq='15min', drop_weekends=False)
    # The period from 00:00 to 01:00 with 15T should have 5 points
    # Confirm continuous datetimes and that the sum(volume) equals the number of original events
    assert 'volume' in filled.columns
    # ensure datetime column exists and is sorted and regular
    assert filled['datetime'].is_monotonic_increasing
    diffs = filled['datetime'].diff().dropna().unique()
    # diffs are timedeltas convertible to seconds; expect all diffs equal 15 minutes (900 seconds)
    assert all(td.total_seconds() == 15 * 60 for td in diffs)
    # total volume equals number of parsed events (2)
    assert int(filled['volume'].sum()) == 2

    # Finally, verify MissingColumnError is raised when required columns are missing
    small_df = df.drop(columns=['date'])
    with pytest.raises(MissingColumnError):
        clean_data(small_df, output_file=None)

