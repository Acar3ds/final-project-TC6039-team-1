import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

def load_data(raw_data_dir="data/raw/", output_file="data/processed/df.csv"):
    """
    Read JSON files from raw_data_dir, convert Unix epoch timestamps to
    Monterrey local time, and write date/time plus day type columns.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monterrey_tz = ZoneInfo("America/Mexico_City")
    
    # Collect all data
    all_data = []
    
    for json_file in Path(raw_data_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Handle both single dict and list of dicts
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    if "frame_timestamp" in item:
                        timestamp_value = item["frame_timestamp"]

                        # Accept Unix epoch in either seconds or milliseconds.
                        epoch = float(timestamp_value)
                        if epoch > 1e12:
                            epoch /= 1000.0

                        dt_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
                        dt_local = dt_utc.astimezone(monterrey_tz)
                        day_type = "weekend" if dt_local.weekday() >= 5 else "weekday"
                        weekday_number = dt_local.weekday()  # Monday=0, Sunday=6
                        
                        all_data.append({
                            'counter': len(all_data) + 1,
                            'date': dt_local.date().isoformat(),
                            'hour': dt_local.hour,
                            'minute': dt_local.minute,
                            'day_type': day_type,
                            'weekday_number': weekday_number

                        })
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Write to CSV
    if all_data:
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['counter', 'date', 'hour', 'minute', 'day_type', 'weekday_number']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        print(f"Data written to {output_path}")
    else:
        print("No data found to write")

if __name__ == "__main__":
    load_data()