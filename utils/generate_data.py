import json
from datetime import datetime, timedelta
import random

def generate_json_data(filename="data.json"):
    """
    Generates a JSON file containing an array of objects with date and value fields.
    The dates span the last two months in "YYYY-MM-DD" format, and the values range from 20 to 90.
    """
    today = datetime.now()
    start_date = today - timedelta(days=60)
    data = []
    for i in range(60):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")
        value = random.randint(20, 90)
        data.append({"date": date_str, "value": value})

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    generate_json_data()
