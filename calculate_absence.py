from datetime import datetime

# Read absence log data from file
with open("absence_log.txt", "r") as file:
    lines = file.readlines()

absence_log = []
for line in lines:
    if "to" in line:
        start, end = line.strip().split(" to ")
        absence_log.append((start, end))

# Convert timestamps to datetime objects
absence_durations = []
for start, end in absence_log:
    start_time = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
    duration = (end_time - start_time).total_seconds()
    
    if duration >= 5:
        absence_durations.append((start, end, duration))

# Display results
print("Absences of 5 seconds or more:")
for entry in absence_durations:
    print(f"Start: {entry[0]}, End: {entry[1]}, Duration: {entry[2]} sec")
