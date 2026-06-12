import os
import sys

# Reconfigure stdout to use utf-8 if possible
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

log_path = r"C:\Users\rumeysagokce\.gemini\antigravity\brain\6c8687fb-fb12-4d02-9615-eb2e20537a3c\.system_generated\tasks\task-14667.log"
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    print(f"Total lines in log: {len(lines)}")
    found = False
    for i, line in enumerate(lines):
        if "ulasli" in line.lower() or "ulaşlı" in line.lower():
            found = True
            # print context around the match
            start = max(0, i - 10)
            end = min(len(lines), i + 25)
            print(f"--- Context for match at line {i+1} ---")
            for j in range(start, end):
                safe_line = lines[j].strip().encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
                print(f"{j+1}: {safe_line}")
    if not found:
        print("No matches found for ulasli/ulaşlı in log yet.")
else:
    print(f"Log path does not exist: {log_path}")
