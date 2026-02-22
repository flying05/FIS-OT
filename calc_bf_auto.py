import os
import re

# ================= Configuration =================
DATA_ROOT = "/opt/data/private/action_seg_ot/data/Breakfast"
LOG_DIR = "bf_logs"  # Assuming run_bf.sh saves logs here
ACTIONS = [
    "cereals", "coffee", "friedegg", "juice", "milk", 
    "pancake", "salat", "sandwich", "scrambledegg", "tea"
]

METRICS_KEYS = [
    "test_f1_full", "test_f1_per", 
    "test_miou_full", "test_miou_per", 
    "test_mof_full", "test_mof_per"
]
# =======================================

def parse_log_file(log_path):
    """Extract metrics from log file"""
    metrics = {}
    if not os.path.exists(log_path):
        return None
        
    with open(log_path, 'r') as f:
        content = f.read()
        
    for key in METRICS_KEYS:
        pattern = re.compile(rf"{key}\s+([0-9\.]+)")
        match = pattern.search(content)
        if match:
            metrics[key] = float(match.group(1))
        else:
            metrics[key] = 0.0
    return metrics

def get_frame_counts():
    """Count ground truth frames for each activity class"""
    gt_path = os.path.join(DATA_ROOT, "groundTruth")
    counts = {act: 0 for act in ACTIONS}
    
    if not os.path.exists(gt_path):
        print(f"Error: GroundTruth path not found: {gt_path}")
        return None

    print(f"Scanning dataset to count frames: {gt_path} ...")
    
    try:
        all_files = os.listdir(gt_path)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return counts

    # Filter out hidden files
    files = [f for f in all_files if not f.startswith('.')]
    
    matched_count = 0
    for fname in files:
        # Breakfast filename format example: P03_cam01_P03_cereals.txt
        # Activity name is usually after the last underscore, remove .txt
        
        # 1. Remove extension
        base_name = os.path.splitext(fname)[0]
        
        # 2. Extract activity name (take the last part)
        # Note: Some filenames might be weird, use a more robust method:
        # Check if filename ends with an activity name
        found_action = None
        for act in ACTIONS:
            if base_name.endswith(act):
                found_action = act
                break
        
        if found_action:
            file_path = os.path.join(gt_path, fname)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    counts[found_action] += sum(1 for _ in f)
                matched_count += 1

    print(f"--> Successfully matched and counted {matched_count} valid annotation files")
    return counts

def main():
    # 1. Get weights
    frame_counts = get_frame_counts()
    if frame_counts is None: return

    total_frames = sum(frame_counts.values())
    
    if total_frames == 0:
        print("Error: Total frames is 0. Please check the path and filenames.")
        return

    # 2. Collect data and calculate
    weighted_sums = {k: 0.0 for k in METRICS_KEYS}
    
    print("\n" + "="*100)
    header = f"{'Activity':<15} | {'Frames':<8} | {'Weight':<8}"
    for k in METRICS_KEYS:
        header += f" | {k.replace('test_', ''):<10}"
    print(header)
    print("-" * 100)

    for action in ACTIONS:
        log_file = os.path.join(LOG_DIR, f"{action}.log")
        metrics = parse_log_file(log_file)
        
        count = frame_counts[action]
        weight = count / total_frames
        
        row = f"{action:<15} | {count:<8} | {weight:.4f}  "
        
        if metrics:
            for k in METRICS_KEYS:
                val = metrics.get(k, 0.0)
                row += f" | {val:.4f}    "
                weighted_sums[k] += val * weight
        else:
            row += " | (No Log)   " * len(METRICS_KEYS)
            
        print(row)

    print("-" * 100)
    
    final_row = f"{'WEIGHTED AVG':<15} | {total_frames:<8} | {'1.0000':<8}  "
    for k in METRICS_KEYS:
        final_row += f" | {weighted_sums[k]:.4f}    "
    print(final_row)
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
