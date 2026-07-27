import json, random

with open('/Users/ritvikmod/Desktop/hn_patients.json') as f:
    hn_patients = json.load(f)

random.seed(42)
random.shuffle(hn_patients)

n = len(hn_patients)
train_end = int(n * 0.70)
val_end   = int(n * 0.80)

split = {
    'train': hn_patients[:train_end],
    'val':   hn_patients[train_end:val_end],
    'test':  hn_patients[val_end:]
}

print(f"Train : {len(split['train'])} patients")
print(f"Val   : {len(split['val'])} patients")
print(f"Test  : {len(split['test'])} patients")

with open('/Users/ritvikmod/Desktop/dataset_split.json', 'w') as f:
    json.dump(split, f, indent=2)

print("\nSplit saved to dataset_split.json — do not modify this file again.")