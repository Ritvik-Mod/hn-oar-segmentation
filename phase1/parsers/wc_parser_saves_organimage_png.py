import os, glob
import numpy as np
import matplotlib.pyplot as plt

patient_dir = '/Users/ritvikmod/Desktop/1~200734/1~CT1'

def is_coord_line(line):
    """Returns True if this line contains coordinate data (has decimal points)"""
    return '.' in line and len(line.strip()) > 3

def is_single_int(line):
    """Returns True if this line is a single plain integer (no decimal)"""
    s = line.strip()
    if not s: return False
    tokens = s.replace(',', ' ').split()
    if len(tokens) != 1: return False
    try:
        int(tokens[0])  # must be exact int, no decimal
        return True
    except ValueError:
        return False

def parse_wc_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    
    z = float(os.path.basename(filepath).split('T.')[1].replace('.WC', ''))
    
    i = 7
    n_contours = int(lines[i].strip())
    i += 1
    
    contours = {}
    
    for _ in range(n_contours):
        if i >= len(lines): break
        
        # Skip the metadata integer before coordinates
        while i < len(lines) and not is_single_int(lines[i]):
            i += 1
        if i >= len(lines): break
        i += 1  # skip it
        
        # Read all coordinate lines
        coords = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if is_coord_line(line):
                vals = [float(v) for v in line.replace(',', ' ').split() if v]
                coords.extend(vals)
                i += 1
            else:
                break  # hit next integer = organ_id
        
        # Read organ_id
        if i >= len(lines): break
        if not is_single_int(lines[i]): continue
        organ_id = int(lines[i].strip())
        i += 1
        
        if len(coords) < 4: continue
        
        if organ_id not in contours:
            contours[organ_id] = []
        xy_pairs = [(coords[j], coords[j+1]) for j in range(0, len(coords)-1, 2)]
        contours[organ_id].append({'z': z, 'points': xy_pairs})
    
    return contours

# Parse all
all_slices = {}
errors = 0
for wc_file in sorted(glob.glob(os.path.join(patient_dir, 'T.*.WC'))):
    try:
        c = parse_wc_file(wc_file)
        z = float(os.path.basename(wc_file).split('T.')[1].replace('.WC', ''))
        all_slices[z] = c
    except Exception as e:
        errors += 1
        print(f"Error in {os.path.basename(wc_file)}: {e}")

print(f"\nParsed: {len(all_slices)}/113, Errors: {errors}")

organ_count = {}
for z, sc in all_slices.items():
    for oid in sc:
        organ_count[oid] = organ_count.get(oid, 0) + 1

print("\nTop organs by slice count:")
for oid, count in sorted(organ_count.items(), key=lambda x: -x[1])[:15]:
    print(f"  ID={oid}: {count} slices")

if all_slices:
    top_ids = [oid for oid, _ in sorted(organ_count.items(), key=lambda x: -x[1])[:12]]
    mid_z = sorted(all_slices.keys())[len(all_slices)//2]
    slice_data = all_slices[mid_z]

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for idx, oid in enumerate(top_ids):
        ax = axes[idx]
        if oid in slice_data:
            for contour in slice_data[oid]:
                pts = np.array(contour['points'])
                ax.plot(pts[:,0], pts[:,1], '-', linewidth=2)
        ax.set_title(f"ID={oid} ({organ_count[oid]} slices)")
        ax.axis('equal')
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig('/Users/ritvikmod/Desktop/organ_shapes.png', dpi=100)
    plt.show()
    print(f"\nSaved organ_shapes.png to Desktop (z={mid_z})")