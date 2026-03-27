import numpy as np
from pathlib import Path
import os
import glob
import numpy as np

def save_scene(points, output_dir):
    """
    points: np.ndarray of shape (N, 7)
            columns: x, y, z, r, g, b, label
    output_dir: Path to scene folder, e.g. data/my_dataset/train/scene_001
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coord   = points[:, 0:3].astype(np.float32)
    color   = points[:, 3:6].astype(np.float32)  # keep in 0-255 range; NormalizeColor handles /127.5-1
    segment = points[:, 6].astype(np.int32)

    np.save(output_dir / "coord.npy",   coord)
    np.save(output_dir / "color.npy",   color)
    np.save(output_dir / "segment.npy", segment)

if __name__ == "__main__":

    data_root = Path("/data")
    output_dir = Path("/data/my_dataset")

    classes = []
    scenes = os.listdir(data_root)
    for scene in scenes: 
        print(f"Processing {scene}")
        anno_dir = os.path.join(data_root, scene, "_PointOut")
        anno = os.listdir(anno_dir)
        print(f"Found {len(anno)} files in {anno_dir}")
        points_list = []
        for a in glob.glob(os.path.join(anno_dir, "*.xyz")):
            if os.path.basename(a) == "_All.xyz":
                continue
            cls = a.rsplit("_", 1)[0]
            if cls not in classes:
                classes.append(cls)
            
            points = np.loadtxt(a, delimiter=",")
            if points.ndim == 1:
                points = points.reshape(1, -1)
            labels = np.ones((points.shape[0], 1)) * classes.index(cls)

            points_list.append(np.concatenate((points, labels), axis=1))
        
        data = np.concatenate(points_list, axis=0)
        xyz_min = np.amin(data[:, 0:3], axis=0)
        data[:, 0:3] -= xyz_min

        out_file = os.path.join(output_dir, scene+".npy")
        np.save(out_file, data)
        print(f"Saved as {out_file}")
            
    class_file = os.path.join(output_dir, "classes.txt")
    with open(class_file, "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")