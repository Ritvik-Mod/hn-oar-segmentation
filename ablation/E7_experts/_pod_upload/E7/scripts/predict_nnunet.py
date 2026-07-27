"""
E7 Part 1 — nnU-Net clean-208 inference on the single-side test cases (or any
imagesTs folder) via the Python API, using the standalone trained-model folder
(no nnUNet_results env needed).

Golden rule (master 15): mirroring OFF (== --disable_tta). checkpoint_final.pth
(the checkpoint the 0.8187 baseline used).
"""
import os, sys, argparse, torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL = os.path.join(ROOT, "Parotid-Project", "Segmentation_Complete",
                     "Parotid_Segmentation_Complete", "03_models", "nnunet",
                     "trained_model_v2_noMirror_FINAL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        dev = torch.device(args.device)
        # quick smoke test that the device actually works
        _ = (torch.zeros(1, device=dev) + 1).cpu()
    except Exception as e:
        print(f"device {args.device} unavailable ({e}); falling back to cpu")
        dev = torch.device("cpu")
    print(f"nnU-Net predict on {dev}; model={args.model}")

    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True,
                           use_mirroring=False,   # == --disable_tta
                           perform_everything_on_device=(dev.type != "cpu"),
                           device=dev, verbose=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(args.model, use_folds=(0,),
                                              checkpoint_name=args.checkpoint)
    pred.predict_from_files(args.in_dir, args.out_dir, save_probabilities=False,
                            overwrite=True, num_processes_preprocessing=4,
                            num_processes_segmentation_export=4)
    print(f"wrote nnU-Net predictions -> {args.out_dir}")


if __name__ == "__main__":
    main()
