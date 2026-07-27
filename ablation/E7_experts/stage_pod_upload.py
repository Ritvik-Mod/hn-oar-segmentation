"""
Stage every E7 pod input into ONE tree (ablation_study/E7_experts/_pod_upload/)
using HARDLINKS for the big files (instant, 0 bytes, same volume) and copies for
the small scripts. Upload this single tree with the 8-stream recipe; it lands in
the exact /workspace layout pod_run_e7.sh expects. Dataset005/006 are built ON
the pod from Dataset003, so no image is uploaded twice.
"""
import os, glob, shutil, sys
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE = os.path.join(os.path.dirname(__file__), "_pod_upload")
E7SS = os.path.join(ROOT, "ablation_study", "E7_singleside")
E7EX = os.path.join(ROOT, "ablation_study", "E7_experts")
PIPE = os.path.join(ROOT, "pipeline")
E6 = os.path.join(ROOT, "ablation_study", "E6_masked_loss")


def link_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.isfile(f):
            d = os.path.join(dst, os.path.basename(f))
            if os.path.exists(d):
                os.remove(d)
            try:
                os.link(f, d)
            except OSError:
                shutil.copy2(f, d)


def link_tree_recursive(src, dst):
    for r, _, files in os.walk(src):
        rel = os.path.relpath(r, src)
        out = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(out, exist_ok=True)
        for f in files:
            if f.startswith("._") or f == ".DS_Store":
                continue
            s = os.path.join(r, f); d = os.path.join(out, f)
            if os.path.exists(d):
                os.remove(d)
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    # 1. Dataset003 raw (training source for both experts)
    d3 = os.path.join(ROOT, "ablation_study", "E2_annotation_gap", "nnUNet_raw", "Dataset003_ParotidDirty")
    d3out = os.path.join(STAGE, "nnunet", "nnUNet_raw", "Dataset003_ParotidDirty")
    link_tree(os.path.join(d3, "imagesTr"), os.path.join(d3out, "imagesTr"))
    link_tree(os.path.join(d3, "labelsTr"), os.path.join(d3out, "labelsTr"))
    for j in ("dataset.json", "case_mapping.json"):
        os.makedirs(d3out, exist_ok=True)
        try: os.link(os.path.join(d3, j), os.path.join(d3out, j))
        except OSError: shutil.copy2(os.path.join(d3, j), os.path.join(d3out, j))

    # 2. E7 single-side inputs
    link_tree(os.path.join(E7SS, "nnraw", "imagesTs"), os.path.join(STAGE, "E7", "nnraw", "imagesTs"))
    link_tree(os.path.join(E7SS, "nnraw", "labelsTs"), os.path.join(STAGE, "E7", "nnraw", "labelsTs"))
    link_tree(os.path.join(E7SS, "e6npz"), os.path.join(STAGE, "E7", "e6npz"))

    # 3. Dataset002 both-parotid 43 test (for expert eval on the easy set)
    d2 = os.path.join(ROOT, "Parotid-Project", "Datasets", "Dataset002_Parotid")
    link_tree(os.path.join(d2, "imagesTs"), os.path.join(STAGE, "E7", "Dataset002_test", "imagesTs"))
    link_tree(os.path.join(d2, "labelsTs"), os.path.join(STAGE, "E7", "Dataset002_test", "labelsTs"))

    # 4. models for Part-1 inference
    clean = os.path.join(ROOT, "Parotid-Project", "Segmentation_Complete",
                         "Parotid_Segmentation_Complete", "03_models", "nnunet",
                         "trained_model_v2_noMirror_FINAL")
    link_tree_recursive(clean, os.path.join(STAGE, "E7", "models", "nnunet_clean208"))
    mstage = os.path.join(STAGE, "E7", "models")
    os.makedirs(mstage, exist_ok=True)
    for name, p in [("e4.pth", os.path.join(ROOT, "ablation_study", "_pod_results_B", "checkpoints", "E4_custom_3d_unet_best.pth")),
                    ("dirty430.pth", os.path.join(ROOT, "ablation_study", "_pod_results_B", "E6_masked_loss", "ckpt_dirty430", "best_model.pth")),
                    ("masked430.pth", os.path.join(ROOT, "ablation_study", "_pod_results_B", "E6_masked_loss", "ckpt_masked430", "best_model.pth"))]:
        d = os.path.join(mstage, name)
        try: os.link(p, d)
        except OSError: shutil.copy2(p, d)

    # 5. scripts
    sout = os.path.join(STAGE, "E7", "scripts")
    os.makedirs(sout, exist_ok=True)
    scripts = [
        (E7EX, "build_experts_pod.py"), (E7EX, "combine_experts.py"), (E7EX, "pod_run_e7.sh"),
        (E7SS, "predict_nnunet.py"), (E6, "predict_e6.py"), (E6, "unet3d.py"), (E6, "dataloader_e6.py"),
        (PIPE, "eval_testset.py"), (E7SS, "eval_singleside.py"), (E7SS, "make_montage.py"),
        (E7SS, "case_map.json"), (PIPE, "nnUNetTrainer_250epochs_noMirror.py"),
    ]
    for base, f in scripts:
        shutil.copy2(os.path.join(base, f), os.path.join(sout, f))

    # inventory
    nfiles = sum(len(fs) for _, _, fs in os.walk(STAGE))
    print(f"staged {nfiles} files into {STAGE}")
    for sub in ("nnunet/nnUNet_raw/Dataset003_ParotidDirty/imagesTr",
                "E7/nnraw/imagesTs", "E7/e6npz", "E7/Dataset002_test/imagesTs",
                "E7/models", "E7/scripts"):
        p = os.path.join(STAGE, sub)
        n = len(glob.glob(os.path.join(p, "*")))
        print(f"  {sub}: {n}")


if __name__ == "__main__":
    main()
