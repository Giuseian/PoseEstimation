"""
Run SAM3 for every prompt listed in:
  /workspace/shared_data/realsense/objects/<latest_timestamp>/sam3_prompts.json

Masks are saved under:
  /workspace/shared_data/realsense/masks_by_object/<timestamp>/<object_id>/000000.png

Run:
python /workspace/PoseEstimation/pipeline/sam_script_fp.py
"""

import os
import gc
import argparse
import json
import re
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def log(message: str) -> None:
    print(f"[SAM3] {message}")


def slugify_prompt(prompt: str) -> str:
    slug = prompt.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    return slug or "object"


def get_latest_timestamp(root: Path) -> str:
    if not root.exists():
        raise FileNotFoundError(f"Timestamp root does not exist: {root}")

    timestamp_dirs = sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )

    if not timestamp_dirs:
        raise RuntimeError(f"No timestamp folders found in: {root}")

    return timestamp_dirs[-1].name


def resolve_prompts_path(
    data_root: Path,
    timestamp: str | None,
    prompts_path: str | None,
    require_prompts_file: bool,
) -> tuple[Path, str]:
    if prompts_path is not None:
        resolved_path = Path(prompts_path).resolve()
        if timestamp is None:
            timestamp = resolved_path.parent.name
    else:
        objects_root = data_root / "objects"
        if timestamp is None:
            timestamp = get_latest_timestamp(objects_root)
        resolved_path = objects_root / timestamp / "sam3_prompts.json"

    if require_prompts_file and not resolved_path.exists():
        raise FileNotFoundError(f"SAM3 prompts file not found: {resolved_path}")
    if timestamp is None:
        raise ValueError("Could not resolve timestamp from prompts path.")

    return resolved_path, timestamp


def load_prompt_list(prompts_path: Path, fallback_prompt: str | None) -> list[dict]:
    if fallback_prompt is not None and fallback_prompt.strip():
        prompt = fallback_prompt.strip()
        return [
            {
                "object_id": slugify_prompt(prompt),
                "prompt": prompt,
            }
        ]

    data = json.loads(prompts_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"SAM3 prompts file must contain a JSON list: {prompts_path}")

    prompts: list[dict] = []
    for index, item in enumerate(data):
        if isinstance(item, str):
            prompt = item.strip()
            if not prompt:
                raise ValueError(f"SAM3 prompt at index {index} must be non-empty.")
            prompts.append(
                {
                    "object_id": slugify_prompt(prompt),
                    "prompt": prompt,
                }
            )
            continue

        if isinstance(item, dict):
            object_id = item.get("object_id")
            prompt = item.get("prompt")
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError(f"SAM3 object_id at index {index} must be non-empty.")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"SAM3 prompt at index {index} must be non-empty.")
            prompts.append(
                {
                    **item,
                    "object_id": object_id.strip(),
                    "prompt": prompt.strip(),
                }
            )
            continue

        raise ValueError(
            f"SAM3 prompt item at index {index} must be a string or JSON object."
        )

    if not prompts:
        raise ValueError(f"SAM3 prompts file is empty: {prompts_path}")

    return prompts


def prepare_masks(masks: torch.Tensor) -> np.ndarray:
    """
    Convert SAM3 masks to a NumPy array with shape [N, H, W].
    """
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)

    masks = masks.detach().cpu()

    if masks.ndim == 4:
        if masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.shape[0] == 1:
            masks = masks[0]
    elif masks.ndim == 3:
        pass
    elif masks.ndim == 2:
        masks = masks.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(masks.shape)}")

    return masks.numpy()


def prepare_scores(scores: torch.Tensor) -> np.ndarray:
    """
    Convert SAM3 scores to a NumPy array with shape [N].
    """
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores)

    scores = scores.detach().cpu()

    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    elif scores.ndim == 0:
        scores = scores.unsqueeze(0)
    elif scores.ndim != 1:
        raise ValueError(f"Unsupported score shape: {tuple(scores.shape)}")

    return scores.to(torch.float32).numpy()


def build_binary_mask(
    masks,
    scores=None,
    score_threshold=None,
    mode: str = "best",
) -> np.ndarray:
    """
    Build a binary mask image:
    - object: 255
    - background: 0

    mode:
    - "best": keep only the highest-score mask
    - "union": merge all valid masks
    """
    masks_np = prepare_masks(masks)

    if masks_np.shape[0] == 0:
        raise ValueError("No masks were found for the prompt.")

    if scores is not None:
        scores_np = prepare_scores(scores)
    else:
        scores_np = None

    if scores_np is not None and score_threshold is not None:
        valid = scores_np >= score_threshold

        if not np.any(valid):
            raise ValueError(
                f"No mask passed score_threshold={score_threshold}. "
                f"Scores: {scores_np}"
            )

        masks_np = masks_np[valid]
        scores_np = scores_np[valid]

    if mode == "best" and scores_np is not None:
        best_idx = int(np.argmax(scores_np))
        masks_np = masks_np[best_idx : best_idx + 1]
    elif mode == "union":
        pass
    else:
        if mode != "best":
            raise ValueError(f"Unsupported mask mode: {mode}")

    # SAM3 may return logits or probabilities.
    # If values include negatives, treat them as logits and threshold at 0.
    # Otherwise, treat them as probabilities and threshold at 0.5.
    if masks_np.dtype == bool:
        masks_bool = masks_np
    elif masks_np.min() < 0:
        masks_bool = masks_np > 0
    else:
        masks_bool = masks_np > 0.5

    combined_mask = np.any(masks_bool, axis=0)

    binary_mask = np.zeros(combined_mask.shape, dtype=np.uint8)
    binary_mask[combined_mask] = 255

    return binary_mask


def segment_image(
    processor: Sam3Processor,
    image_path: Path,
    prompt: str,
    score_threshold: float | None,
    mask_mode: str,
) -> np.ndarray:
    """
    Run SAM3 on a single image and return a binary mask.
    """
    image = Image.open(image_path).convert("RGB")

    state = processor.set_image(image)
    processor.reset_all_prompts(state)

    output = processor.set_text_prompt(
        state=state,
        prompt=prompt,
    )

    if not isinstance(output, dict):
        raise TypeError(f"SAM3 output is not a dict: {type(output)}")

    for key in ["masks", "boxes", "scores"]:
        if key not in output:
            raise KeyError(f"Missing key in SAM3 output: {key}")

    masks = output["masks"]
    scores = output["scores"]

    binary_mask = build_binary_mask(
        masks=masks,
        scores=scores,
        score_threshold=score_threshold,
        mode=mask_mode,
    )

    return binary_mask


def main():
    parser = argparse.ArgumentParser()
    default_data_root = Path("/workspace/shared_data/realsense")

    parser.add_argument(
        "--data_root",
        type=str,
        default=str(default_data_root),
        help="Root directory containing rgb/depth/camera/objects folders.",
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help=(
            "Timestamp folder to process. If omitted, the latest folder under "
            "data_root/objects is used."
        ),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Optional single text prompt for SAM3. If omitted, prompts are loaded "
            "from data_root/objects/<timestamp>/sam3_prompts.json."
        ),
    )

    parser.add_argument(
        "--prompts_path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to sam3_prompts.json. If omitted, uses "
            "data_root/objects/<timestamp>/sam3_prompts.json."
        ),
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Optional minimum SAM3 score threshold.",
    )

    parser.add_argument(
        "--mask_mode",
        type=str,
        default="best",
        choices=["best", "union"],
        help="Use the best mask or union of all masks.",
    )

    parser.add_argument(
        "--image_id",
        type=str,
        default="000000",
        help="Image id to process, e.g. 000000.",
    )

    args = parser.parse_args()

    try:
        log("Starting SAM3 segmentation script")

        if torch.cuda.is_available():
            log("CUDA is available")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            autocast_context = torch.autocast("cuda", dtype=torch.bfloat16)
            autocast_context.__enter__()
        else:
            log("CUDA is not available, using CPU")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        data_root = Path(args.data_root)
        prompts_path, timestamp = resolve_prompts_path(
            data_root=data_root,
            timestamp=args.timestamp,
            prompts_path=args.prompts_path,
            require_prompts_file=args.prompt is None,
        )
        prompts = load_prompt_list(
            prompts_path=prompts_path,
            fallback_prompt=args.prompt,
        )

        rgb_dir = data_root / "rgb" / timestamp
        masks_root = data_root / "masks_by_object" / timestamp
        masks_root.mkdir(parents=True, exist_ok=True)

        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory does not exist: {rgb_dir}")

        sam3_root = Path(sam3.__file__).resolve().parent.parent
        bpe_path = sam3_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"

        log(f"SAM3 root: {sam3_root}")
        log(f"BPE path: {bpe_path}")
        log(f"Timestamp: {timestamp}")
        log(f"Prompts file: {prompts_path}")
        log(f"Prompt records: {prompts}")
        log(f"Output root: {masks_root}")

        model = build_sam3_image_model(bpe_path=str(bpe_path))
        processor = Sam3Processor(model, confidence_threshold=0.5)

        image_paths = [rgb_dir / f"{args.image_id}.png"]

        if len(image_paths) == 0:
            raise RuntimeError(f"No PNG images found in: {rgb_dir}")

        for image_path in image_paths:
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            log(f"Processing {image_path}")

            frame_id = image_path.stem
            for prompt_record in prompts:
                object_id = prompt_record["object_id"]
                prompt = prompt_record["prompt"]
                object_mask_dir = masks_root / object_id
                object_mask_dir.mkdir(parents=True, exist_ok=True)
                output_path = object_mask_dir / f"{frame_id}.png"

                log(f"Prompt: {prompt!r} -> object_id={object_id}")
                binary_mask = segment_image(
                    processor=processor,
                    image_path=image_path,
                    prompt=prompt,
                    score_threshold=args.score_threshold,
                    mask_mode=args.mask_mode,
                )

                Image.fromarray(binary_mask, mode="L").save(output_path)
                log(f"Saved mask to {output_path}")

        log("Done")

    except Exception as exc:
        print("\n[ERROR] SAM3 segmentation failed:")
        print(type(exc).__name__, exc)
        print("\n[TRACEBACK]")
        traceback.print_exc()


if __name__ == "__main__":
    main()








# import os
# import gc
# import traceback

# import numpy as np
# from PIL import Image

# import torch

# import sam3
# from sam3 import build_sam3_image_model
# from sam3.model.sam3_image_processor import Sam3Processor


# def log(msg):
#     print(f"[LOG] {msg}")


# def log_tensor(name, x, max_elements=10):
#     print(f"\n[DEBUG] {name}")
#     print(f"  type: {type(x)}")

#     if torch.is_tensor(x):
#         print(f"  shape: {tuple(x.shape)}")
#         print(f"  dtype: {x.dtype}")
#         print(f"  device: {x.device}")
#         try:
#             flat = x.detach().flatten()
#             n = min(flat.numel(), max_elements)
#             print(f"  first_values: {flat[:n].cpu().tolist()}")
#         except Exception as e:
#             print(f"  first_values: <errore nel dump: {e}>")
#     else:
#         try:
#             arr = np.array(x)
#             print(f"  np_shape: {arr.shape}")
#             print(f"  np_dtype: {arr.dtype}")
#             flat = arr.flatten()
#             n = min(flat.size, max_elements)
#             print(f"  first_values: {flat[:n].tolist()}")
#         except Exception as e:
#             print(f"  dump_error: {e}")


# def prepare_masks(masks):
#     """
#     Normalizza le masks in shape [N, H, W].
#     """
#     if not torch.is_tensor(masks):
#         masks = torch.as_tensor(masks)

#     masks = masks.detach().cpu()

#     log_tensor("prepare_masks.input", masks)

#     if masks.ndim == 4:
#         if masks.shape[1] == 1:
#             masks = masks[:, 0]
#         elif masks.shape[0] == 1:
#             masks = masks[0]
#     elif masks.ndim == 3:
#         pass
#     elif masks.ndim == 2:
#         masks = masks.unsqueeze(0)
#     else:
#         raise ValueError(f"Shape masks non supportata: {tuple(masks.shape)}")

#     log_tensor("prepare_masks.output", masks)
#     return masks.numpy()


# def prepare_scores(scores):
#     """
#     Normalizza gli scores in shape [N].
#     """
#     if not torch.is_tensor(scores):
#         scores = torch.as_tensor(scores)

#     scores = scores.detach().cpu()

#     log_tensor("prepare_scores.input", scores)

#     if scores.ndim == 2 and scores.shape[0] == 1:
#         scores = scores[0]
#     elif scores.ndim == 0:
#         scores = scores.unsqueeze(0)
#     elif scores.ndim != 1:
#         raise ValueError(f"Shape scores non supportata: {tuple(scores.shape)}")

#     scores = scores.to(torch.float32)

#     log_tensor("prepare_scores.output", scores)
#     return scores.numpy()


# def build_binary_segmentation(masks, scores=None, score_threshold=None):
#     """
#     Crea un'immagine binaria:
#     - oggetto segmentato = bianco, valore 255
#     - sfondo = nero, valore 0

#     Se ci sono più maschere per il prompt, le unisce tutte.
#     """
#     masks_np = prepare_masks(masks)

#     if len(masks_np) == 0:
#         raise ValueError("Nessuna maschera trovata per il prompt.")

#     if scores is not None and score_threshold is not None:
#         scores_np = prepare_scores(scores)
#         valid = scores_np >= score_threshold

#         if not np.any(valid):
#             raise ValueError(
#                 f"Nessuna maschera supera score_threshold={score_threshold}. "
#                 f"Scores trovati: {scores_np}"
#             )

#         masks_np = masks_np[valid]

#     # SAM può restituire logits/probabilità: soglia a 0 per logits o 0.5 per probabilità.
#     # Questa regola gestisce entrambi i casi in modo robusto.
#     if masks_np.dtype != bool:
#         if masks_np.min() < 0:
#             masks_bool = masks_np > 0
#         else:
#             masks_bool = masks_np > 0.5
#     else:
#         masks_bool = masks_np

#     # Unisce tutte le istanze segmentate del target.
#     combined_mask = np.any(masks_bool, axis=0)

#     binary_image = np.zeros(combined_mask.shape, dtype=np.uint8)
#     binary_image[combined_mask] = 255

#     return binary_image


# def run_prompt(processor, base_state, prompt_text):
#     log(f"Eseguo prompt: {prompt_text!r}")

#     processor.reset_all_prompts(base_state)
#     state = processor.set_text_prompt(state=base_state, prompt=prompt_text)

#     if not isinstance(state, dict):
#         raise TypeError(f"Lo state ritornato non è un dict: {type(state)}")

#     log(f"Chiavi state per prompt {prompt_text!r}: {list(state.keys())}")

#     required_keys = ["masks", "boxes", "scores"]
#     for key in required_keys:
#         if key not in state:
#             raise KeyError(f"Manca la chiave {key!r} nello state per prompt {prompt_text!r}")

#     masks = state["masks"]
#     boxes = state["boxes"]
#     scores = state["scores"]

#     log_tensor(f"{prompt_text}.masks", masks)
#     log_tensor(f"{prompt_text}.boxes", boxes)
#     log_tensor(f"{prompt_text}.scores", scores)

#     return masks, boxes, scores


# def main():
#     try:
#         log("Avvio script")

#         if torch.cuda.is_available():
#             log("CUDA disponibile")
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.backends.cudnn.allow_tf32 = True
#             torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#             log("Autocast CUDA bfloat16 attivato")
#         else:
#             log("CUDA NON disponibile: userò CPU")

#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             log("Cache CUDA svuotata")

#         sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
#         bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

#         log(f"sam3_root: {sam3_root}")
#         log(f"bpe_path: {bpe_path}")
#         log(f"bpe exists: {os.path.exists(bpe_path)}")

#         log("Costruisco il modello SAM3...")
#         model = build_sam3_image_model(bpe_path=bpe_path)
#         log("Modello SAM3 caricato")

#         INPUT_FILE = "0000001.png"
#         input_dir = "/home/ecuzzocrea-iit.local/sam3/sam3/input/fp"
#         image_path = os.path.join(input_dir, INPUT_FILE)

#         log(f"image_path: {image_path}")
#         log(f"image exists: {os.path.exists(image_path)}")

#         image = Image.open(image_path).convert("RGB")
#         width, height = image.size
#         log(f"image size: width={width}, height={height}")

#         processor = Sam3Processor(model, confidence_threshold=0.5)
#         log("Processor creato")

#         base_state = processor.set_image(image)
#         log("Immagine caricata nel processor")

#         if isinstance(base_state, dict):
#             log(f"Chiavi base_state: {list(base_state.keys())}")
#         else:
#             log(f"base_state type: {type(base_state)}")

#         masks_apple, boxes_apple, scores_apple = run_prompt(
#             processor,
#             base_state,
#             "box"
#         )

#         binary_segmentation = build_binary_segmentation(
#             masks=masks_apple,
#             scores=scores_apple,
#             score_threshold=None
#         )

#         output_dir = "output/fp"
#         os.makedirs(output_dir, exist_ok=True)

#         output_path = os.path.join(output_dir, INPUT_FILE)

#         Image.fromarray(binary_segmentation, mode="L").save(output_path)

#         log(f"Immagine binaria salvata in: {output_path}")
#         log("Oggetto segmentato = bianco, sfondo = nero")

#     except Exception as e:
#         print("\n[ERROR] Eccezione durante l'esecuzione:")
#         print(type(e).__name__, e)
#         print("\n[TRACEBACK COMPLETO]")
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()
