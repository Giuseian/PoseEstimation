import os
import gc
import traceback

import numpy as np
from PIL import Image

import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def log(msg):
    print(f"[LOG] {msg}")


def log_tensor(name, x, max_elements=10):
    print(f"\n[DEBUG] {name}")
    print(f"  type: {type(x)}")

    if torch.is_tensor(x):
        print(f"  shape: {tuple(x.shape)}")
        print(f"  dtype: {x.dtype}")
        print(f"  device: {x.device}")
        try:
            flat = x.detach().flatten()
            n = min(flat.numel(), max_elements)
            print(f"  first_values: {flat[:n].cpu().tolist()}")
        except Exception as e:
            print(f"  first_values: <errore nel dump: {e}>")
    else:
        try:
            arr = np.array(x)
            print(f"  np_shape: {arr.shape}")
            print(f"  np_dtype: {arr.dtype}")
            flat = arr.flatten()
            n = min(flat.size, max_elements)
            print(f"  first_values: {flat[:n].tolist()}")
        except Exception as e:
            print(f"  dump_error: {e}")


def prepare_masks(masks):
    """
    Normalizza le masks in shape [N, H, W].
    """
    if not torch.is_tensor(masks):
        masks = torch.as_tensor(masks)

    masks = masks.detach().cpu()

    log_tensor("prepare_masks.input", masks)

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
        raise ValueError(f"Shape masks non supportata: {tuple(masks.shape)}")

    log_tensor("prepare_masks.output", masks)
    return masks.numpy()


def prepare_scores(scores):
    """
    Normalizza gli scores in shape [N].
    """
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores)

    scores = scores.detach().cpu()

    log_tensor("prepare_scores.input", scores)

    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    elif scores.ndim == 0:
        scores = scores.unsqueeze(0)
    elif scores.ndim != 1:
        raise ValueError(f"Shape scores non supportata: {tuple(scores.shape)}")

    scores = scores.to(torch.float32)

    log_tensor("prepare_scores.output", scores)
    return scores.numpy()


def build_binary_segmentation(masks, scores=None, score_threshold=None):
    """
    Crea un'immagine binaria:
    - oggetto segmentato = bianco, valore 255
    - sfondo = nero, valore 0

    Se ci sono più maschere per il prompt, le unisce tutte.
    """
    masks_np = prepare_masks(masks)

    if len(masks_np) == 0:
        raise ValueError("Nessuna maschera trovata per il prompt.")

    if scores is not None and score_threshold is not None:
        scores_np = prepare_scores(scores)
        valid = scores_np >= score_threshold

        if not np.any(valid):
            raise ValueError(
                f"Nessuna maschera supera score_threshold={score_threshold}. "
                f"Scores trovati: {scores_np}"
            )

        masks_np = masks_np[valid]

    # SAM può restituire logits/probabilità: soglia a 0 per logits o 0.5 per probabilità.
    # Questa regola gestisce entrambi i casi in modo robusto.
    if masks_np.dtype != bool:
        if masks_np.min() < 0:
            masks_bool = masks_np > 0
        else:
            masks_bool = masks_np > 0.5
    else:
        masks_bool = masks_np

    # Unisce tutte le istanze segmentate del target.
    combined_mask = np.any(masks_bool, axis=0)

    binary_image = np.zeros(combined_mask.shape, dtype=np.uint8)
    binary_image[combined_mask] = 255

    return binary_image


def run_prompt(processor, base_state, prompt_text):
    log(f"Eseguo prompt: {prompt_text!r}")

    processor.reset_all_prompts(base_state)
    state = processor.set_text_prompt(state=base_state, prompt=prompt_text)

    if not isinstance(state, dict):
        raise TypeError(f"Lo state ritornato non è un dict: {type(state)}")

    log(f"Chiavi state per prompt {prompt_text!r}: {list(state.keys())}")

    required_keys = ["masks", "boxes", "scores"]
    for key in required_keys:
        if key not in state:
            raise KeyError(f"Manca la chiave {key!r} nello state per prompt {prompt_text!r}")

    masks = state["masks"]
    boxes = state["boxes"]
    scores = state["scores"]

    log_tensor(f"{prompt_text}.masks", masks)
    log_tensor(f"{prompt_text}.boxes", boxes)
    log_tensor(f"{prompt_text}.scores", scores)

    return masks, boxes, scores


def main():
    try:
        log("Avvio script")

        if torch.cuda.is_available():
            log("CUDA disponibile")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            log("Autocast CUDA bfloat16 attivato")
        else:
            log("CUDA NON disponibile: userò CPU")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log("Cache CUDA svuotata")

        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

        log(f"sam3_root: {sam3_root}")
        log(f"bpe_path: {bpe_path}")
        log(f"bpe exists: {os.path.exists(bpe_path)}")

        log("Costruisco il modello SAM3...")
        model = build_sam3_image_model(bpe_path=bpe_path)
        log("Modello SAM3 caricato")

        INPUT_FILE = "0000001.png"
        input_dir = "/home/ecuzzocrea-iit.local/sam3/sam3/input/fp"
        image_path = os.path.join(input_dir, INPUT_FILE)

        log(f"image_path: {image_path}")
        log(f"image exists: {os.path.exists(image_path)}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        log(f"image size: width={width}, height={height}")

        processor = Sam3Processor(model, confidence_threshold=0.5)
        log("Processor creato")

        base_state = processor.set_image(image)
        log("Immagine caricata nel processor")

        if isinstance(base_state, dict):
            log(f"Chiavi base_state: {list(base_state.keys())}")
        else:
            log(f"base_state type: {type(base_state)}")

        masks_apple, boxes_apple, scores_apple = run_prompt(
            processor,
            base_state,
            "box"
        )

        binary_segmentation = build_binary_segmentation(
            masks=masks_apple,
            scores=scores_apple,
            score_threshold=None
        )

        output_dir = "output/fp"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, INPUT_FILE)

        Image.fromarray(binary_segmentation, mode="L").save(output_path)

        log(f"Immagine binaria salvata in: {output_path}")
        log("Oggetto segmentato = bianco, sfondo = nero")

    except Exception as e:
        print("\n[ERROR] Eccezione durante l'esecuzione:")
        print(type(e).__name__, e)
        print("\n[TRACEBACK COMPLETO]")
        traceback.print_exc()


if __name__ == "__main__":
    main()