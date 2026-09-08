from backend.services.statistics import update_statistics
from backend.services.memory_monitor import log_memory

import io
import base64
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import time
import torch.nn.functional as F
from torchvision import transforms

from anomaly_detection import config
from anomaly_detection.model import AnomalyAutoencoder
from anomaly_detection.classifier import DefectClassifier, CATEGORY_DEFECT_CLASSES
from anomaly_detection.preprocessor import validate_and_preprocess_image, get_category_transforms
from anomaly_detection.severity import calculate_severity_score
from anomaly_detection.yolo_helper import crop_product

# ── Global Model Caches ─────────────────────────────────────────────────────
_AUTOENCODER_CACHE = {}
_CLASSIFIER_CACHE = {}

_CURRENT_AUTOENCODER_CATEGORY = None
_CURRENT_CLASSIFIER_CATEGORY = None

# ── Model Loaders ───────────────────────────────────────────────────────────

def load_autoencoder(category: str) -> AnomalyAutoencoder:
    """
    Loads only the currently required autoencoder.
    The previous category's model is released before loading a new one.
    """
    global _AUTOENCODER_CACHE, _CURRENT_AUTOENCODER_CATEGORY

    category = category.lower()

    # Reuse model if the same category is requested
    if (
        _CURRENT_AUTOENCODER_CATEGORY == category
        and category in _AUTOENCODER_CACHE
    ):
        return _AUTOENCODER_CACHE[category]

    # Release previous autoencoder
    if _AUTOENCODER_CACHE:
        old_category = _CURRENT_AUTOENCODER_CATEGORY
        print(f"[inference] Releasing autoencoder for '{old_category}'")

        old_model = _AUTOENCODER_CACHE.pop(old_category, None)

        if old_model is not None:
            del old_model

        _CURRENT_AUTOENCODER_CATEGORY = None

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = torch.device(config.DEVICE)

    model = AnomalyAutoencoder().to(device)

    weights_path = config.MODEL_DIR / f"autoencoder_{category}.pth"

    if weights_path.exists():
        try:
            model.load_state_dict(
                torch.load(
                    weights_path,
                    map_location=device,
                    weights_only=True
                )
            )
            print(f"[inference] Loaded autoencoder weights: {weights_path.name}")
        except Exception as e:
            print(
                f"[inference] Warning loading autoencoder "
                f"for '{category}': {e}"
            )
    else:
        print(
            f"[inference] No weights found for '{category}' "
            f"— running with random init."
        )

    model.eval()

    _AUTOENCODER_CACHE[category] = model
    _CURRENT_AUTOENCODER_CATEGORY = category

    return model


def load_classifier_model(category: str):
    """
    Loads only the currently required classifier.
    Releases the previous category's classifier from memory.
    """
    global _CLASSIFIER_CACHE, _CURRENT_CLASSIFIER_CATEGORY

    category = category.lower()

    # Reuse current classifier
    if (
        _CURRENT_CLASSIFIER_CATEGORY == category
        and category in _CLASSIFIER_CACHE
    ):
        return _CLASSIFIER_CACHE[category]

    # Release previous classifier
    if _CLASSIFIER_CACHE:
        old_category = _CURRENT_CLASSIFIER_CATEGORY
        print(f"[inference] Releasing classifier for '{old_category}'")

        old_entry = _CLASSIFIER_CACHE.pop(old_category, None)

        if old_entry is not None:
            old_model = old_entry[0]

            if old_model is not None:
                del old_model

        _CURRENT_CLASSIFIER_CATEGORY = None

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    device = torch.device(config.DEVICE)

    class_list = CATEGORY_DEFECT_CLASSES.get(
        category,
        ["good", "defective"]
    )

    model = DefectClassifier(
        num_classes=len(class_list)
    ).to(device)

    weights_path = config.MODEL_DIR / f"classifier_{category}.pth"

    loaded = False

    if weights_path.exists():
        try:
            model.load_state_dict(
                torch.load(
                    weights_path,
                    map_location=device,
                    weights_only=True
                )
            )

            loaded = True

            print(
                f"[inference] Loaded classifier weights: "
                f"{weights_path.name}"
            )

        except Exception as e:
            print(
                f"[inference] Warning loading classifier "
                f"for '{category}': {e}"
            )

    model.eval()

    result = (model if loaded else None, class_list)

    _CLASSIFIER_CACHE[category] = result
    _CURRENT_CLASSIFIER_CATEGORY = category

    return result


# ── SSIM Computation ────────────────────────────────────────────────────────

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Creates a 2D Gaussian kernel for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()


def compute_ssim_map(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Computes the per-pixel SSIM map between two image tensors.
    Both tensors should be shape (B, C, H, W) in [0, 1].
    Returns an anomaly map of shape (B, H, W) where 1.0 = perfect match.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    kernel = _gaussian_kernel(11, 1.5).to(img1.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, 11, 11)

    ssim_channels = []
    for c in range(img1.shape[1]):
        x = img1[:, c:c+1, :, :]
        y = img2[:, c:c+1, :, :]

        mu_x = F.conv2d(x, kernel, padding=5)
        mu_y = F.conv2d(y, kernel, padding=5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sig_x2  = F.conv2d(x * x, kernel, padding=5) - mu_x2
        sig_y2  = F.conv2d(y * y, kernel, padding=5) - mu_y2
        sig_xy  = F.conv2d(x * y, kernel, padding=5) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sig_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
        ssim_map = num / (den + 1e-8)
        ssim_channels.append(ssim_map)

    # Average across channels; anomaly = 1 - SSIM
    ssim = torch.stack(ssim_channels, dim=1).mean(dim=1)  # (B, H, W)
    return ssim


# ── Image Utilities ─────────────────────────────────────────────────────────

def pil_to_base64_uri(pil_img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Converts a (C, H, W) tensor in [0,1] to a PIL Image."""
    arr = t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ── Main Inference Pipeline ─────────────────────────────────────────────────

def predict_defect(image_input, category: str = "bottle", enable_yolo: bool = True) -> dict:
    """
    Full manufacturing quality inspection pipeline.

    Stages:
        1. Image preprocessing & quality check
        2. YOLO object crop (for applicable categories)
        3. Autoencoder reconstruction
        4. SSIM anomaly map computation
        5. PASS / REJECT decision via calibrated threshold
        6. Defect classification (naming)
        7. Severity scoring
        8. Heatmap & overlay generation

    Returns a dict with result, images (base64), scores, and metadata.
    """
    start_time = time.perf_counter()
    category = category.lower()
    log_memory("START")

    # ── 1. Input Parsing & Auto Category Detection ───────────────────────────
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        pil_img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image_input type.")

    # Product category auto-detection if requested
    if category.lower() in ["auto", "autodetect", "detect", ""]:
        from anomaly_detection.classifier import detect_product_category
        category, auto_conf = detect_product_category(pil_img)
        print(f"[inference] Auto-detected product category: '{category}' ({auto_conf}% confidence)")
    else:
        category = category.lower()

    # ── 2. Quality Preprocessing (Non-destructive) ───────────────────────────
    enhanced_img, quality_report = validate_and_preprocess_image(pil_img, enhance=False)

    # ── 3. YOLO Object Crop ─────────────────────────────────────────────────
    yolo_skip = getattr(config, "YOLO_SKIP_CATEGORIES",
                        {"carpet", "grid", "leather", "tile", "wood"})
    use_yolo = enable_yolo and (category not in yolo_skip)

    if use_yolo:
        cropped_img, bbox, yolo_status = crop_product(pil_img, category=category, enable_yolo=True)
        # Fallback: if YOLO returned OOD signal, use full image and flag
        if yolo_status.startswith("INVALID"):
            cropped_img = pil_img.copy()
            bbox = [0, 0, pil_img.width, pil_img.height]
    else:
        from anomaly_detection.yolo_helper import _fallback_crop

        cropped_img, bbox = _fallback_crop(pil_img)

        yolo_status = (
            f"YOLO: Bypassed for category '{category}' "
            f"— using fallback crop"
        )

    # ── 4. Autoencoder Reconstruction ───────────────────────────────────────
    device = torch.device(config.DEVICE)
    ae_model = load_autoencoder(category)

    # Use centralized category-aware transforms matching training
    transform = get_category_transforms(
        category=category,
        split="test",
        image_size=config.IMAGE_SIZE
    )
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Enable cuDNN benchmark for inference acceleration
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    log_memory("BEFORE AUTOENCODER")
    with torch.inference_mode():
        reconstructed_tensor = ae_model(input_tensor)              # (1, 3, 128, 128)
    log_memory("AFTER AUTOENCODER")

    # ── 5. Hybrid MAE Anomaly Map & Peak Score ──────────────────────────────
    diff_tensor = torch.abs(input_tensor - reconstructed_tensor)     # (1, 3, H, W)
    anomaly_map = diff_tensor.mean(dim=1)                            # (1, H, W)

    mean_mae = float(anomaly_map.mean().item())
    # Top 1.5% peak pixel errors to detect localized cracks, scratches, holes
    top_k_val = int(anomaly_map.numel() * 0.015)
    top_mae = float(torch.topk(anomaly_map.view(-1), k=max(1, top_k_val)).values.mean().item())

    # Hybrid composite anomaly score
    anomaly_score = round(0.4 * mean_mae + 0.6 * top_mae, 5)
    threshold     = float(config.CATEGORY_THRESHOLDS.get(category, config.ANOMALY_THRESHOLD))

    # ── 6. PASS / REJECT Decision ────────────────────────────────────────────
    is_anomaly = anomaly_score > threshold
    is_ood = anomaly_score > (threshold * 10.0)

    if is_ood:
        defect_result = "INVALID_IMAGE"
    elif is_anomaly:
        defect_result = "REJECT"
    else:
        defect_result = "PASS"

    # ── 7. Defect Classification ─────────────────────────────────────────────
    predicted_class   = "good"
    class_confidence  = 99.0
    class_probs       = {"good": 99.0}

    if is_ood:
        predicted_class  = "out_of_distribution"
        class_confidence = 99.9
    elif is_anomaly:
        classifier_model, class_list = load_classifier_model(category)
        log_memory("BEFORE CLASSIFIER")
        if classifier_model is not None:
            # Exclude 'good' when an anomaly is detected so specific defect type is identified
            predicted_class, class_confidence, class_probs = classifier_model.predict_class(
                input_tensor, class_list, exclude_good=True
            )
            log_memory("AFTER CLASSIFIER")
            if predicted_class == "good" or predicted_class == "class_0":
                predicted_class  = "defect_detected"
                class_confidence = round(min(99.0, max(70.0, (anomaly_score / max(1e-6, threshold)) * 40.0)), 2)
        else:
            predicted_class  = "defect_detected"
            class_confidence = round(min(99.0, max(70.0, (anomaly_score / max(1e-6, threshold)) * 40.0)), 2)

    # ── 8. Severity Scoring ──────────────────────────────────────────────────
    anomaly_map_np = anomaly_map.squeeze().cpu().numpy()          # (H, W) — severity.py needs 2D

    severity_dict = calculate_severity_score(
        anomaly_map=anomaly_map_np,
        anomaly_score=anomaly_score,
        threshold=threshold,
        defect_type=predicted_class if predicted_class not in ("good", None) else None,
    )

    # Refine defect name from severity inference if classifier was vague
    if is_anomaly and predicted_class in ("defect_detected", "good"):
        inferred = severity_dict.get("inferred_defect_type", "defect_detected")
        predicted_class  = inferred
        class_confidence = round(min(99.0, max(75.0, (anomaly_score / max(1e-6, threshold)) * 50.0)), 2)
        class_probs[predicted_class] = class_confidence

    # ── 9. Heatmap & Overlay Generation ────────────────────────────────────
    import cv2

    norm_map   = (anomaly_map_np - anomaly_map_np.min()) / (anomaly_map_np.max() - anomaly_map_np.min() + 1e-8)
    heat_u8    = (norm_map * 255.0).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heat_color).resize(cropped_img.size)

    crop_np   = np.array(cropped_img)
    overlay_np = cv2.addWeighted(crop_np, 0.55, np.array(heatmap_pil), 0.45, 0)
    overlay_pil = Image.fromarray(overlay_np)

    reconstructed_pil = tensor_to_pil(reconstructed_tensor)
    reconstructed_pil = reconstructed_pil.resize(cropped_img.size)

    # ── 10. Update Statistics ───────────────────────────────────────────────

    if defect_result == "PASS":
        update_statistics("Good")
    elif defect_result == "REJECT":
        update_statistics("Defective")

    # ── 11. Return Result ───────────────────────────────────────────────────


    processing_time_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2
    )

    result = {
        "category": category,

        "is_anomaly": is_anomaly,
        "defect_result": defect_result,
        "defect_class": predicted_class,
        "confidence_score": class_confidence,

        "anomaly_score": round(anomaly_score, 6),
        "threshold": round(threshold, 6),

        "severity_score": round(
            severity_dict["severity_score"],
            2
        ),
        "severity_level": severity_dict["severity_level"],
        "recommended_action": severity_dict["recommended_action"],

        "yolo_status": yolo_status,
        "bbox": bbox,

        "class_probabilities": class_probs,

        "quality_report": quality_report,

        "severity_breakdown": severity_dict["breakdown"],

        "processing_time_ms": processing_time_ms,

        "images": {
            "original": pil_to_base64_uri(pil_img),
            "cropped": pil_to_base64_uri(cropped_img),
            "reconstructed": pil_to_base64_uri(reconstructed_pil),
            "heatmap": pil_to_base64_uri(heatmap_pil),
            "overlay": pil_to_base64_uri(overlay_pil),
        },
    }

    # -----------------------------------------
    # Release inference-only tensors
    # -----------------------------------------
    del input_tensor
    del reconstructed_tensor
    del diff_tensor
    del anomaly_map
    del anomaly_map_np
    del heat_u8
    del heat_color
    del crop_np
    del overlay_np
    del ae_model

    import gc
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log_memory("END")

    return result