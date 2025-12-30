import io
import time
import json
import numpy as np
import cv2
from collections import defaultdict, Counter
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from PIL import Image
from skimage import color
from sklearn.cluster import KMeans
from pathlib import Path
import uvicorn

app = FastAPI(title="Knit pixel - Image Conversion")

# --- DEFAULT CONFIG ---
DEFAULT_PALETTE_HEX = [
    "#FF90A7", "#EE444A", "#FFE201", "#FF8100", "#8F56FE",
    "#F443E0", "#00DA00", "#00B587", "#148EFF", "#00ECFE",
    "#E0E1EB", "#FED495", "#39414B", "#77422A", "#966E46",
    "#441b6d", "#718d24", "#403dc9", "#849ca8", "#efad29"
]

def hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

# ==========================================
# PART 1: ADVANCED GRID DETECTION (Dual Strategy)
# ==========================================

def merge_close_positions(pos: np.ndarray, merge_dist: int) -> np.ndarray:
    if pos.size == 0: return pos
    pos = np.sort(pos)
    groups = [[int(pos[0])]]
    for p in pos[1:]:
        if abs(int(p) - groups[-1][-1]) <= merge_dist:
            groups[-1].append(int(p))
        else:
            groups.append([int(p)])
    return np.array([int(np.mean(g)) for g in groups], dtype=int)

def find_lines_segments(profile: np.ndarray, thr: float, min_run: int = 2, merge_dist: int = 2) -> np.ndarray:
    x = profile.astype(np.float32)
    mask = x >= float(thr)
    if mask.sum() == 0: return np.array([], dtype=int)
    idx = np.where(mask)[0]
    cuts = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, cuts)
    centers = []
    for g in groups:
        if g.size >= min_run:
            centers.append(int(np.mean(g)))
    if not centers: return np.array([], dtype=int)
    centers = np.array(centers, dtype=int)
    return merge_close_positions(centers, merge_dist=merge_dist)

def vote_step_from_lines(lines: np.ndarray, min_step: int = 4, max_step: int = 200, bin_tol: int = 2) -> tuple[int, float]:
    if lines.size < 3: return 0, 0.0
    diffs = np.diff(np.sort(lines)).astype(int)
    diffs = diffs[(diffs >= min_step) & (diffs <= max_step)]
    if diffs.size == 0: return 0, 0.0
    
    votes = defaultdict(int)
    for d in diffs:
        key = int(round(d / bin_tol) * bin_tol)
        votes[key] += 1
    
    best_bin = max(votes.items(), key=lambda kv: kv[1])[0]
    total = int(diffs.size)
    conf = votes[best_bin] / total if total else 0.0
    
    near = diffs[(diffs >= best_bin - bin_tol) & (diffs <= best_bin + bin_tol)]
    best_val = int(np.median(near)) if near.size else best_bin
    return best_val, float(conf)

def enhance_for_grid(img_bgr: np.ndarray) -> np.ndarray:
    x = cv2.fastNlMeansDenoisingColored(img_bgr, None, 3, 3, 7, 21)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L2 = clahe.apply(L)
    x = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    return x

def detect_grid_by_lines(img_bgr: np.ndarray) -> dict:
    """Strategy 1: Find dark/light grid lines using thresholding"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k3, iterations=1)
    
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 40), 1))
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 40)))
    hor = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k)
    ver = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k)
    
    lines_mask = cv2.bitwise_or(hor, ver)
    
    row_prof = lines_mask.sum(axis=1)
    col_prof = lines_mask.sum(axis=0)
    
    r_thr = max(np.mean(row_prof) + 0.5 * np.std(row_prof), w * 0.1)
    c_thr = max(np.mean(col_prof) + 0.5 * np.std(col_prof), h * 0.1)
    
    h_lines = find_lines_segments(row_prof, r_thr)
    v_lines = find_lines_segments(col_prof, c_thr)
    
    step_h, conf_h = vote_step_from_lines(h_lines)
    step_v, conf_v = vote_step_from_lines(v_lines)
    
    return {"step_h": step_h, "conf_h": conf_h, "step_v": step_v, "conf_v": conf_v}

def detect_grid_by_gradients(img_bgr: np.ndarray) -> dict:
    """Strategy 2: Find sharp edges (gradients) for grid-less pixel art"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
    
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # Project gradients to profiles
    col_prof = np.sum(abs_sobelx, axis=0)
    row_prof = np.sum(abs_sobely, axis=1)
    
    r_thr = np.mean(row_prof) + 1.0 * np.std(row_prof)
    c_thr = np.mean(col_prof) + 1.0 * np.std(col_prof)
    
    # Edges are usually thin, min_run=1
    h_edges = find_lines_segments(row_prof, r_thr, min_run=1, merge_dist=2)
    v_edges = find_lines_segments(col_prof, c_thr, min_run=1, merge_dist=2)
    
    step_h, conf_h = vote_step_from_lines(h_edges, min_step=4)
    step_v, conf_v = vote_step_from_lines(v_edges, min_step=4)
    
    return {"step_h": step_h, "conf_h": conf_h, "step_v": step_v, "conf_v": conf_v}

def detect_grid_auto(image: Image.Image) -> dict:
    """Auto-detect grid using dual strategy (Lines vs Gradients)"""
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]
    
    img_enhanced = enhance_for_grid(img_bgr)
    
    res_lines = detect_grid_by_lines(img_enhanced)
    res_grads = detect_grid_by_gradients(img_enhanced)
    
    conf_lines = (res_lines["conf_h"] + res_lines["conf_v"]) / 2
    conf_grads = (res_grads["conf_h"] + res_grads["conf_v"]) / 2
    
    print(f"DEBUG: Line Conf={conf_lines:.2f}, Gradient Conf={conf_grads:.2f}")
    
    if conf_lines >= conf_grads and conf_lines > 0.1:
        chosen = res_lines
        method = "lines"
    elif conf_grads > 0.1:
        chosen = res_grads
        method = "gradients"
    else:
        chosen = res_lines if conf_lines > conf_grads else res_grads
        method = "fallback"

    step_h = chosen["step_h"]
    step_v = chosen["step_v"]
    
    final_w, final_h = 0, 0
    
    if step_h > 0 and step_v > 0:
        if abs(step_h - step_v) <= 2:
            avg = int(round((step_h + step_v) / 2))
            final_w = final_h = avg
        else:
            if chosen["conf_h"] > chosen["conf_v"]:
                final_h = final_w = step_h
            else:
                final_w = final_h = step_v
    elif step_h > 0:
        final_w = final_h = step_h
    elif step_v > 0:
        final_w = final_h = step_v
    else:
        # Fallback
        final_w = max(1, w // 20)
        final_h = max(1, h // 20)

    num_cols = max(1, w // final_w)
    num_rows = max(1, h // final_h)
    
    return {
        "cell_width": final_w,
        "cell_height": final_h,
        "num_cols": num_cols,
        "num_rows": num_rows,
        "method": method
    }

# ==========================================
# PART 2: SMART COLOR MAPPING (LOGIC)
# ==========================================

def quantize_image_smart(img: Image.Image, cols: int, rows: int, cell_size: int, target_k: int, palette_hex_list: list) -> Image.Image:
    """
    Resize -> K-Means (LAB) -> Mapping (CIEDE2000) -> Collision Resolution -> Tái tạo
    """
    
    # --- PREPARE PALETTE ---
    palette_rgb_u8 = np.array([hex_to_rgb(c) for c in palette_hex_list], dtype=np.uint8)
    palette_rgb_norm = palette_rgb_u8.astype(np.float32) / 255.0
    palette_lab = color.rgb2lab(palette_rgb_norm.reshape(1, -1, 3)).reshape(-1, 3)

    # --- RESIZE & PREPARE IMAGE ---
    img_small = img.resize((cols, rows), Image.NEAREST).convert("RGB")
    img_small_array = np.array(img_small)
    h, w, d = img_small_array.shape 
    
    pixels_small = img_small_array.reshape((h * w, d))

    # Convert pixels to LAB
    pixels_small_float = pixels_small.astype(np.float32) / 255.0
    pixels_lab = color.rgb2lab(pixels_small_float.reshape(1, -1, 3)).reshape(-1, 3)

    print(f"Clustering into {target_k} segments (in LAB space)...")

    # --- K-MEANS CLUSTERING ---
    n_samples = pixels_lab.shape[0]
    actual_k = min(target_k, n_samples)
    if actual_k < 1: actual_k = 1
    
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=20)
    pixel_labels = kmeans.fit_predict(pixels_lab)
    dominant_colors_lab = kmeans.cluster_centers_

    # --- MAPPING: BUILD DISTANCE MATRIX ---
    dist_matrix = np.zeros((actual_k, palette_lab.shape[0]), dtype=np.float64)

    for i, center_lab in enumerate(dominant_colors_lab):
        center_stack = np.tile(center_lab, (palette_lab.shape[0], 1))
        dist_matrix[i, :] = color.deltaE_ciede2000(center_stack, palette_lab)

    # --- INITIAL ASSIGNMENT ---
    initial_assignment = np.argmin(dist_matrix, axis=1)

    # --- CONFLICT RESOLUTION ---
    final_mapping = {} 
    palette_usage = defaultdict(list)
    
    for cluster_idx, pal_idx in enumerate(initial_assignment):
        palette_usage[pal_idx].append(cluster_idx)

    HALLUCINATION_THRESHOLD = 1.0
    used_palette_indices = set(palette_usage.keys())

    for pal_idx, clusters in palette_usage.items():
        if len(clusters) == 1:
            final_mapping[clusters[0]] = pal_idx
        else:
            clusters.sort(key=lambda c_idx: dist_matrix[c_idx][pal_idx])
            winner = clusters[0]
            final_mapping[winner] = pal_idx
            
            losers = clusters[1:]
            for l_idx in losers:
                sorted_indices = np.argsort(dist_matrix[l_idx])
                found_new_home = False
                original_best_dist = dist_matrix[l_idx][pal_idx]
                
                for candidate_pal_idx in sorted_indices:
                    if candidate_pal_idx == pal_idx: continue
                    
                    if candidate_pal_idx not in used_palette_indices:
                        new_dist = dist_matrix[l_idx][candidate_pal_idx]
                        if new_dist - original_best_dist < HALLUCINATION_THRESHOLD:
                            final_mapping[l_idx] = candidate_pal_idx
                            used_palette_indices.add(candidate_pal_idx)
                            found_new_home = True
                            break
                
                if not found_new_home:
                    final_mapping[l_idx] = pal_idx

    # --- RECONSTRUCTION ---
    final_pixels = np.zeros_like(pixels_small)
    
    for i in range(actual_k):
        mask = (pixel_labels == i)
        pal_idx = final_mapping.get(i, initial_assignment[i])
        color_val = palette_rgb_u8[pal_idx]
        final_pixels[mask] = color_val

    result_array = final_pixels.reshape((rows, cols, 3)).astype(np.uint8)
    img_result_small = Image.fromarray(result_array)
    
    out_w = cols * cell_size
    out_h = rows * cell_size
    return img_result_small.resize((out_w, out_h), Image.NEAREST)


def calculate_color_stats(image: Image.Image) -> dict:
    img_array = np.array(image.convert("RGB"))
    h, w = img_array.shape[:2]
    pixels = img_array.reshape(-1, 3)
    
    pixel_tuples = [tuple(p) for p in pixels]
    counts = Counter(pixel_tuples)
    sorted_colors = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    result = {
        "total_pixels": h * w,
        "unique_colors": len(counts),
        "color_stats": []
    }
    
    def rgb2hex(r, g, b): return f"#{r:02X}{g:02X}{b:02X}"

    for rgb, count in sorted_colors:
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        percentage = (count / result["total_pixels"]) * 100.0
        result["color_stats"].append({
            "hex": rgb2hex(r, g, b),
            "rgb": (r, g, b),
            "count": count,
            "percentage": round(percentage, 2)
        })
    return result

# ==========================================
# PART 3: API ENDPOINTS
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_path = Path("templates/index.html")
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Template not found. Please ensure 'templates/index.html' exists.</h1>"

@app.post("/detect-grid")
async def detect_grid(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        result = detect_grid_auto(img)
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process-image")
async def process_image_endpoint(
    file: UploadFile = File(...),
    cell_width: int = Form(..., gt=0),
    cell_height: int = Form(..., gt=0),
    cell_size: int = Form(1, gt=0),
    use_palette: str = Form("true"),
    target_k: int = Form(20),
    palette: str = Form(None)
):
    try:
        start = time.time()
        use_palette_bool = use_palette.lower() in ("true", "1", "on", "yes")

        current_palette = DEFAULT_PALETTE_HEX
        if palette:
            try:
                parsed = json.loads(palette)
                if isinstance(parsed, list) and len(parsed) > 0:
                    current_palette = parsed
            except:
                pass 

        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        width, height = img.size
        cols = max(1, round(width / cell_width))
        rows = max(1, round(height / cell_height))
        
        if use_palette_bool:
            out = quantize_image_smart(
                img, 
                cols=cols, 
                rows=rows, 
                cell_size=cell_size, 
                target_k=target_k, 
                palette_hex_list=current_palette
            )
        else:
            img_small = img.resize((cols, rows), Image.NEAREST)
            out = img_small.resize((cols * cell_size, rows * cell_size), Image.NEAREST)

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)

        print(f"Processed: {width}x{height} -> {cols}x{rows} grid using K={target_k}. Time: {time.time() - start:.2f}s")
        return StreamingResponse(buf, media_type="image/png")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/get-color-stats")
async def get_color_stats(
    file: UploadFile = File(...),
    cell_width: int = Form(..., gt=0),
    cell_height: int = Form(..., gt=0),
    cell_size: int = Form(1, gt=0),
    use_palette: str = Form("true"),
    target_k: int = Form(20),
    palette: str = Form(None)
):
    try:
        use_palette_bool = use_palette.lower() in ("true", "1", "on", "yes")
        
        current_palette = DEFAULT_PALETTE_HEX
        if palette:
            try:
                parsed = json.loads(palette)
                if isinstance(parsed, list) and len(parsed) > 0:
                    current_palette = parsed
            except:
                pass

        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        width, height = img.size
        cols = max(1, round(width / cell_width))
        rows = max(1, round(height / cell_height))

        if use_palette_bool:
            out = quantize_image_smart(
                img, 
                cols=cols, 
                rows=rows, 
                cell_size=cell_size, 
                target_k=target_k, 
                palette_hex_list=current_palette
            )
        else:
            img_small = img.resize((cols, rows), Image.NEAREST)
            out = img_small.resize((cols * cell_size, rows * cell_size), Image.NEAREST)
        
        stats = calculate_color_stats(out)
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)

