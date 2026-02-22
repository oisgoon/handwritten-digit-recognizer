import glob
import os

import joblib
import numpy as np
import pygame
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_fill_holes, center_of_mass, gaussian_filter, label, rotate, shift, zoom
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = os.path.join(os.path.dirname(__file__), "digit_model.joblib")
MODEL_VERSION = 5
FEATURE_SIZE = 20
DRAW_SIZE = 280
BRUSH_RADIUS = 10

WINDOW_W = 1000
WINDOW_H = 620
DRAW_RECT = pygame.Rect(40, 90, 420, 420)
GRAPH_RECT = pygame.Rect(520, 120, 430, 360)

BG = (245, 248, 252)
BLACK = (15, 20, 28)
WHITE = (245, 245, 245)
BORDER = (180, 190, 205)
BAR = (153, 176, 201)
BAR_HI = (255, 127, 80)
TEXT = (23, 33, 52)
BTN = (80, 110, 160)
BTN_TEXT = (245, 245, 245)


def _center_crop_or_pad(img, target):
    if img.shape[0] > target:
        s = (img.shape[0] - target) // 2
        img = img[s : s + target, s : s + target]
    elif img.shape[0] < target:
        pad = target - img.shape[0]
        p1 = pad // 2
        p2 = pad - p1
        img = np.pad(img, ((p1, p2), (p1, p2)), mode="constant")
    return img


def _gray_to_feature(gray):
    canvas28 = _gray_to_canvas28(gray)
    out = Image.fromarray(np.uint8(np.clip(canvas28, 0, 255)), mode="L").resize(
        (FEATURE_SIZE, FEATURE_SIZE), Image.Resampling.LANCZOS
    )
    feat = np.asarray(out, dtype=np.float32) / 255.0
    return feat.reshape(-1)


def _gray_to_canvas28(gray):
    gray = gray.astype(np.float32)
    ys, xs = np.where(gray > 12)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = gray[y1 : y2 + 1, x1 : x2 + 1]

    h, w = crop.shape
    side = max(h, w)
    margin = int(side * 0.25) + 2
    sq = np.zeros((side + 2 * margin, side + 2 * margin), dtype=np.float32)
    y_off = (sq.shape[0] - h) // 2
    x_off = (sq.shape[1] - w) // 2
    sq[y_off : y_off + h, x_off : x_off + w] = crop

    sq = gaussian_filter(sq, sigma=0.8)
    img20 = Image.fromarray(np.uint8(np.clip(sq, 0, 255)), mode="L").resize(
        (20, 20), Image.Resampling.LANCZOS
    )
    arr20 = np.asarray(img20, dtype=np.float32)

    canvas28 = np.zeros((28, 28), dtype=np.float32)
    canvas28[4:24, 4:24] = arr20

    cy, cx = center_of_mass(canvas28)
    if not np.isnan(cx) and not np.isnan(cy):
        canvas28 = shift(canvas28, shift=(13.5 - cy, 13.5 - cx), order=1, mode="constant")

    return canvas28


def _extract_shape_stats(canvas28):
    bw = canvas28 > 52.0
    total = int(bw.sum())
    if total == 0:
        return {
            "hole_count": 0,
            "upper_ratio": 0.0,
            "right_ratio": 0.0,
            "top_band_ratio": 0.0,
            "diag_ratio": 0.0,
            "mid_cross_ratio": 0.0,
        }

    filled = binary_fill_holes(bw)
    holes = np.logical_and(filled, np.logical_not(bw))
    _, hole_count = label(holes)

    upper_ratio = float(bw[:14, :].sum() / total)
    right_ratio = float(bw[:, 14:].sum() / total)
    top_band_ratio = float(bw[2:7, :].sum() / total)
    mid_cross_ratio = float(bw[11:17, 9:19].sum() / total)

    ys = np.linspace(4, 24, num=20).astype(int)
    xs = np.linspace(22, 8, num=20).astype(int)
    diag_ratio = float(np.mean(bw[ys, xs]))

    return {
        "hole_count": int(hole_count),
        "upper_ratio": upper_ratio,
        "right_ratio": right_ratio,
        "top_band_ratio": top_band_ratio,
        "diag_ratio": diag_ratio,
        "mid_cross_ratio": mid_cross_ratio,
    }


def _refine_confusing_digits(probs, stats):
    top3 = np.argsort(probs)[-3:]
    if not any(d in (4, 7, 9) for d in top3):
        return probs

    tuned = probs.astype(np.float64).copy()
    hole_count = stats["hole_count"]
    upper_ratio = stats["upper_ratio"]
    top_band_ratio = stats["top_band_ratio"]
    diag_ratio = stats["diag_ratio"]
    mid_cross_ratio = stats["mid_cross_ratio"]

    # 9 typically has an enclosed hole; 4/7 usually do not.
    if hole_count >= 1:
        tuned[9] *= 1.55
        tuned[7] *= 0.58
        tuned[4] *= 0.72
    else:
        tuned[9] *= 0.52

    # 7 tends to have strong top bar + descending diagonal, and weak center cross.
    if top_band_ratio > 0.22 and diag_ratio > 0.34 and mid_cross_ratio < 0.22:
        tuned[7] *= 1.30
        tuned[4] *= 0.88

    # 4 tends to keep a stronger center crossbar than 7.
    if mid_cross_ratio > 0.26 and upper_ratio < 0.78:
        tuned[4] *= 1.28
        tuned[7] *= 0.84

    s = tuned.sum()
    if s > 0:
        tuned /= s
    return tuned.astype(np.float32)


def _find_font_paths():
    roots = [
        "/System/Library/Fonts",
        "/Library/Fonts",
    ]
    paths = []
    for root in roots:
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))

    # Keep deterministic subset for reproducible training speed.
    paths = sorted(set(paths))[:120]
    return paths


def _render_font_digit(digit, rng, fonts):
    canvas = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(canvas)

    font = None
    if fonts:
        for _ in range(4):
            fp = fonts[int(rng.integers(0, len(fonts)))]
            size = int(rng.integers(46, 80))
            try:
                font = ImageFont.truetype(fp, size)
                break
            except OSError:
                font = None

    if font is None:
        font = ImageFont.load_default()

    txt = str(digit)
    stroke = int(rng.integers(0, 3))

    try:
        bbox = draw.textbbox((0, 0), txt, font=font, stroke_width=stroke)
    except OSError:
        stroke = 0
        bbox = draw.textbbox((0, 0), txt, font=font)

    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    x = (96 - tw) // 2 + int(rng.integers(-10, 11))
    y = (96 - th) // 2 + int(rng.integers(-10, 11))
    color = int(rng.integers(180, 256))

    try:
        draw.text((x, y), txt, font=font, fill=color, stroke_width=stroke, stroke_fill=color)
    except OSError:
        draw.text((x, y), txt, font=font, fill=color)

    arr = np.asarray(canvas, dtype=np.float32)
    angle = float(rng.uniform(-24, 24))
    dx = float(rng.uniform(-5.0, 5.0))
    dy = float(rng.uniform(-5.0, 5.0))
    scale = float(rng.uniform(0.85, 1.2))

    z = zoom(arr, scale, order=1)
    z = _center_crop_or_pad(z, 96)
    r = rotate(z, angle, reshape=False, order=1, mode="constant")
    m = shift(r, shift=(dy, dx), order=1, mode="constant")

    sigma = float(rng.uniform(0.2, 1.2))
    m = gaussian_filter(m, sigma=sigma)
    return np.clip(m, 0, 255)


def _jitter_point(pt, rng, scale=0.03):
    return (pt[0] + float(rng.uniform(-scale, scale)), pt[1] + float(rng.uniform(-scale, scale)))


def _norm_to_xy(pt):
    return int(pt[0] * 95), int(pt[1] * 95)


def _draw_stroke_path(draw, pts, width, rng):
    jittered = [_jitter_point(p, rng) for p in pts]
    xy = [_norm_to_xy(p) for p in jittered]
    draw.line(xy, fill=255, width=width, joint="curve")


def _render_stroke_digit(digit, rng):
    canvas = Image.new("L", (96, 96), 0)
    draw = ImageDraw.Draw(canvas)
    w = int(rng.integers(7, 13))

    if digit == 0:
        draw.ellipse((20, 14, 76, 82), outline=255, width=w)
    elif digit == 1:
        _draw_stroke_path(draw, [(0.52, 0.18), (0.52, 0.86)], w, rng)
        _draw_stroke_path(draw, [(0.40, 0.30), (0.52, 0.18)], max(4, w - 2), rng)
    elif digit == 2:
        _draw_stroke_path(draw, [(0.20, 0.28), (0.42, 0.14), (0.74, 0.24), (0.70, 0.45)], w, rng)
        _draw_stroke_path(draw, [(0.70, 0.45), (0.26, 0.84), (0.78, 0.84)], w, rng)
    elif digit == 3:
        _draw_stroke_path(draw, [(0.25, 0.22), (0.65, 0.20), (0.56, 0.48), (0.30, 0.50)], w, rng)
        _draw_stroke_path(draw, [(0.30, 0.50), (0.64, 0.50), (0.66, 0.80), (0.25, 0.82)], w, rng)
    elif digit == 4:
        _draw_stroke_path(draw, [(0.70, 0.18), (0.70, 0.86)], w, rng)
        _draw_stroke_path(draw, [(0.24, 0.56), (0.80, 0.56)], w, rng)
        _draw_stroke_path(draw, [(0.26, 0.56), (0.62, 0.18)], w, rng)
    elif digit == 5:
        _draw_stroke_path(draw, [(0.72, 0.18), (0.30, 0.18), (0.26, 0.48), (0.66, 0.48)], w, rng)
        _draw_stroke_path(draw, [(0.66, 0.48), (0.72, 0.80), (0.30, 0.82)], w, rng)
    elif digit == 6:
        _draw_stroke_path(draw, [(0.68, 0.20), (0.32, 0.40), (0.26, 0.66), (0.44, 0.82), (0.70, 0.72), (0.62, 0.52), (0.34, 0.52)], w, rng)
    elif digit == 7:
        _draw_stroke_path(draw, [(0.22, 0.20), (0.78, 0.20)], w, rng)
        _draw_stroke_path(draw, [(0.78, 0.20), (0.38, 0.84)], w, rng)
    elif digit == 8:
        draw.ellipse((26, 14, 70, 48), outline=255, width=w)
        draw.ellipse((24, 44, 72, 86), outline=255, width=w)
    elif digit == 9:
        _draw_stroke_path(draw, [(0.68, 0.58), (0.34, 0.56), (0.28, 0.30), (0.52, 0.16), (0.74, 0.28), (0.66, 0.84)], w, rng)

    arr = np.asarray(canvas, dtype=np.float32)
    angle = float(rng.uniform(-18, 18))
    dx = float(rng.uniform(-4.5, 4.5))
    dy = float(rng.uniform(-4.5, 4.5))
    scale = float(rng.uniform(0.90, 1.15))

    z = zoom(arr, scale, order=1)
    z = _center_crop_or_pad(z, 96)
    r = rotate(z, angle, reshape=False, order=1, mode="constant")
    m = shift(r, shift=(dy, dx), order=1, mode="constant")
    m = gaussian_filter(m, sigma=float(rng.uniform(0.25, 1.1)))
    return np.clip(m, 0, 255)


def _build_train_data():
    rng = np.random.default_rng(42)
    digits = load_digits()
    x = digits.images.astype(np.float32)
    y = digits.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    feats = []
    labels = []

    # 1) Build from sklearn digits with stronger augmentation.
    for img8, lbl in zip(x_train, y_train):
        base = np.uint8(np.clip((img8 / 16.0) * 255.0, 0, 255))
        base32 = np.asarray(Image.fromarray(base, mode="L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)

        feats.append(_gray_to_feature(base32))
        labels.append(lbl)

        for _ in range(4):
            angle = float(rng.uniform(-24, 24))
            dx = float(rng.uniform(-3.0, 3.0))
            dy = float(rng.uniform(-3.0, 3.0))
            scale = float(rng.uniform(0.8, 1.25))

            z = zoom(base32, scale, order=1)
            z = _center_crop_or_pad(z, 32)
            r = rotate(z, angle, reshape=False, order=1, mode="constant")
            m = shift(r, shift=(dy, dx), order=1, mode="constant")
            m = gaussian_filter(m, sigma=float(rng.uniform(0.1, 1.0)))

            feats.append(_gray_to_feature(np.clip(m, 0, 255)))
            labels.append(lbl)

    # 2) Add synthetic font digits.
    fonts = _find_font_paths()
    print(f"[INFO] font count: {len(fonts)}")

    for digit in range(10):
        count = 420 if digit in (2, 4, 7, 9) else 260
        for _ in range(count):
            synth = _render_font_digit(digit, rng, fonts)
            feats.append(_gray_to_feature(synth))
            labels.append(digit)

    # 3) Add handwriting-like stroke samples (extra for confusing classes).
    for digit in range(10):
        count = 640 if digit in (2, 4, 7, 9) else 300
        for _ in range(count):
            synth = _render_stroke_digit(digit, rng)
            feats.append(_gray_to_feature(synth))
            labels.append(digit)

    train_x = np.asarray(feats, dtype=np.float32)
    train_y = np.asarray(labels, dtype=np.int64)

    eval_x = []
    for img8 in x_test:
        base = np.uint8(np.clip((img8 / 16.0) * 255.0, 0, 255))
        base32 = np.asarray(Image.fromarray(base, mode="L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        eval_x.append(_gray_to_feature(base32))

    eval_x = np.asarray(eval_x, dtype=np.float32)
    eval_y = y_test
    return train_x, train_y, eval_x, eval_y


def train_or_load_model():
    if os.path.exists(MODEL_PATH):
        payload = joblib.load(MODEL_PATH)
        if isinstance(payload, dict) and payload.get("version") == MODEL_VERSION:
            return payload["model"]

    print("[INFO] building dataset...")
    train_x, train_y, eval_x, eval_y = _build_train_data()
    print(f"[INFO] train samples: {len(train_x)}")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(384, 192),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.0007,
                    alpha=2e-4,
                    batch_size=128,
                    max_iter=520,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)

    pred = model.predict(eval_x)
    acc = float((pred == eval_y).mean())
    print(f"[INFO] eval accuracy: {acc:.4f}")

    # Quick per-class signal to verify class 7 improved.
    report = classification_report(eval_y, pred, output_dict=True)
    r7 = report.get("7", {}).get("recall", 0.0)
    print(f"[INFO] class-7 recall: {r7:.4f}")

    joblib.dump({"version": MODEL_VERSION, "model": model}, MODEL_PATH)
    return model


def surface_to_feature(surface):
    arr = pygame.surfarray.array3d(surface)
    gray = arr.mean(axis=2).T.astype(np.float32)
    canvas28 = _gray_to_canvas28(gray)
    out = Image.fromarray(np.uint8(np.clip(canvas28, 0, 255)), mode="L").resize(
        (FEATURE_SIZE, FEATURE_SIZE), Image.Resampling.LANCZOS
    )
    feat = (np.asarray(out, dtype=np.float32) / 255.0).reshape(1, -1)
    stats = _extract_shape_stats(canvas28)
    return feat, stats


def draw_button(screen, rect, label, font):
    pygame.draw.rect(screen, BTN, rect, border_radius=8)
    text = font.render(label, True, BTN_TEXT)
    screen.blit(text, text.get_rect(center=rect.center))


def main():
    pygame.init()
    pygame.display.set_caption("손글씨 숫자 인식 (0~9)")
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("AppleSDGothicNeo", 30)
    font_mid = pygame.font.SysFont("AppleSDGothicNeo", 24)
    font_small = pygame.font.SysFont("AppleSDGothicNeo", 18)

    model = train_or_load_model()

    draw_surface = pygame.Surface((DRAW_SIZE, DRAW_SIZE))
    draw_surface.fill((0, 0, 0))

    probs = np.zeros(10, dtype=np.float32)
    pred = None
    conf = 0.0

    clear_btn = pygame.Rect(40, 540, 130, 48)
    predict_btn = pygame.Rect(190, 540, 130, 48)

    drawing = False

    def do_predict():
        nonlocal probs, pred, conf
        x, stats = surface_to_feature(draw_surface)
        p = model.predict_proba(x)[0]
        p = _refine_confusing_digits(p, stats)
        probs = p.astype(np.float32)
        pred = int(np.argmax(probs))
        conf = float(np.max(probs))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                if DRAW_RECT.collidepoint(mx, my):
                    drawing = True
                elif clear_btn.collidepoint(mx, my):
                    draw_surface.fill((0, 0, 0))
                    probs = np.zeros(10, dtype=np.float32)
                    pred = None
                    conf = 0.0
                elif predict_btn.collidepoint(mx, my):
                    do_predict()

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if drawing:
                    drawing = False
                    do_predict()

            elif event.type == pygame.MOUSEMOTION and drawing:
                mx, my = event.pos
                local_x = int((mx - DRAW_RECT.x) * DRAW_SIZE / DRAW_RECT.w)
                local_y = int((my - DRAW_RECT.y) * DRAW_SIZE / DRAW_RECT.h)
                if 0 <= local_x < DRAW_SIZE and 0 <= local_y < DRAW_SIZE:
                    pygame.draw.circle(draw_surface, WHITE, (local_x, local_y), BRUSH_RADIUS)

        screen.fill(BG)

        t = font_title.render("마우스로 중앙에 숫자(0~9)를 써보세요", True, TEXT)
        screen.blit(t, (40, 28))

        pygame.draw.rect(screen, BORDER, DRAW_RECT, width=2, border_radius=8)
        scaled = pygame.transform.smoothscale(draw_surface, (DRAW_RECT.w, DRAW_RECT.h))
        screen.blit(scaled, (DRAW_RECT.x, DRAW_RECT.y))

        draw_button(screen, clear_btn, "지우기", font_small)
        draw_button(screen, predict_btn, "인식", font_small)

        pred_text = f"예측: {pred if pred is not None else '-'}"
        conf_text = f"신뢰도: {conf*100:.1f}%" if pred is not None else "신뢰도: -"
        screen.blit(font_mid.render(pred_text, True, (12, 57, 104)), (340, 545))
        screen.blit(font_small.render(conf_text, True, TEXT), (520, 545))

        pygame.draw.rect(screen, BORDER, GRAPH_RECT, width=2, border_radius=8)
        graph_title = font_mid.render("0~9 유사율(확률)", True, TEXT)
        screen.blit(graph_title, (GRAPH_RECT.x + 14, GRAPH_RECT.y - 38))

        tip = font_small.render("팁: 2는 아랫꼬리, 4는 가로획을 분명히 쓰면 더 잘 맞습니다.", True, (70, 83, 105))
        screen.blit(tip, (GRAPH_RECT.x + 14, GRAPH_RECT.y - 62))

        bar_base_y = GRAPH_RECT.y + GRAPH_RECT.h - 30
        bar_area_h = GRAPH_RECT.h - 70
        bar_w = 28
        gap = 13
        start_x = GRAPH_RECT.x + 28

        max_p = max(1.0, float(np.max(probs) + 0.1))
        hi = int(np.argmax(probs)) if pred is not None else -1

        for i in range(10):
            value = float(probs[i])
            h = int((value / max_p) * bar_area_h)
            x = start_x + i * (bar_w + gap)
            y = bar_base_y - h
            color = BAR_HI if i == hi else BAR
            pygame.draw.rect(screen, color, pygame.Rect(x, y, bar_w, h), border_radius=4)

            nlab = font_small.render(str(i), True, TEXT)
            screen.blit(nlab, (x + 8, bar_base_y + 8))

            if value > 0:
                plab = font_small.render(f"{value*100:.0f}%", True, TEXT)
                screen.blit(plab, (x - 4, y - 24))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
