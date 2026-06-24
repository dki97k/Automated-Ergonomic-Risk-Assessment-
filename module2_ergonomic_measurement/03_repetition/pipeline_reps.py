# ============================================================
# End-to-End Pipeline: JSONL → Embeddings/SSM → Rep Counting (RepNet-style)
# Single-run script: input JSONL path in, all results under ./output
# ============================================================

import os, json, math, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn

# ============================================================
# Section A. JSONL → Embeddings & SSM
# ============================================================


def preprocess_jsonl(jsonl_path: str, emb_dim: int = 128):
    """
    1) Read JSONL with 'joints' dict per frame
    2) Build (T,J,3) array
    3) Pelvis-origin & Pelvis–Neck scale normalization
    4) Pack into (B=1, C=3, T, H=4, W=4) grid per joint layout
    5) 3D Conv encoder → (T,d) embeddings
    6) SSM = -||Ea - Eb||^2
    7) Save embeddings.npy and SSM.npy under ./output
    """
    df = pd.read_json(jsonl_path, lines=True)

    # Flatten joints
    j = pd.json_normalize(df["joints"]).add_prefix("joints.")
    base = (
        df[["frame"]].copy()
        if "frame" in df.columns
        else pd.DataFrame({"frame": range(len(df))})
    )
    out = pd.concat([base, j], axis=1)
    print("out shape =", out.shape, "| columns =", len(out.columns))

    # (T,J,3) tensor
    joint_cols = [
        c for c in out.columns if c.startswith("joints.") and c.count(".") == 2
    ]
    joint_names = sorted({c.split(".")[1] for c in joint_cols})
    coords = ["x", "y", "z"]

    T = len(out)
    J = len(joint_names)
    X = np.full((T, J, 3), np.nan, dtype=np.float32)

    for j_idx, name in enumerate(joint_names):
        for k, coord in enumerate(coords):
            col = f"joints.{name}.{coord}"
            if col in out.columns:
                X[:, j_idx, k] = out[col].to_numpy(dtype=np.float32, copy=False)

    if np.isnan(X).any():
        n_nan = int(np.isnan(X).sum())
        raise ValueError(
            f"입력 좌표에 NaN이 {n_nan}개 포함되어 있습니다. 전처리(보간/드롭) 후 다시 실행하세요."
        )

    # Pelvis–Neck normalization
    def _canon(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())

    name2idx = {_canon(n): i for i, n in enumerate(joint_names)}

    def find_idx(candidates):
        for c in candidates:
            key = _canon(c)
            if key in name2idx:
                return name2idx[key]
        return None

    pelvis = find_idx(["Pelvis (Origin)", "Pelvis"])
    neck = find_idx(["Neck"])
    if pelvis is None or neck is None:
        raise ValueError(
            "필수 관절(Pelvis, Neck)을 찾지 못했습니다. 입력 관절명을 확인하세요."
        )

    Xc = X - X[:, pelvis : pelvis + 1, :]
    d = np.linalg.norm(Xc[:, pelvis, :] - Xc[:, neck, :], axis=1)
    if not np.any(d > 1e-6):
        raise ValueError("Pelvis–Neck 스케일 기준이 유효하지 않습니다.")
    s = float(np.median(d[d > 1e-6]))
    Xc /= s

    # 4×4 grid layout
    grid_spec = [
        ["Head", "Neck", "Spine", "Pelvis (Origin)"],
        ["Hip (L)", "Shoulder (L)", "Shoulder (R)", "Hip (R)"],
        ["Knee (L)", "Elbow (L)", "Elbow (R)", "Knee (R)"],
        ["Ankle (L)", "Wrist (L)", "Wrist (R)", "Ankle (R)"],
    ]
    B, H, W = 1, 4, 4

    def match_name(q):
        cq = _canon(q)
        if cq in name2idx:
            return name2idx[cq]
        for nm, i in name2idx.items():  # loose match
            if cq in nm or nm in cq:
                return i
        return None

    x5d = np.zeros((B, 3, T, H, W), dtype=np.float32)
    placed = set()
    for r in range(H):
        for c in range(W):
            idx = match_name(grid_spec[r][c])
            if idx is None:
                continue
            if idx in placed:
                raise ValueError(f"중복 배치 감지: {joint_names[idx]} at ({r},{c})")
            placed.add(idx)
            x5d[0, :, :, r, c] = Xc[:, idx, :].T  # (3,T)

    # Simple 3D encoder → (T,d)
    # [FIX 2026-06-21 재현성] Encoder3D는 미학습 random-init → seed 없으면 매 실행 SSM/count가 달라짐.
    #   결정성 위해 seed 고정. (근본해결=결정론적 임베딩으로 교체, 재설계 #D1 — 별도 사항.)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class Encoder3D(nn.Module):
        def __init__(self, in_ch=3, d=128):
            super().__init__()
            self.conv = nn.Conv3d(in_ch, d, kernel_size=3, padding=1)
            self.act = nn.ReLU(inplace=True)

        def forward(self, x):  # x: (B,3,T,H,W)
            z = self.act(self.conv(x))  # (B,d,T,H,W)
            z = z.amax(dim=-1).amax(dim=-1)  # (B,d,T)
            z = z.squeeze(0).transpose(0, 1)  # (T,d)
            return z

    enc = Encoder3D(d=emb_dim).to(device).eval()
    x_t = torch.from_numpy(x5d).to(device)

    with torch.inference_mode():
        E_t = enc(x_t).to(torch.float32)  # (T,d)
        D2 = torch.cdist(E_t, E_t, p=2) ** 2  # (T,T) Euclidean squared
        SSM_t = -D2  # SSM = -||Ea - Eb||^2

    E = E_t.detach().cpu().numpy()
    SSM = SSM_t.detach().cpu().numpy()
    print("Embeddings:", E.shape, "| SSM:", SSM.shape)

    # save under ./output
    root_dir = Path(__file__).parent
    save_dir = root_dir / "output"
    save_dir.mkdir(parents=True, exist_ok=True)

    emb_path = save_dir / "embeddings.npy"
    ssm_path = save_dir / "SSM.npy"

    np.save(emb_path, E)
    np.save(ssm_path, SSM)
    print("Saved:", emb_path, "|", ssm_path)

    # quick viz (optional)
    try:
        plt.figure(figsize=(6, 6))
        plt.imshow(SSM, aspect="equal")
        plt.title("Self-Similarity Matrix (SSM = -||Ea - Eb||^2)")
        plt.xlabel("time")
        plt.ylabel("time")
        plt.tight_layout()
        plt.savefig(save_dir / "plot_ssm.png", dpi=180)
        plt.close()
    except Exception as e:
        print("SSM plot skipped:", e)

    return str(ssm_path), str(save_dir)


# ============================================================
# Section B. RepNet HEAD-only from SSM — Segmented Integration Counting
# ============================================================

CONFIG = dict(
    SSM_PATH=None,
    CKPT_PATH=None,
    SAVE_DIR=None,
    FPS=30.0,
    DEVICE="cuda",
    QUIET=True,
    PERIOD_BINS=32,
    LAGS_USED=64,
    D_MODEL=512,
    NHEAD=4,
    POS_LEARNED=True,
    USE_ROW_SOFTMAX=True,
    SOFTMAX_TAU=13.5,
    MIN_PERIOD_SEC=0.6,
    MAX_PERIOD_SEC=None,
    K_HARMONICS=6,
    LAM_HALF=0.6,
    LAM_THIRD=0.3,
    GAMMA_SUPER=0.3,
    W_HEAD=0.5,
    HEAD_WEIGHT_LONGP=0.25,
    BAND_FRAC=1.0,
    SMOOTH_FUSED=True,
    PBinMapping="log",
    P_MIN_FRAMES=None,
    P_MAX_FRAMES=None,
    USE_HEAD_FOR_PSTAR=True,
    PSTAR_HEAD_WEIGHT=0.25,
    COUNT_MODE="peaks",
    THR_HI_PCTL=80,
    THR_LO_PCTL=60,
    MIN_SEG_LEN_FRAC=0.30,
    INTEG_WEIGHT_MODE="fused",
    INTEG_WEIGHT_POWER=1.0,
    P_FRAME_CLIP_PCTL=(1.0, 99.0),
    SAVE_PEAKS_DIAG=True,
    MIN_PEAK_SEP_SEC=1.0,
    PEAK_HEIGHT_PCTL=70,
    PEAK_DIST_FRAC=0.60,
)


class PeriodHeadCoreFFN(nn.Module):
    def __init__(self, d_in=2048, d_model=512, max_len=64, nhead=4, pos_learned=True):
        super().__init__()
        self.input_projection = nn.Linear(d_in, d_model, bias=True)
        if pos_learned:
            self.pos_encoding = nn.Parameter(
                torch.zeros(1, max_len, d_model), requires_grad=True
            )
        else:
            pe = torch.zeros(1, max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
            )
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.pos_encoding = nn.Parameter(pe, requires_grad=False)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model, batch_first=True
        )

    def forward(self, x):
        T = x.size(1)
        h = self.input_projection(x)
        L = min(T, self.pos_encoding.size(1))
        h[:, :L, :] = h[:, :L, :] + self.pos_encoding[:, :L, :]
        return self.transformer_layer(h)


class RepHeadFromSSM_ShapeMatched(nn.Module):
    def __init__(
        self, period_bins=32, lags_used=64, d_model=512, nhead=4, pos_learned=True
    ):
        super().__init__()
        self.lags_used = lags_used
        self.tsm_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True)
        )
        self.period_length_head = nn.Sequential(
            PeriodHeadCoreFFN(
                d_in=32 * lags_used,
                d_model=d_model,
                max_len=lags_used,
                nhead=nhead,
                pos_learned=pos_learned,
            ),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, period_bins),
        )
        self.periodicity_head = nn.Sequential(
            PeriodHeadCoreFFN(
                d_in=32 * lags_used,
                d_model=d_model,
                max_len=lags_used,
                nhead=nhead,
                pos_learned=pos_learned,
            ),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, ssm):
        f = self.tsm_conv[0](ssm)  # (B,32,T,T)
        B, C, T, _ = f.shape
        L = min(self.lags_used, T)
        g_part = (
            f[:, :, :, :L].permute(0, 2, 1, 3).contiguous().view(B, T, C * L)
        )  # (B,T,32*L)
        if L < self.lags_used:
            pad = torch.zeros(
                B, T, C * (self.lags_used - L), device=g_part.device, dtype=g_part.dtype
            )
            g = torch.cat([g_part, pad], dim=-1)
        else:
            g = g_part

        h1 = self.period_length_head[0](g)
        out1 = self.period_length_head[5](
            self.period_length_head[4](
                self.period_length_head[3](
                    self.period_length_head[2](self.period_length_head[1](h1))
                )
            )
        )
        h2 = self.periodicity_head[0](g)
        out2 = self.periodicity_head[5](
            self.periodicity_head[4](
                self.periodicity_head[3](
                    self.periodicity_head[2](self.periodicity_head[1](h2))
                )
            )
        )
        return out1, out2  # (B,T,P), (B,T,1)


# -----------------------------
# Utilities
# -----------------------------
def normalize_ssm_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x, np.float32) if mx - mn < 1e-12 else (x - mn) / (mx - mn)


def row_softmax_ssm(x: np.ndarray, tau=13.5) -> np.ndarray:
    x = x.astype(np.float32)
    xt = x / float(tau)
    xt -= xt.max(axis=1, keepdims=True)
    ex = np.exp(xt)
    den = ex.sum(axis=1, keepdims=True) + 1e-12
    return (ex / den).astype(np.float32)


def movavg1d(x, w):
    w = int(max(1, w))
    k = np.ones(w, np.float32) / w
    return np.convolve(x.astype(np.float32), k, mode="same")


def zero_one(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo + 1e-12) if hi > lo else np.zeros_like(x, np.float32)


def lag_curve_from_ssm(ssm_np):
    T = ssm_np.shape[0]
    if T <= 1:
        return np.zeros(1, np.float32)
    rho = np.array([np.mean(np.diag(ssm_np, k=l)) for l in range(1, T)], np.float32)
    return movavg1d(rho, max(3, len(rho) // 300))


def _harm_sum(rho, p, mult=1.0, K=6):
    Tm1 = len(rho)
    ks = np.arange(1, K + 1)
    idx = np.clip(np.round(ks * p * mult).astype(int), 1, Tm1)
    return float(rho[idx - 1].mean())


def pick_period_unbiased(
    rho,
    fps,
    min_sec=0.6,
    max_sec=None,
    K=6,
    lam_half=0.6,
    lam_third=0.3,
    gamma_super=0.3,
    return_all=False,
):
    Tm1 = len(rho)
    p_min = int(round(min_sec * fps)) if min_sec else 10
    p_max = int(round(max_sec * fps)) if (max_sec is not None) else (Tm1 // 2)
    p_min = max(10, min(p_min, Tm1 // 2))
    p_max = max(p_min + 1, p_max)
    scores, parts = [], []
    for p in range(p_min, p_max + 1):
        s_main = _harm_sum(rho, p, 1.0, K)
        s_half = _harm_sum(rho, p, 0.5, K)
        s_third = _harm_sum(rho, p, 1.0 / 3.0, K)
        s_super = 0.0
        if 2 * p <= Tm1:
            s_super += _harm_sum(rho, 2 * p, 1.0, K)
        if 3 * p <= Tm1:
            s_super += _harm_sum(rho, 3 * p, 1.0, K)
        if (2 * p <= Tm1) and (3 * p <= Tm1):
            s_super *= 0.5
        score = s_main + gamma_super * s_super - lam_half * s_half - lam_third * s_third
        scores.append(score)
        parts.append((s_main, s_half, s_third, s_super))
    scores = np.asarray(scores, np.float32)
    idx = int(np.argmax(scores))
    p_star = p_min + idx
    meta = {
        "score": float(scores[idx]),
        "s_main": float(parts[idx][0]),
        "s_half": float(parts[idx][1]),
        "s_third": float(parts[idx][2]),
        "s_super": float(parts[idx][3]),
        "K": int(K),
        "lam_half": float(lam_half),
        "lam_third": float(lam_third),
        "gamma_super": float(gamma_super),
    }
    if return_all:
        return int(p_star), meta, (np.arange(p_min, p_max + 1, dtype=int), scores)
    return int(p_star), meta


def periodicity_curve_from_ssm(ssm_np, p, band=2, max_harm=4):
    T = ssm_np.shape[0]
    curve = np.zeros(T, np.float32)
    for t in range(T):
        vals = []
        for k in range(1, max_harm + 1):
            j = t + k * p
            if j >= T:
                break
            lo, hi = max(0, j - band), min(T - 1, j + band)
            vals.append(np.mean(ssm_np[t, lo : hi + 1]))
        curve[t] = np.mean(vals) if vals else 0.0
    return movavg1d(curve, max(3, int(round(p / 2))))


def snr_proxy_of(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    p10, p90 = np.percentile(x, 10), np.percentile(x, 90)
    std = float(np.std(x) + 1e-12)
    return float((p90 - p10) / std)


def _find_peaks_nms(y, distance=1, height=None):
    y = np.asarray(y, np.float32)
    T = len(y)
    if T < 3:
        return np.array([], dtype=int)
    if height is None:
        height = float(np.percentile(y, 75))
    cand = [
        i
        for i in range(1, T - 1)
        if y[i] >= height and y[i - 1] < y[i] and y[i] >= y[i + 1]
    ]
    if not cand:
        return np.array([], dtype=int)
    cand = np.array(cand, dtype=int)
    order = np.argsort(y[cand])[::-1]
    used = np.zeros(T, dtype=bool)
    sel = []
    for idx in order:
        i = int(cand[idx])
        L = max(0, i - distance)
        R = min(T, i + distance + 1)
        if used[L:R].any():
            continue
        sel.append(i)
        used[L:R] = True
    return np.array(sorted(sel), dtype=int)


def _save_peak_plots_and_ssm(
    out_dir,
    fused,
    peaks,
    peaks_sec,
    height_thr,
    fps,
    total_duration_sec,
    ssm=None,
    gaps_sec=None,
):
    os.makedirs(out_dir, exist_ok=True)
    try:
        t = np.arange(len(fused), dtype=float) / float(fps)
        plt.figure(figsize=(13, 4))
        plt.plot(t, fused, linewidth=1.0, label="fused periodicity (norm)")
        if len(peaks) > 0:
            plt.scatter(
                peaks_sec,
                fused[peaks],
                s=28,
                marker="o",
                label=f"peaks (n={len(peaks)})",
            )
        plt.axhline(height_thr, linestyle="--", linewidth=1.0, label="height threshold")
        plt.xlabel("time (sec)")
        plt.ylabel("periodicity (norm)")
        plt.title("Fused periodicity with detected peaks")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_fused_with_peaks.png"), dpi=180)
        plt.close()
    except Exception as e:
        print("Plot fused_with_peaks failed:", e)
    try:
        plt.figure(figsize=(13, 1.8))
        plt.hlines(1.0, 0.0, total_duration_sec, linewidth=6, alpha=0.08)
        if len(peaks_sec) > 0:
            plt.scatter(peaks_sec, np.ones_like(peaks_sec), s=28, marker="o")
        plt.yticks([])
        plt.xlim(0, total_duration_sec)
        plt.xlabel("time (sec)")
        plt.title("Peak timeline")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_peak_timeline.png"), dpi=180)
        plt.close()
    except Exception as e:
        print("Plot peak_timeline failed:", e)
    try:
        if gaps_sec is not None and len(gaps_sec) > 0:
            mu = float(np.mean(gaps_sec))
            med = float(np.median(gaps_sec))
            plt.figure(figsize=(7, 4))
            plt.hist(
                gaps_sec, bins=max(6, int(np.sqrt(len(gaps_sec)))), edgecolor="black"
            )
            plt.xlabel("inter-peak interval (sec)")
            plt.ylabel("count")
            plt.title(f"Inter-peak intervals (mean={mu:.2f}s, median={med:.2f}s)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plot_interpeak_hist.png"), dpi=180)
            plt.close()
    except Exception as e:
        print("Plot interpeak_hist failed:", e)
    try:
        if ssm is not None and ssm.ndim == 2:
            plt.figure(figsize=(6, 5))
            plt.imshow(ssm, aspect="auto", origin="lower", interpolation="nearest")
            for p in np.asarray(peaks, dtype=int):
                plt.axvline(p, lw=0.6, alpha=0.7)
                plt.axhline(p, lw=0.6, alpha=0.7)
            plt.colorbar(shrink=0.8)
            plt.title(f"SSM with peak lines (n={len(peaks)})")
            plt.xlabel("time (frame)")
            plt.ylabel("time (frame)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plot_ssm_with_peaks.png"), dpi=180)
            plt.close()
    except Exception as e:
        print("Plot ssm_with_peaks failed:", e)


def _save_integration_plots(out_dir, contrib, fps, active, p_frame=None, head_per=None):
    os.makedirs(out_dir, exist_ok=True)
    try:
        t = np.arange(len(contrib)) / float(fps)
        csum = np.cumsum(np.where(active, contrib, 0.0))
        plt.figure(figsize=(13, 4))
        plt.plot(t, csum, linewidth=1.0)
        plt.xlabel("time (sec)")
        plt.ylabel("expected repetitions (cumulative)")
        plt.title("RepNet-style integrated count (cumulative)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_integrated_cumsum.png"), dpi=180)
        plt.close()
    except Exception as e:
        print("Plot integrated_cumsum failed:", e)
    try:
        if (p_frame is not None) and (head_per is not None):
            plt.figure(figsize=(13, 3))
            plt.plot(t, head_per, linewidth=0.8, label="head periodicity (sigmoid)")
            plt.twinx()
            plt.plot(t, p_frame, linewidth=0.8, label="period (frames)")
            plt.title("Head outputs (periodicity & per-frame period)")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "plot_head_outputs.png"), dpi=180)
            plt.close()
    except Exception as e:
        print("Plot head_outputs failed:", e)


def _segment_mask(fused, p_star, hi_p, lo_p, min_len_frac):
    thr_hi = float(np.percentile(fused, hi_p))
    thr_lo = float(np.percentile(fused, lo_p))
    active = np.zeros_like(fused, dtype=bool)
    on = False
    for i, v in enumerate(fused):
        if not on and v >= thr_hi:
            on = True
        elif on and v <= thr_lo:
            on = False
        active[i] = on
    min_len = max(3, int(round(min_len_frac * p_star)))
    i = 0
    while i < len(active) - 1:
        if not active[i]:
            j = i
            while j < len(active) and not active[j]:
                j += 1
            if (j - i) < min_len:
                active[i:j] = True
            i = j
        else:
            i += 1
    i = 0
    while i < len(active) - 1:
        if active[i]:
            j = i
            while j < len(active) and active[j]:
                j += 1
            if (j - i) < min_len:
                active[i:j] = False
            i = j
        else:
            i += 1
    return active, thr_hi, thr_lo


def _clip_by_percentile(x, p_lo, p_hi):
    if not np.isfinite(x).any():
        return x
    lo, hi = np.percentile(x[np.isfinite(x)], [p_lo, p_hi])
    return np.clip(x, lo, hi)


def _integration_count(head_per, p_frame, fused, active, cfg, p_min, p_max):
    l_t = p_frame.astype(np.float32)
    if not np.isfinite(l_t).any():
        l_t = np.full_like(head_per, float(max(p_min, 10)))
    l_t = np.clip(l_t, p_min, p_max)
    if cfg["P_FRAME_CLIP_PCTL"] is not None:
        plo, phi = cfg["P_FRAME_CLIP_PCTL"]
        l_t = _clip_by_percentile(l_t, plo, phi)

    if cfg["INTEG_WEIGHT_MODE"] == "fused":
        w = zero_one(fused)
    elif cfg["INTEG_WEIGHT_MODE"] == "head":
        w = zero_one(head_per)
    else:
        w = np.ones_like(head_per, dtype=np.float32)
    if cfg["INTEG_WEIGHT_POWER"] != 1.0:
        w = np.power(w, float(cfg["INTEG_WEIGHT_POWER"])).astype(np.float32)

    contrib = (head_per.astype(np.float32) * w) / (l_t + 1e-6)
    contrib[~active] = 0.0
    E = float(np.sum(contrib))
    return E, contrib, w, l_t


def run_pipeline(cfg: dict):
    os.makedirs(cfg["SAVE_DIR"], exist_ok=True)

    # 1) Load/process SSM
    ssm = np.load(cfg["SSM_PATH"])
    ssm = normalize_ssm_minmax(ssm)
    if cfg["USE_ROW_SOFTMAX"]:
        ssm = row_softmax_ssm(ssm, tau=cfg["SOFTMAX_TAU"])
    T = int(ssm.shape[0])
    total_duration_sec = float(T / cfg["FPS"])

    # 2) RepHead
    device = torch.device(
        cfg["DEVICE"]
        if torch.cuda.is_available() and str(cfg["DEVICE"]).startswith("cuda")
        else "cpu"
    )
    model = RepHeadFromSSM_ShapeMatched(
        cfg["PERIOD_BINS"],
        cfg["LAGS_USED"],
        d_model=cfg["D_MODEL"],
        nhead=cfg["NHEAD"],
        pos_learned=cfg["POS_LEARNED"],
    ).to(device)

    # Robust checkpoint load
    dropped_mismatch = []
    state = torch.load(cfg["CKPT_PATH"], map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_sd = model.state_dict()
    adapted = {}
    for k, v in state.items():
        if k in model_sd:
            tgt = model_sd[k].shape
            if (
                k.endswith("pos_encoding")
                and isinstance(v, torch.Tensor)
                and v.ndim == 3
                and v.shape != tgt
            ):
                if (
                    v.shape[0] == tgt[0]
                    and v.shape[1] == tgt[1]
                    and v.shape[2] == 1
                    and tgt[2] == cfg["D_MODEL"]
                ):
                    v = v.repeat(1, 1, cfg["D_MODEL"])
                else:
                    dropped_mismatch.append((k, tuple(v.shape), tuple(tgt)))
                    continue
            if tuple(v.shape) != tuple(tgt):
                dropped_mismatch.append((k, tuple(v.shape), tuple(tgt)))
                continue
            adapted[k] = v
    msg = model.load_state_dict(adapted, strict=False)
    ckpt_diag = {
        "missing_keys_count": int(len(msg.missing_keys)),
        "unexpected_keys_count": int(len(msg.unexpected_keys)),
        "missing_keys_preview": list(msg.missing_keys)[:15],
        "unexpected_keys_preview": list(msg.unexpected_keys)[:15],
        "adapted_keys_loaded": int(len(adapted)),
        "dropped_mismatch_count": int(len(dropped_mismatch)),
        "dropped_mismatch_preview": [
            {"key": k, "ckpt_shape": s, "model_shape": t}
            for (k, s, t) in dropped_mismatch[:15]
        ],
    }

    ssm_t = torch.from_numpy(ssm)[None, None, :, :].to(device)
    model.eval()
    with torch.inference_mode():
        period_logits, periodicity_logit = model(ssm_t)  # (1,T,P), (1,T,1)

    head_per = (
        torch.sigmoid(periodicity_logit).squeeze(0).squeeze(-1).detach().cpu().numpy()
    )
    logits = period_logits.squeeze(0).detach().cpu().numpy()  # (T,P)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True) + 1e-12

    # p_frame mapping
    rho = lag_curve_from_ssm(ssm)
    P_MIN_FRAMES = (
        int(round(max(10, (cfg["MIN_PERIOD_SEC"] or 0.6) * cfg["FPS"])))
        if cfg["P_MIN_FRAMES"] is None
        else int(cfg["P_MIN_FRAMES"])
    )
    P_MAX_FRAMES = cfg["P_MAX_FRAMES"] or int(max(20, len(rho) // 2))

    def _period_bin_centers(P, pmin, pmax, mode="log"):
        if mode == "linear":
            return np.linspace(pmin, pmax, P, dtype=np.float32)
        lo, hi = np.log(max(1.0, pmin)), np.log(max(pmin + 1, pmax))
        return np.exp(np.linspace(lo, hi, P)).astype(np.float32)

    bin_centers = _period_bin_centers(
        cfg["PERIOD_BINS"], P_MIN_FRAMES, P_MAX_FRAMES, mode=cfg["PBinMapping"]
    )
    p_frame = (probs * bin_centers[None, :]).sum(axis=1).astype(np.float32)  # (T,)

    # 3) p* + fusion
    p_star_lag, meta, _ = pick_period_unbiased(
        rho,
        cfg["FPS"],
        min_sec=cfg["MIN_PERIOD_SEC"],
        max_sec=cfg["MAX_PERIOD_SEC"],
        K=cfg["K_HARMONICS"],
        lam_half=cfg["LAM_HALF"],
        lam_third=cfg["LAM_THIRD"],
        gamma_super=cfg["GAMMA_SUPER"],
        return_all=True,
    )
    p_head_med = (
        float(np.median(p_frame[np.isfinite(p_frame)]))
        if np.isfinite(p_frame).any()
        else None
    )
    if (
        cfg["USE_HEAD_FOR_PSTAR"]
        and (p_head_med is not None)
        and (cfg["PSTAR_HEAD_WEIGHT"] > 0.0)
    ):
        p_star = int(
            round(
                (1.0 - cfg["PSTAR_HEAD_WEIGHT"]) * p_star_lag
                + cfg["PSTAR_HEAD_WEIGHT"] * p_head_med
            )
        )
        p_star = int(np.clip(p_star, P_MIN_FRAMES, P_MAX_FRAMES))
    else:
        p_star = int(p_star_lag)

    band_adapt = max(2, int(round(cfg["BAND_FRAC"] * p_star)))
    ssm_per = periodicity_curve_from_ssm(ssm, p_star, band=band_adapt, max_harm=4)
    W_HEAD_eff = cfg["W_HEAD"]
    if p_star >= 90:
        W_HEAD_eff = min(W_HEAD_eff, cfg["HEAD_WEIGHT_LONGP"])
    fused = W_HEAD_eff * zero_one(head_per) + (1.0 - W_HEAD_eff) * zero_one(ssm_per)
    if cfg.get("SMOOTH_FUSED", True):
        fused = movavg1d(fused, max(3, int(round(p_star / 2))))
    fused_snr = snr_proxy_of(fused)

    # 4) counting
    active = np.ones_like(fused, dtype=bool)
    thr_hi = thr_lo = 0.0
    if cfg["COUNT_MODE"] in ("integrate", "hybrid"):
        active, thr_hi, thr_lo = _segment_mask(
            fused,
            p_star,
            cfg["THR_HI_PCTL"],
            cfg["THR_LO_PCTL"],
            cfg["MIN_SEG_LEN_FRAC"],
        )

    integ_E = 0.0
    integ_contrib = np.zeros_like(fused, dtype=np.float32)
    integ_w = np.zeros_like(fused, dtype=np.float32)
    l_t_used = np.zeros_like(fused, dtype=np.float32)
    if cfg["COUNT_MODE"] in ("integrate", "hybrid"):
        integ_E, integ_contrib, integ_w, l_t_used = _integration_count(
            head_per=head_per,
            p_frame=p_frame,
            fused=fused,
            active=active,
            cfg=cfg,
            p_min=P_MIN_FRAMES,
            p_max=P_MAX_FRAMES,
        )
    integ_E_round = float(round(integ_E))

    # peak diagnostics
    reps_total_peaks = 0
    peak_stats = dict(
        mean_period_sec=0.0,
        period_median_sec=0.0,
        rpm_mean=0.0,
        rpm_median=0.0,
        min_gap_sec=0.0,
        max_gap_sec=0.0,
        min_sep_violations=0,
        first_peak_sec=0.0,
        last_peak_sec=0.0,
        span_sec=0.0,
        active_duration_sec_peak_span=0.0,
        active_fraction_peak_span=0.0,
    )
    peaks = np.array([], dtype=int)
    peaks_sec = np.array([], dtype=float)
    gaps_sec = np.array([], dtype=float)
    height_thr = 0.0
    min_sep_frames = 0

    if cfg["SAVE_PEAKS_DIAG"] or cfg["COUNT_MODE"] == "peaks":
        height_thr = float(np.percentile(fused, cfg["PEAK_HEIGHT_PCTL"]))
        min_sep_frames = max(
            int(round(cfg["MIN_PEAK_SEP_SEC"] * cfg["FPS"])),
            int(round(cfg["PEAK_DIST_FRAC"] * p_star)),
        )
        peaks = _find_peaks_nms(fused, distance=min_sep_frames, height=height_thr)
        peaks = np.asarray(peaks, dtype=int)
        peaks_sec = peaks / cfg["FPS"]
        reps_total_peaks = int(len(peaks))
        if len(peaks_sec) >= 2:
            gaps_sec = np.diff(peaks_sec)
            mean_period_sec = float(np.mean(gaps_sec))
            period_median_sec = float(np.median(gaps_sec))
            rpm_mean = float(60.0 / (mean_period_sec + 1e-12))
            rpm_median = float(60.0 / (period_median_sec + 1e-12))
            min_gap_sec = float(np.min(gaps_sec))
            max_gap_sec = float(np.max(gaps_sec))
            min_sep_violations = int(np.sum(gaps_sec < cfg["MIN_PEAK_SEP_SEC"]))
            first_peak_sec = float(peaks_sec[0])
            last_peak_sec = float(peaks_sec[-1])
            span_sec = float(last_peak_sec - first_peak_sec)
            active_duration_sec_peak_span = float(span_sec)
            active_fraction_peak_span = float(
                active_duration_sec_peak_span / (total_duration_sec + 1e-12)
            )
            peak_stats.update(
                dict(
                    mean_period_sec=mean_period_sec,
                    period_median_sec=period_median_sec,
                    rpm_mean=rpm_mean,
                    rpm_median=rpm_median,
                    min_gap_sec=min_gap_sec,
                    max_gap_sec=max_gap_sec,
                    min_sep_violations=min_sep_violations,
                    first_peak_sec=first_peak_sec,
                    last_peak_sec=last_peak_sec,
                    span_sec=span_sec,
                    active_duration_sec_peak_span=active_duration_sec_peak_span,
                    active_fraction_peak_span=active_fraction_peak_span,
                )
            )

    # plots
    if cfg["SAVE_PEAKS_DIAG"]:
        _save_peak_plots_and_ssm(
            out_dir=cfg["SAVE_DIR"],
            fused=fused,
            peaks=peaks,
            peaks_sec=peaks_sec,
            height_thr=height_thr,
            fps=cfg["FPS"],
            total_duration_sec=total_duration_sec,
            ssm=ssm,
            gaps_sec=gaps_sec,
        )
    _save_integration_plots(
        out_dir=cfg["SAVE_DIR"],
        contrib=integ_contrib,
        fps=cfg["FPS"],
        active=active,
        p_frame=p_frame,
        head_per=head_per,
    )

    # extra metrics
    head_p05, head_p50, head_p95 = np.percentile(head_per, [5, 50, 95])
    head_per_mean = float(np.mean(head_per))
    head_frac_gt_02 = float(np.mean(head_per > 0.2))
    head_frac_gt_05 = float(np.mean(head_per > 0.5))

    if np.isfinite(p_frame).any():
        pf_p05, pf_p50, pf_p95 = np.percentile(
            p_frame[np.isfinite(p_frame)], [5, 50, 95]
        )
        pf_cv = float(np.std(p_frame) / (np.mean(p_frame) + 1e-12))
        pf_grad = np.gradient(p_frame) * cfg["FPS"]
        p_frame_grad_med_fps = float(np.median(np.abs(pf_grad)))
    else:
        pf_p05 = pf_p50 = pf_p95 = pf_cv = p_frame_grad_med_fps = 0.0

    s_main = float(meta.get("s_main", 1e-12))
    s_half = float(meta.get("s_half", 0.0))
    s_third = float(meta.get("s_third", 0.0))
    s_super = float(meta.get("s_super", 0.0))
    harm_ratio_half = float(s_half / (s_main + 1e-12))
    harm_ratio_third = float(s_third / (s_main + 1e-12))
    harm_ratio_super = float(s_super / (s_main + 1e-12))

    def _row_entropy_stats(prob_rows):
        H = -np.sum(prob_rows * np.log(prob_rows + 1e-12), axis=1)
        return (
            float(np.mean(H)),
            float(np.percentile(H, 95)),
            float(np.mean(prob_rows.max(axis=1))),
        )

    if cfg["USE_ROW_SOFTMAX"]:
        prob_rows = ssm
    else:
        prob_rows = row_softmax_ssm(ssm, tau=cfg["SOFTMAX_TAU"])
    row_entropy_mean, row_entropy_p95, row_max_mean = _row_entropy_stats(prob_rows)

    def _stripe_coherence(ssm, p):
        T = ssm.shape[0]
        if p <= 0 or p >= T:
            return 0.0, 0.0, 0.0
        vals, base = [], []
        b = max(1, int(max(2, round(0.05 * p))))
        for t in range(T):
            j = t + p
            if j >= T:
                break
            lo, hi = max(0, j - b), min(T - 1, j + b)
            vals.append(float(np.mean(ssm[t, lo : hi + 1])))
            jb = t + p // 2
            if jb < T:
                lo2, hi2 = max(0, jb - b), min(T - 1, jb + b)
                base.append(float(np.mean(ssm[t, lo2 : hi2 + 1])))
        coh = float(np.mean(vals)) if len(vals) > 0 else 0.0
        base = float(np.mean(base)) if len(base) > 0 else 0.0
        return coh, base, float(coh - base)

    stripe_coh, stripe_base, stripe_diff = _stripe_coherence(ssm, p_star)

    if len(peaks) > 0:
        peak_amp = fused[peaks]
        peak_amp_p50 = float(np.median(peak_amp))
        peak_amp_p95 = float(np.percentile(peak_amp, 95))
    else:
        peak_amp_p50 = peak_amp_p95 = 0.0

    if len(gaps_sec) > 0:
        interpeak_cv = float(np.std(gaps_sec) / (np.mean(gaps_sec) + 1e-12))
        med_gap = float(np.median(gaps_sec))
        mad = float(np.median(np.abs(gaps_sec - med_gap)) + 1e-12)
        interpeak_outlier_frac_MAD3 = float(
            np.mean(np.abs(gaps_sec - med_gap) > 3.0 * mad)
        )
    else:
        interpeak_cv = interpeak_outlier_frac_MAD3 = 0.0

    def _peak_count_sweep(fused, p_star, cfg, pctls=(60, 70, 80)):
        min_sep_frames = max(
            int(round(cfg["MIN_PEAK_SEP_SEC"] * cfg["FPS"])),
            int(round(cfg["PEAK_DIST_FRAC"] * p_star)),
        )
        counts = []
        for q in pctls:
            thr = float(np.percentile(fused, q))
            counts.append(
                int(len(_find_peaks_nms(fused, distance=min_sep_frames, height=thr)))
            )
        return counts, int(max(counts) - min(counts))

    count_sweep, count_range_pctl_sweep = _peak_count_sweep(fused, p_star, cfg)
    peaks_per_min_whole = float(
        (reps_total_peaks if reps_total_peaks > 0 else 0)
        / (total_duration_sec / 60.0 + 1e-12)
    )

    quality_flag = "poor"
    if (
        fused_snr > 0.45
        and head_p50 > 0.2
        and harm_ratio_half < 0.6
        and count_range_pctl_sweep <= 2
    ):
        quality_flag = "good"
    elif fused_snr > 0.25 and head_p50 > 0.1 and count_range_pctl_sweep <= 4:
        quality_flag = "marginal"

    out_dir = cfg["SAVE_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    if (cfg["SAVE_PEAKS_DIAG"] or cfg["COUNT_MODE"] == "peaks") and len(peaks) > 0:
        rep_df = pd.DataFrame(
            {"rep_frame": peaks.astype(int), "rep_time_sec": peaks_sec.astype(float)}
        )
        rep_df.to_csv(
            os.path.join(out_dir, "repetitions_centers.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    with open(os.path.join(out_dir, "ckpt_load_diag.json"), "w", encoding="utf-8") as f:
        json.dump(ckpt_diag, f, ensure_ascii=False, indent=2)

    active_frames = int(np.count_nonzero(active))
    summary = dict(
        T=int(T),
        fps=float(cfg["FPS"]),
        duration_total_sec=float(total_duration_sec),
        count_mode=str(cfg["COUNT_MODE"]),
        p_star_lag=int(p_star_lag),
        p_head_median=(None if p_head_med is None else float(p_head_med)),
        refined_period_frames=int(p_star),
        refined_period_sec=float(p_star / cfg["FPS"]),
        band_used=int(band_adapt),
        W_HEAD_eff=float(W_HEAD_eff),
        fused_snr_proxy=float(fused_snr),
        thr_hi_used=float(thr_hi),
        thr_lo_used=float(thr_lo),
        THR_HI_PCTL=int(cfg["THR_HI_PCTL"]),
        THR_LO_PCTL=int(cfg["THR_LO_PCTL"]),
        active_frames=int(active_frames),
        active_duration_sec=float(active_frames / cfg["FPS"]),
        min_seg_len_frac=float(cfg["MIN_SEG_LEN_FRAC"]),
        repetitions_total_integrated=float(round(integ_E)),
        expected_reps_est=float(integ_E),
        integ_weight_mode=str(cfg["INTEG_WEIGHT_MODE"]),
        integ_weight_power=float(cfg["INTEG_WEIGHT_POWER"]),
        p_frame_min_frames=int(P_MIN_FRAMES),
        p_frame_max_frames=int(P_MAX_FRAMES),
        repetitions_total_peaks=int(reps_total_peaks),
        peak_height_pctl_used=int(cfg["PEAK_HEIGHT_PCTL"]),
        min_peak_sep_sec=float(cfg["MIN_PEAK_SEP_SEC"]),
        peak_dist_frac_used=float(cfg["PEAK_DIST_FRAC"]),
        mean_period_sec=float(peak_stats["mean_period_sec"]),
        period_median_sec=float(peak_stats["period_median_sec"]),
        rpm_mean=float(peak_stats["rpm_mean"]),
        rpm_median=float(peak_stats["rpm_median"]),
        min_gap_sec=float(peak_stats["min_gap_sec"]),
        max_gap_sec=float(peak_stats["max_gap_sec"]),
        min_sep_violations=int(peak_stats["min_sep_violations"]),
        first_peak_sec=float(peak_stats["first_peak_sec"]),
        last_peak_sec=float(peak_stats["last_peak_sec"]),
        span_sec=float(peak_stats["span_sec"]),
        active_duration_sec_peak_span=float(
            peak_stats["active_duration_sec_peak_span"]
        ),
        active_fraction_peak_span=float(peak_stats["active_fraction_peak_span"]),
        head_per_p05=float(head_p05),
        head_per_p50=float(head_p50),
        head_per_p95=float(head_p95),
        head_per_mean=float(head_per_mean),
        head_frac_gt_0_2=float(head_frac_gt_02),
        head_frac_gt_0_5=float(head_frac_gt_05),
        p_frame_p05=float(pf_p05),
        p_frame_p50=float(pf_p50),
        p_frame_p95=float(pf_p95),
        p_frame_cv=float(pf_cv),
        p_frame_grad_med_fps=float(p_frame_grad_med_fps),
        harm_ratio_half=float(harm_ratio_half),
        harm_ratio_third=float(harm_ratio_third),
        harm_ratio_super=float(harm_ratio_super),
        row_entropy_mean=float(row_entropy_mean),
        row_entropy_p95=float(row_entropy_p95),
        row_max_mean=float(row_max_mean),
        stripe_coh=float(stripe_coh),
        stripe_base=float(stripe_base),
        stripe_diff=float(stripe_diff),
        peak_amp_p50=float(peak_amp_p50),
        peak_amp_p95=float(peak_amp_p95),
        interpeak_cv=float(interpeak_cv),
        interpeak_outlier_frac_MAD3=float(interpeak_outlier_frac_MAD3),
        count_sweep_60_70_80=str(count_sweep),
        count_range_pctl_sweep=int(count_range_pctl_sweep),
        peaks_per_min_whole=float(peaks_per_min_whole),
        quality_flag=str(quality_flag),
        ckpt_missing_keys_count=int(ckpt_diag["missing_keys_count"]),
        ckpt_unexpected_keys_count=int(ckpt_diag["unexpected_keys_count"]),
        ckpt_dropped_mismatch_count=int(ckpt_diag["dropped_mismatch_count"]),
    )
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, "summary_reps_integrated.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(
        json.dumps(
            {
                "count_mode": cfg["COUNT_MODE"],
                "repetitions_total_integrated": round(float(round(integ_E)), 3),
                "expected_reps_est": round(float(integ_E), 6),
                "refined_period_sec": round(float(p_star / cfg["FPS"]), 6),
                "active_duration_sec": round(float(active_frames / cfg["FPS"]), 6),
                "fused_snr_proxy": round(float(fused_snr), 4),
                "head_per_p50": round(float(head_p50), 4),
                "p_frame_p50": round(float(pf_p50), 3),
                "harm_ratio_half": round(float(harm_ratio_half), 3),
                "row_entropy_mean": round(float(row_entropy_mean), 3),
                "peaks_diag": {
                    "repetitions_total_peaks": int(reps_total_peaks),
                    "mean_period_sec": round(float(peak_stats["mean_period_sec"]), 6),
                    "rpm_mean": round(float(peak_stats["rpm_mean"]), 4),
                },
                "quality_flag": quality_flag,
                "ckpt_load_diag": {
                    "missing_keys_count": int(ckpt_diag["missing_keys_count"]),
                    "unexpected_keys_count": int(ckpt_diag["unexpected_keys_count"]),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# Section C. CLI Entry
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end: JSONL → Embeddings/SSM → Rep Counting"
    )
    parser.add_argument("--jsonl", type=str, required=True, help="입력 JSONL 파일 경로")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="RepNet checkpoint .pth 경로"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0, help="입력 시퀀스 FPS (default: 30)"
    )
    parser.add_argument("--cpu", action="store_true", help="강제 CPU 모드")
    parser.add_argument(
        "--emb_dim", type=int, default=128, help="임베딩 차원 (Encoder3D)"
    )
    args = parser.parse_args()

    # Stage A: JSONL → SSM (save under ./output)
    ssm_path, save_dir = preprocess_jsonl(args.jsonl, emb_dim=args.emb_dim)

    # Stage B: Counting
    cfg = CONFIG.copy()
    cfg["SSM_PATH"] = ssm_path
    cfg["CKPT_PATH"] = args.ckpt
    cfg["SAVE_DIR"] = save_dir  # ./output
    cfg["FPS"] = float(args.fps)
    if args.cpu:
        cfg["DEVICE"] = "cpu"

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
