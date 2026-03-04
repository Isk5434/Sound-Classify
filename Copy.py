# !pip -q install gradio fastapi uvicorn scikit-learn pyngrok scikit-learn qrcode[pil]
# =========================
# Colab-ready single script
# - Runs FastAPI + Gradio mounted app on Colab
# - Uses sid in JSON payload (cookie may be unreliable in Colab/iframes)
# =========================

# (1) Install deps (Colab only)
import sys, os, threading, time
if "google.colab" in sys.modules:
    pass
    # !pip -q install gradio fastapi uvicorn scikit-learn

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # unused ok

import gradio as gr
from sklearn.model_selection import train_test_split

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# =========================
# Config
# =========================
@dataclass
class PreprocConfig:
    fs_target: float = 50.0
    hp_alpha: float = 0.92
    window_sec: float = 1.0
    hop_sec: float = 0.2

CFG = PreprocConfig()
DEFAULT_LABELS = ["idle", "shake", "flip"]


# =========================
# Utilities
# =========================
class GravityHighPass:
    """Estimate gravity via EMA and subtract it: a_dyn = a - g_est"""
    def __init__(self, alpha=0.92):
        self.alpha = float(alpha)
        self.g = np.zeros(3, dtype=np.float32)
        self.inited = False

    def reset(self):
        self.g[:] = 0
        self.inited = False

    def step(self, a_xyz: np.ndarray) -> np.ndarray:
        a = a_xyz.astype(np.float32)
        if not self.inited:
            self.g = a.copy()
            self.inited = True
        self.g = self.alpha * self.g + (1 - self.alpha) * a
        return a - self.g


def resample_linear(ts: np.ndarray, X: np.ndarray, fs_target: float):
    """Resample irregular timestamps to uniform grid using linear interpolation."""
    if len(ts) < 2:
        return ts, X
    t0, t1 = float(ts[0]), float(ts[-1])
    dt = 1.0 / fs_target
    t_new = np.arange(t0, t1, dt, dtype=np.float32)
    if len(t_new) < 2:
        return ts, X
    X_new = np.zeros((len(t_new), X.shape[1]), dtype=np.float32)
    for d in range(X.shape[1]):
        X_new[:, d] = np.interp(t_new, ts, X[:, d])
    return t_new, X_new


# =========================
# ESN Classifier (minimal)
# =========================
class ESNClassifier:
    """ESN state features + ridge multi-class regression."""
    def __init__(self, in_dim: int, res_size: int, spectral_radius: float, leak: float, ridge: float, seed: int = 0):
        self.in_dim = in_dim
        self.res_size = res_size
        self.spectral_radius = float(spectral_radius)
        self.leak = float(leak)
        self.ridge = float(ridge)
        self.seed = int(seed)

        rng = np.random.default_rng(self.seed)
        self.Win = (rng.uniform(-1, 1, size=(res_size, in_dim + 1)) * 0.5).astype(np.float32)

        W = rng.uniform(-1, 1, size=(res_size, res_size)).astype(np.float32)
        v = rng.normal(size=(res_size,)).astype(np.float32)
        for _ in range(30):
            v = W @ v
            v = v / (np.linalg.norm(v) + 1e-9)
        eig_approx = float(np.linalg.norm(W @ v) / (np.linalg.norm(v) + 1e-9))
        W *= (self.spectral_radius / (eig_approx + 1e-9))
        self.W = W

        self.x = np.zeros((res_size,), dtype=np.float32)
        self.Wout = None
        self.class_names: List[str] = []

    def reset(self):
        self.x[:] = 0

    def step(self, u: np.ndarray):
        u = u.astype(np.float32)
        aug = np.concatenate([np.array([1.0], np.float32), u], axis=0)
        pre = self.W @ self.x + self.Win @ aug
        x_new = np.tanh(pre)
        self.x = (1 - self.leak) * self.x + self.leak * x_new
        return self.x

    def fit(self, X_feat: np.ndarray, y: np.ndarray, class_names: List[str]):
        self.class_names = class_names
        n, f = X_feat.shape
        k = len(class_names)
        Y = np.zeros((n, k), dtype=np.float32)
        Y[np.arange(n), y] = 1.0

        XtX = X_feat.T @ X_feat
        I = np.eye(f, dtype=np.float32)
        self.Wout = np.linalg.solve(XtX + self.ridge * I, X_feat.T @ Y).astype(np.float32)

    def predict_proba(self, feat: np.ndarray):
        logits = feat.astype(np.float32) @ self.Wout
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        return ex / (float(np.sum(ex)) + 1e-9)


def make_window_feature(esn: ESNClassifier, X_seq: np.ndarray, mode: str = "last"):
    esn.reset()
    states = []
    for t in range(len(X_seq)):
        st = esn.step(X_seq[t])
        if mode == "mean":
            states.append(st.copy())
    if mode == "mean" and len(states) > 0:
        s = np.mean(np.stack(states, axis=0), axis=0)
    else:
        s = esn.x.copy()

    u_mean = X_seq.mean(axis=0)
    u_std = X_seq.std(axis=0)
    feat = np.concatenate([np.array([1.0], np.float32), u_mean, u_std, s], axis=0)
    return feat


# =========================
# Per-session state
# =========================
@dataclass
class SessionState:
    stream_t: List[float] = field(default_factory=list)
    stream_a: List[List[float]] = field(default_factory=list)

    collecting: bool = False
    collect_label: str = ""
    collect_tmp_t: List[float] = field(default_factory=list)
    collect_tmp_a: List[List[float]] = field(default_factory=list)

    data: Dict[str, List[Dict[str, np.ndarray]]] = field(default_factory=dict)

    trained: bool = False
    train_cfg: Dict = field(default_factory=dict)
    pp_mean: Optional[np.ndarray] = None
    pp_std: Optional[np.ndarray] = None
    esn_model: Optional[ESNClassifier] = None

    infer_running: bool = False
    infer_last_label: str = ""
    infer_last_conf: float = 0.0
    infer_pred_log: List[Tuple[float, str, float]] = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)


SESS: Dict[str, SessionState] = {}
SESS_LOCK = threading.Lock()


def _get_sid_from_request(request: Optional[gr.Request]) -> str:
    if request is None:
        return "unknown"
    try:
        sid = request.cookies.get("sid", "") if request.cookies else ""
        return sid or "unknown"
    except Exception:
        return "unknown"


def get_state(sid: str) -> SessionState:
    with SESS_LOCK:
        st = SESS.get(sid)
        if st is None:
            st = SessionState()
            SESS[sid] = st
        return st


def reset_state(st: SessionState):
    st.stream_t = []
    st.stream_a = []
    st.collecting = False
    st.collect_label = ""
    st.collect_tmp_t = []
    st.collect_tmp_a = []
    st.data = {}

    st.trained = False
    st.train_cfg = {}
    st.pp_mean = None
    st.pp_std = None
    st.esn_model = None

    st.infer_running = False
    st.infer_last_label = ""
    st.infer_last_conf = 0.0
    st.infer_pred_log = []


def counts_dict_for(st: SessionState):
    c = {k: len(v) for k, v in st.data.items()}
    c["TOTAL"] = int(sum(c.values()))
    return c


def ui_status_for(st: SessionState):
    if not st.trained:
        return "<span style='font-size:18px;font-weight:700;color:#b00020'>MODEL: not trained</span>"
    return (f"<span style='font-size:18px;font-weight:700;color:#0b6b0b'>MODEL: trained</span> "
            f"<span style='font-size:12px;opacity:.85'>val_acc={st.train_cfg.get('val_acc',0):.3f}, "
            f"classes={st.train_cfg.get('classes',[])}</span>")


def format_pred_log_md(st: SessionState, max_rows: int = 80):
    if len(st.infer_pred_log) == 0:
        return "(log empty)"
    rows = st.infer_pred_log[-max_rows:]
    md = ["| time(s) | label | conf |", "|---:|:---|---:|"]
    for t, lab, conf in rows:
        md.append(f"| {t:6.2f} | {lab} | {conf:.2f} |")
    return "\n".join(md)


# =========================
# FastAPI endpoints
# - prefer cookie sid; if missing, use payload sid
# =========================
api = FastAPI()

def _sid_from_fastapi(request: Request, payload: dict) -> str:
    try:
        sid = request.cookies.get("sid", "") or ""
    except Exception:
        sid = ""
    if sid:
        return sid
    sid2 = str(payload.get("sid", "") or "")
    return sid2 if sid2 else "unknown"


@api.post("/api/ingest")
async def ingest(request: Request):
    try:
        obj = await request.json()
        sid = _sid_from_fastapi(request, obj)
        samples = obj.get("samples", [])

        st = get_state(sid)
        with st.lock:
            for s in samples:
                t = float(s.get("t", 0.0))
                ax = float(s.get("ax", 0.0)); ay = float(s.get("ay", 0.0)); az = float(s.get("az", 0.0))
                st.stream_t.append(t)
                st.stream_a.append([ax, ay, az])
                if st.collecting:
                    st.collect_tmp_t.append(t)
                    st.collect_tmp_a.append([ax, ay, az])

            if len(st.stream_t) > 6000:
                st.stream_t = st.stream_t[-6000:]
                st.stream_a = st.stream_a[-6000:]

        return JSONResponse({"ok": True, "n": len(samples), "sid": sid, "stream_len": len(st.stream_t)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@api.post("/api/reset")
async def reset_endpoint(request: Request):
    try:
        obj = await request.json()
        sid = _sid_from_fastapi(request, obj)
        st = get_state(sid)
        with st.lock:
            reset_state(st)
        return JSONResponse({"ok": True, "sid": sid})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


# =========================
# Collect handlers
# =========================
def collect_start(label: str, request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)
    with st.lock:
        st.collecting = True
        st.collect_label = label
        st.collect_tmp_t = []
        st.collect_tmp_a = []
    return f"収集中: {label}", gr.update(interactive=False), gr.update(interactive=True)

def collect_stop_and_save(request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)

    with st.lock:
        st.collecting = False

        if len(st.collect_tmp_t) < 10:
            return ("収集停止（データが少なすぎるため未保存）",
                    gr.update(interactive=True), gr.update(interactive=False),
                    counts_dict_for(st))

        ts = np.array(st.collect_tmp_t, dtype=np.float32)
        A = np.array(st.collect_tmp_a, dtype=np.float32)
        lab = st.collect_label
        st.data.setdefault(lab, []).append({"t": ts, "a": A})

        msg = f"保存: label={lab}, samples={len(ts)}, total={len(st.data[lab])}"
        return (msg, gr.update(interactive=True), gr.update(interactive=False),
                counts_dict_for(st))


# =========================
# Training
# =========================
def build_dataset(st: SessionState, window_sec: float, hop_sec: float, fs_target: float):
    class_names = sorted([k for k in st.data.keys() if k.strip()])
    if len(class_names) < 2:
        return [], np.zeros((0,), np.int64), class_names

    seqs = []
    ys = []
    for lab_idx, lab in enumerate(class_names):
        for item in st.data[lab]:
            ts = item["t"].astype(np.float32)
            A = item["a"].astype(np.float32)

            ts2, A2 = resample_linear(ts, A, fs_target)

            hp = GravityHighPass(alpha=CFG.hp_alpha)
            Ad = np.stack([hp.step(A2[i]) for i in range(len(A2))], axis=0)

            an = np.linalg.norm(Ad, axis=1, keepdims=True)
            X = np.concatenate([Ad, an], axis=1)  # (T, 4)

            win = int(round(window_sec * fs_target))
            hop = int(round(hop_sec * fs_target))
            if len(X) < win:
                continue

            for s in range(0, len(X) - win + 1, hop):
                seqs.append(X[s:s + win])
                ys.append(lab_idx)

    return seqs, np.array(ys, dtype=np.int64), class_names


def train_click(window_sec, hop_sec, fs_target, feat_mode, request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)

    window_sec = float(window_sec); hop_sec = float(hop_sec); fs_target = float(fs_target)

    with st.lock:
        seqs, y, class_names = build_dataset(st, window_sec, hop_sec, fs_target)

    if len(seqs) < 12 or len(class_names) < 2:
        return ui_status_for(st), {}, "データ不足（2ラベル以上、各数回〜推奨）", ""

    idx = np.arange(len(seqs))
    try:
        tr_idx, va_idx = train_test_split(idx, test_size=0.25, random_state=0, stratify=y)
    except Exception:
        tr_idx, va_idx = train_test_split(idx, test_size=0.25, random_state=0)

    Xtr_all = np.concatenate([seqs[i] for i in tr_idx], axis=0)
    mu = Xtr_all.mean(axis=0, keepdims=True).astype(np.float32)
    sd = (Xtr_all.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    def norm_seq(seg):
        return (seg - mu) / sd

    cand_res = [80, 120]
    cand_sr = [0.8, 1.0]
    cand_leak = [0.2, 0.5, 0.8]
    cand_ridge = [1e-3]

    best_acc = -1.0
    best_pack = None
    logs = []

    for res_size in cand_res:
        for sr in cand_sr:
            for leak in cand_leak:
                for ridge in cand_ridge:
                    esn = ESNClassifier(
                        in_dim=4,
                        res_size=int(res_size),
                        spectral_radius=float(sr),
                        leak=float(leak),
                        ridge=float(ridge),
                        seed=0
                    )

                    Xtr = []
                    for i in tr_idx:
                        feat = make_window_feature(esn, norm_seq(seqs[i]), mode=feat_mode)
                        Xtr.append(feat)
                    Xtr = np.stack(Xtr, axis=0).astype(np.float32)

                    esn.fit(Xtr, y[tr_idx], class_names)

                    correct = 0
                    for i in va_idx:
                        feat = make_window_feature(esn, norm_seq(seqs[i]), mode=feat_mode)
                        p = esn.predict_proba(feat)
                        pred = int(np.argmax(p))
                        correct += (pred == int(y[i]))
                    acc = correct / max(1, len(va_idx))

                    logs.append(f"res={res_size}, sr={sr}, leak={leak}, ridge={ridge} -> val_acc={acc:.3f}")

                    if acc > best_acc:
                        best_acc = acc
                        best_pack = (int(res_size), float(sr), float(leak), float(ridge), esn)

    res_size, sr, leak, ridge, esn = best_pack

    with st.lock:
        st.trained = True
        st.pp_mean, st.pp_std = mu, sd
        st.esn_model = esn
        st.train_cfg = {
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "fs_target": fs_target,
            "mode": feat_mode,
            "res_size": res_size,
            "spectral_radius": sr,
            "leak": leak,
            "ridge": ridge,
            "val_acc": float(best_acc),
            "classes": class_names,
        }

    tail = "\n".join(logs[-12:])
    return ui_status_for(st), st.train_cfg, f"学習完了: val_acc={best_acc:.3f}", tail


# =========================
# Inference
# =========================
def infer_step(st: SessionState):
    if (not st.trained) or (st.esn_model is None) or (st.pp_mean is None) or (st.pp_std is None):
        return "(not trained)", 0.0, {}

    fs = float(st.train_cfg["fs_target"])
    win = int(round(float(st.train_cfg["window_sec"]) * fs))
    if len(st.stream_t) < win + 2:
        return "(buffering)", 0.0, {}

    ts = np.array(st.stream_t, dtype=np.float32)
    A = np.array(st.stream_a, dtype=np.float32)

    t_end = ts[-1]
    t_start = max(ts[0], t_end - (float(st.train_cfg["window_sec"]) + 0.4))
    m = ts >= t_start
    ts2, A2 = resample_linear(ts[m], A[m], fs)
    if len(ts2) < win:
        return "(buffering)", 0.0, {}
    A2 = A2[-win:]

    hp = GravityHighPass(alpha=CFG.hp_alpha)
    Ad = np.stack([hp.step(A2[i]) for i in range(len(A2))], axis=0)
    an = np.linalg.norm(Ad, axis=1, keepdims=True)
    X = np.concatenate([Ad, an], axis=1).astype(np.float32)

    Xn = (X - st.pp_mean) / st.pp_std
    feat = make_window_feature(st.esn_model, Xn, mode=st.train_cfg["mode"])
    p = st.esn_model.predict_proba(feat)

    i = int(np.argmax(p))
    conf = float(p[i])
    lab = st.train_cfg["classes"][i]

    prev_lab = st.infer_last_label
    st.infer_last_label = lab
    st.infer_last_conf = conf
    if (lab != prev_lab) and (lab not in ["(buffering)", "(not trained)"]):
        st.infer_pred_log.append((float(t_end), lab, conf))
        st.infer_pred_log = st.infer_pred_log[-500:]

    info = {"probs": {st.train_cfg["classes"][j]: float(p[j]) for j in range(len(p))}}
    return lab, conf, info


def infer_start(request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)
    with st.lock:
        if not st.trained:
            return "学習してから推論してください", "<div style='font-size:24px;font-weight:800;opacity:.6'>-</div>", {}, ui_status_for(st)
        st.infer_running = True
        st.infer_pred_log = []
        st.infer_last_label = ""
        st.infer_last_conf = 0.0
    return "推論: ON", "<div style='font-size:24px;font-weight:800;opacity:.6'>-</div>", {}, ui_status_for(st)


def infer_stop(request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)
    with st.lock:
        st.infer_running = False
    return "推論: OFF", "<div style='font-size:24px;font-weight:800;opacity:.6'>-</div>", {}, ui_status_for(st)


def infer_tick(request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)

    with st.lock:
        if not st.infer_running:
            return gr.update(), gr.update(), gr.update()
        lab, conf, info = infer_step(st)
        stline = ui_status_for(st)

    pred_html = (
        f"<div style='padding:10px 12px;border:1px solid #ddd;border-radius:12px;background:#fff'>"
        f"<div style='font-size:30px;font-weight:900;line-height:1.1'>{lab}</div>"
        f"<div style='font-size:13px;opacity:.85'>conf={conf:.2f}</div>"
        f"</div>"
    )
    probs = info.get("probs", {})
    return pred_html, probs, stline


def chat_tick(request: gr.Request):
    sid = _get_sid_from_request(request)
    st = get_state(sid)
    with st.lock:
        if not st.infer_running:
            big = "<div style='font-size:22px;font-weight:800;opacity:.6'>推論がOFFです</div>"
            log_md = format_pred_log_md(st)
            return big, log_md
        big = (
            f"<div style='padding:12px 14px;border:1px solid #ddd;border-radius:14px;background:#fff'>"
            f"<div style='font-size:34px;font-weight:900;line-height:1.05'>{st.infer_last_label or '-'}</div>"
            f"<div style='font-size:14px;opacity:.85'>conf={st.infer_last_conf:.2f}</div>"
            f"</div>"
        )
        log_md = format_pred_log_md(st)
        return big, log_md


# =========================
# JS UI + Boot
# - sid is stored in localStorage and sent in JSON payload
# =========================

# ── UI変更: SENSOR_UI をフェミニンデザインに全面リデザイン ──
# パステル背景・中央寄せ・やわらかい角丸ボタン・透明感・余白多め
SENSOR_UI = r"""
<div class="sensor-hero">
  <div class="sensor-hero__icon">&#9752;</div>
  <div class="sensor-hero__title">Motion Sensor</div>
  <div class="sensor-hero__subtitle">スマホを振って、動きを学習させよう</div>
  <div class="sensor-hero__buttons">
    <button id="btn_perm" class="hero-btn hero-btn--outline" type="button">PERMISSION</button>
    <button id="btn_start" class="hero-btn hero-btn--dark" type="button">START</button>
    <button id="btn_stop" class="hero-btn hero-btn--outline" type="button">STOP</button>
    <button id="btn_reset" class="hero-btn hero-btn--dark" type="button">RESET</button>
  </div>
  <div class="sensor-hero__status">
    <span id="sensor_status">status: idle</span>
  </div>
  <div class="sensor-hero__share">
    URL: <span id="share_url" class="mono"></span>
  </div>
</div>
"""

JS_BOOT = r"""
() => {
  const setStatus = (s) => {
    const el = document.getElementById('sensor_status');
    if (el) el.textContent = 'status: ' + s;
  };

  const setShareUrl = () => {
    const el = document.getElementById('share_url');
    if (!el) return;
    el.textContent = window.location.href;
  };

  const genSid = () => {
    if (crypto && crypto.randomUUID) return crypto.randomUUID();
    const r = () => Math.floor(Math.random() * 1e9).toString(16);
    return `${Date.now().toString(16)}-${r()}-${r()}-${r()}`;
  };

  const getSid = () => {
    try{
      const k = "sid_v1";
      let sid = localStorage.getItem(k);
      if(!sid){ sid = genSid(); localStorage.setItem(k, sid); }
      return sid;
    }catch(e){
      return genSid();
    }
  };

  // Gradio root (subpath/iframeでも壊れにくい)
  const apiUrl = (path) => {
    const root = (window.gradio_config && window.gradio_config.root) ? window.gradio_config.root : '';
    return `${root}${path}`;
  };

  const sid = getSid();

  let running=false, buf=[], t0=null, timer=null;
  let accel=null, dmHandler=null;

  const pushSample = (ax,ay,az) => {
    if(t0===null) t0=performance.now();
    const t=(performance.now()-t0)/1000.0;
    buf.push({t, ax:ax||0, ay:ay||0, az:az||0});
    if(buf.length>600) buf=buf.slice(-600);
  };

  const postJson = async (path, payloadObj) => {
    try{
      payloadObj = payloadObj || {};
      payloadObj.sid = sid; // <-- include sid in body
      const res = await fetch(apiUrl(path), {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payloadObj),
        credentials: 'include'
      });
      if(!res.ok){
        setStatus(`ERR: ${path} http ${res.status}`);
        return false;
      }
      return true;
    }catch(e){
      setStatus(`ERR: fetch(${path}) failed`);
      return false;
    }
  };

  const flush = async () => {
    if(!running || buf.length===0) return;
    const samples = buf;
    buf = [];
    await postJson('/api/ingest', {samples});
  };

  const startDeviceMotion = () => {
    dmHandler = (e) => {
      if(!running) return;
      const acc = e.accelerationIncludingGravity || e.acceleration;
      if(!acc) return;
      pushSample(acc.x, acc.y, acc.z);
    };
    window.addEventListener('devicemotion', dmHandler, {passive:true});
  };

  const startGeneric = () => {
    if(!('Accelerometer' in window)) return false;
    try{
      accel = new Accelerometer({frequency: 50});
      accel.addEventListener('reading', ()=>{ if(running) pushSample(accel.x, accel.y, accel.z); }, {passive:true});
      accel.addEventListener('error', ()=>{ try{accel.stop();}catch(e){} accel=null; startDeviceMotion(); });
      accel.start();
      return true;
    }catch(e){
      accel=null;
      return false;
    }
  };

  const requestPerm = async () => {
    try{
      if(typeof DeviceMotionEvent!=='undefined' && typeof DeviceMotionEvent.requestPermission==='function'){
        const res = await DeviceMotionEvent.requestPermission();
        setStatus('permission: '+res + ' / sid=' + sid.slice(0,8));
      } else {
        setStatus('permission: not-needed / sid=' + sid.slice(0,8));
      }
    }catch(e){
      setStatus('permission error');
    }
  };

  const start = () => {
    if(running) return;
    running=true; buf=[]; t0=null;
    if(!startGeneric()) startDeviceMotion();
    timer=setInterval(flush, 200);
    setStatus('running / sid=' + sid.slice(0,8));
  };

  const stop = () => {
    if(!running) return;
    running=false;
    if(accel){ try{accel.stop();}catch(e){} accel=null; }
    if(dmHandler){ window.removeEventListener('devicemotion', dmHandler); dmHandler=null; }
    if(timer){ clearInterval(timer); timer=null; }
    flush();
    setStatus('stopped / sid=' + sid.slice(0,8));
  };

  const reset = async () => {
    stop();
    await postJson('/api/reset', {});
    setStatus('reset done / sid=' + sid.slice(0,8));
  };

  const bind = () => {
    const p=document.getElementById('btn_perm');
    const s=document.getElementById('btn_start');
    const x=document.getElementById('btn_stop');
    const r=document.getElementById('btn_reset');
    if(!p || !s || !x || !r){ setTimeout(bind, 300); return; }
    p.onclick=requestPerm;
    s.onclick=start;
    x.onclick=stop;
    r.onclick=reset;
    setStatus('ready / sid=' + sid.slice(0,8));
    setShareUrl();
  };
  bind();
}
"""

# ── UI変更: CSS をフェミニン・パステルデザインに全面リデザイン ──
# くすみピンク / ミント / ラベンダー / 低彩度 / 黒不使用 / 透明感 / 余白多め
CSS = """
/* ========================================
   グローバル: フェミニン・パステルテーマ
   ======================================== */
html {
    scroll-behavior: smooth !important;
    -webkit-overflow-scrolling: touch !important;
}

/* Gradio コンテナ: 淡いグラデーション背景 */
.gradio-container {
    background: linear-gradient(175deg, #fdf2f8 0%, #faf5ff 35%, #f0fdf4 70%, #fdf2f8 100%) !important;
    color: #1a1a1a !important;
    font-family: 'Inter', 'Hiragino Kaku Gothic ProN', 'Noto Sans JP', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 400 !important;
    max-width: 100% !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* フッター非表示 */
footer { display: none !important; }

/* ========================================
   センサーヒーローセクション
   ======================================== */
.sensor-hero {
    min-height: 65vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 56px 24px 48px 24px;
    background: linear-gradient(170deg,
        rgba(253,242,248,0.9) 0%,
        rgba(250,245,255,0.85) 40%,
        rgba(240,253,244,0.8) 100%);
    margin-bottom: 8px;
}

.sensor-hero__icon {
    font-size: 36px;
    margin-bottom: 16px;
    opacity: 0.6;
    filter: grayscale(30%);
}

.sensor-hero__title {
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', serif;
    font-size: clamp(30px, 8vw, 48px);
    font-weight: 400;
    letter-spacing: 0.08em;
    color: #1a1a1a;
    margin-bottom: 10px;
    line-height: 1.15;
    text-transform: uppercase;
}

.sensor-hero__subtitle {
    font-size: clamp(13px, 3.2vw, 16px);
    font-weight: 400;
    color: #333333;
    margin-bottom: 44px;
    letter-spacing: 0.03em;
    line-height: 1.6;
}

.sensor-hero__buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-bottom: 36px;
    max-width: 360px;
}

.sensor-hero__status {
    font-size: 12px;
    color: #333333;
    font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace;
    margin-bottom: 6px;
    letter-spacing: 0.02em;
}

.sensor-hero__share {
    font-size: 11px;
    color: #333333;
    word-break: break-all;
    max-width: 85vw;
}
.sensor-hero__share .mono {
    font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace;
}

/* ========================================
   ヒーローボタン: VIEW MORE風 / セリフ体 / シャープ
   ======================================== */
.hero-btn {
    flex: 1 1 calc(50% - 5px);
    box-sizing: border-box;
    padding: 15px 10px;
    border-radius: 0;
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', 'YuMincho', serif;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    touch-action: manipulation;
    -webkit-tap-highlight-color: transparent;
}

/* 白背景 + 細線ボーダー（左のボタン） */
.hero-btn--outline {
    background: #ffffff;
    color: #555555;
    border: 1px solid #aaaaaa;
}
.hero-btn--outline:active {
    background: #f5f5f5;
    border-color: #888888;
}

/* ダーク背景（右のボタン） */
.hero-btn--dark {
    background: #3a3a3a;
    color: #d8d8d8;
    border: 1px solid #3a3a3a;
}
.hero-btn--dark:active {
    background: #4a4a4a;
}

/* ========================================
   タブナビゲーション: ピル型パステル
   ======================================== */
div.tab-nav {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(0,0,0,0.06) !important;
    border-radius: 22px !important;
    padding: 5px !important;
    margin: 16px 16px 20px 16px !important;
    display: flex !important;
    justify-content: center !important;
    gap: 3px !important;
    box-shadow: 0 2px 12px rgba(107,91,123,0.06) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
}
div.tab-nav button {
    background: transparent !important;
    color: #444444 !important;
    border: none !important;
    border-radius: 18px !important;
    padding: 10px 18px !important;
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.1em !important;
}
div.tab-nav button.selected {
    background: rgba(244,196,212,0.35) !important;
    color: #222222 !important;
    box-shadow: 0 1px 8px rgba(244,196,212,0.2) !important;
}

/* ========================================
   タブコンテンツ: 二重フレーム（黒+グレーずらし）
   ======================================== */
.tabitem {
    background: transparent !important;
    border: none !important;
}
.tabitem > div {
    position: relative !important;
    background: #ffffff !important;
    border-radius: 0 !important;
    padding: 36px 24px !important;
    margin: 20px 22px 32px 22px !important;
    border: 1.25px solid #888888 !important;
    box-shadow: 10px 10px 0px 0px #c8c8c8 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

/* ========================================
   Gradioコンポーネントのスタイリング
   ======================================== */

/* ラベル */
label span, .label-wrap span {
    color: #222222 !important;
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', serif !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    letter-spacing: 0.06em !important;
}

/* テキスト入力 / Dropdown */
input[type="text"], textarea, select {
    background: rgba(255,255,255,0.7) !important;
    border: 1px solid rgba(200,191,224,0.3) !important;
    border-radius: 16px !important;
    color: #1a1a1a !important;
    font-weight: 400 !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: rgba(244,196,212,0.5) !important;
    box-shadow: 0 0 0 3px rgba(244,196,212,0.15) !important;
    outline: none !important;
}

/* Slider number input: 大人っぽく角ばった四角 */
input[type="number"] {
    background: #ffffff !important;
    border: 1.25px solid #888888 !important;
    border-radius: 0 !important;
    color: #1a1a1a !important;
    font-family: 'Cormorant Garamond', 'Georgia', serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.05em !important;
    text-align: center !important;
    padding: 4px 6px !important;
    box-shadow: 3px 3px 0px 0px #c8c8c8 !important;
    outline: none !important;
    -moz-appearance: textfield !important;
}
input[type="number"]:focus {
    border-color: #555555 !important;
    box-shadow: 4px 4px 0px 0px #aaaaaa !important;
    outline: none !important;
}
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}

/* Dropdown: 全体リセット */
[data-testid="dropdown"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}
[data-testid="dropdown"] > div,
[data-testid="dropdown"] .wrap,
[data-testid="dropdown"] .wrap-inner,
[data-testid="dropdown"] .secondary-wrap,
[data-testid="dropdown"] input,
[data-testid="dropdown"] .multiselect {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    outline: none !important;
}
/* 入力ラッパーのみ細い黒線で囲む */
[data-testid="dropdown"] .wrap,
[data-testid="dropdown"] .secondary-wrap {
    border: 1px solid #1a1a1a !important;
    box-shadow: 3px 3px 0px 0px #cccccc !important;
    padding: 8px 10px !important;
}
/* 子要素の文字色 */
[data-testid="dropdown"] *:not(ul):not(ul *) {
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border-radius: 0 !important;
}

/* Dropdown選択肢リスト: 黒背景・白文字 */
ul.options,
ul.options li,
.options,
.options .item,
.secondary-wrap .item,
.secondary-wrap ul li {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border-radius: 0 !important;
}
ul.options li:hover,
.options .item:hover,
.secondary-wrap .item:hover {
    background: #333333 !important;
    color: #ffffff !important;
}
ul.options li.selected,
.options .item.active {
    background: #555555 !important;
    color: #ffffff !important;
}

/* Slider: ラグジュアリー仕様 */
input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    height: 3px !important;
    background: linear-gradient(90deg,
        #e8c8d4 0%,
        #d4b8e0 40%,
        #b8d4e8 100%) !important;
    border-radius: 0 !important;
    outline: none !important;
    cursor: pointer !important;
    overflow: visible !important;
    margin: 12px 0 !important;
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 14px !important;
    height: 14px !important;
    background: #3a3a3a !important;
    border: 1.5px solid #888888 !important;
    border-radius: 0 !important;
    transform: rotate(45deg) !important;
    cursor: pointer !important;
    box-shadow: 2px 2px 4px rgba(0,0,0,0.2) !important;
    margin-top: -6px !important;
    position: relative !important;
}
input[type="range"]::-moz-range-thumb {
    width: 14px !important;
    height: 14px !important;
    background: #3a3a3a !important;
    border: 1.5px solid #888888 !important;
    border-radius: 0 !important;
    transform: rotate(45deg) !important;
    cursor: pointer !important;
}
input[type="range"]::-webkit-slider-runnable-track {
    height: 3px !important;
    background: linear-gradient(90deg,
        #e8c8d4 0%,
        #d4b8e0 40%,
        #b8d4e8 100%) !important;
    border-radius: 0 !important;
    overflow: visible !important;
}

/* Sliderブロック全体: 角ばったコンテナ */
[data-testid="slider"] {
    background: #fafafa !important;
    border: 1.25px solid #cccccc !important;
    border-radius: 0 !important;
    padding: 12px 14px 16px 14px !important;
    box-shadow: 3px 3px 0px 0px #d8d8d8 !important;
    overflow: visible !important;
}
[data-testid="slider"] > div,
[data-testid="slider"] .wrap,
[data-testid="slider"] .wrap-inner {
    overflow: visible !important;
}
[data-testid="slider"] .label-wrap span,
[data-testid="slider"] label span {
    font-family: 'Cormorant Garamond', 'Georgia', serif !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #555555 !important;
}
/* sliderのrefreshボタン（リセットアイコン）を角ばりに */
[data-testid="slider"] button {
    border-radius: 0 !important;
    border: 1.25px solid #aaaaaa !important;
    background: #f0f0f0 !important;
    padding: 4px 6px !important;
}

/* Radio */
.gr-radio-row label, [data-testid="radio-group"] label {
    color: #222222 !important;
    font-weight: 400 !important;
}

/* JSON表示 */
.json-holder, [data-testid="json"] {
    background: rgba(255,255,255,0.5) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(200,191,224,0.15) !important;
}

/* Textbox */
textarea {
    background: rgba(255,255,255,0.6) !important;
    color: #1a1a1a !important;
    border-radius: 16px !important;
    font-family: 'SF Mono', 'Fira Code', ui-monospace, monospace !important;
    font-size: 12px !important;
}

/* Markdown */
.prose, .markdown-text, .md {
    color: #1a1a1a !important;
}
.prose h2, .prose h3 {
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', serif !important;
    color: #222222 !important;
    font-weight: 400 !important;
    letter-spacing: 0.08em !important;
}
.prose table {
    color: #1a1a1a !important;
}
.prose table th {
    color: #222222 !important;
    font-weight: 500 !important;
    background: rgba(244,196,212,0.1) !important;
}
.prose table td {
    border-color: rgba(200,191,224,0.2) !important;
}

/* ========================================
   Gradioボタン: VIEW MORE風 / セリフ体 / シャープ
   ======================================== */
button[class*="primary"], button[class*="secondary"],
button.lg {
    border-radius: 0 !important;
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', 'YuMincho', serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 16px 28px !important;
    transition: all 0.3s ease !important;
    touch-action: manipulation !important;
    -webkit-tap-highlight-color: transparent !important;
}

/* Primary ボタン: ダーク背景 */
button[class*="primary"] {
    background: #3a3a3a !important;
    color: #d8d8d8 !important;
    border: 1px solid #3a3a3a !important;
    box-shadow: none !important;
}
button[class*="primary"]:hover {
    background: #4a4a4a !important;
    transform: none !important;
    box-shadow: none !important;
}
button[class*="primary"]:active {
    background: #555555 !important;
}

/* Secondary ボタン: 白背景 + 細線ボーダー */
button[class*="secondary"] {
    background: #ffffff !important;
    color: #555555 !important;
    border: 1px solid #aaaaaa !important;
    box-shadow: none !important;
}
button[class*="secondary"]:hover {
    background: #f5f5f5 !important;
    border-color: #888888 !important;
    transform: none !important;
    box-shadow: none !important;
}
button[class*="secondary"]:active {
    background: #eeeeee !important;
}

/* ========================================
   セクションタイトル
   ======================================== */
.section-title {
    text-align: center !important;
    padding: 20px 16px 4px 16px !important;
}
.section-title h2 {
    font-family: 'Cormorant Garamond', 'Georgia', 'Times New Roman', serif !important;
    font-size: clamp(16px, 4vw, 22px) !important;
    font-weight: 400 !important;
    color: #222222 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ========================================
   WORKFLOW以下: 背景を統一して視認性UP
   ======================================== */
.section-title,
.section-title ~ * {
    background-color: #ffffff !important;
}

/* ========================================
   レスポンシブ: PC = 中央固定幅
   ======================================== */
@media (min-width: 768px) {
    .gradio-container > .main,
    .gradio-container > div > .main {
        max-width: 480px !important;
        margin: 0 auto !important;
    }
    .sensor-hero {
        min-height: 55vh;
    }
    .tabitem > div {
        margin: 20px auto 32px auto !important;
        max-width: 440px !important;
    }
    div.tab-nav {
        max-width: 440px !important;
        margin: 16px auto 20px auto !important;
    }
}

/* ========================================
   スマホ特化: タッチ最適化
   ======================================== */
@media (max-width: 767px) {
    .sensor-hero {
        min-height: 70vh;
        padding: 48px 20px 40px 20px;
    }
    .sensor-hero__buttons {
        width: 100%;
        max-width: 300px;
    }
    .hero-btn {
        flex: 1 1 calc(50% - 5px);
        min-width: 130px;
        padding: 14px 10px;
        font-size: 13px;
    }
    .section-title {
        padding: 24px 16px 4px 16px !important;
    }
    div.tab-nav {
        margin: 12px 12px 16px 12px !important;
        padding: 4px !important;
    }
    div.tab-nav button {
        padding: 9px 12px !important;
        font-size: 13px !important;
    }
    .tabitem > div {
        margin: 16px 14px 28px 14px !important;
        padding: 28px 16px !important;
        box-shadow: 8px 8px 0px 0px #c8c8c8 !important;
    }
    button[class*="primary"], button[class*="secondary"],
    button.lg {
        width: 100% !important;
        padding: 15px 20px !important;
        font-size: 15px !important;
    }
    label span, .label-wrap span {
        font-size: 13px !important;
    }
    input[type="text"], input[type="number"], textarea, select {
        font-size: 16px !important;
    }
    /* Rowを縦並びに */
    .row, [class*="row"] {
        flex-direction: column !important;
    }
}

/* ========================================
   アニメーション
   ======================================== */
.tabitem > div {
    animation: softFadeIn 0.5s ease forwards;
}
@keyframes softFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ========================================
   スクロールバー: やわらかく
   ======================================== */
::-webkit-scrollbar {
    width: 4px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(200,191,224,0.3);
    border-radius: 4px;
}

/* ========================================
   Gradio内部のpadding/border補正
   ======================================== */
.block {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}
.form {
    background: transparent !important;
    border: none !important;
}
.container {
    background: transparent !important;
}
.tabs {
    background: #ffffff !important;
}

/* ========================================
   全テキスト強制黒（最終手段）
   ボタン・ヒーロー系は除外
   ======================================== */
body, body * {
    color: #1a1a1a !important;
}

/* ヒーローボタン: 色を個別に戻す */
.hero-btn--outline,
.hero-btn--outline * {
    color: #555555 !important;
}
.hero-btn--dark,
.hero-btn--dark * {
    color: #d8d8d8 !important;
}

/* Gradioボタン */
button[class*="primary"],
button[class*="primary"] * {
    color: #d8d8d8 !important;
}
button[class*="secondary"],
button[class*="secondary"] * {
    color: #555555 !important;
}

/* JSON表示の色 */
.json-holder *,
[data-testid="json"] * {
    color: #1a1a1a !important;
}
"""

# ── UI変更: HEAD メタタグ + セリフ体Webフォント読み込み ──
HEAD = """
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover, user-scalable=no">
<meta name="theme-color" content="#fdf2f8">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600&display=swap" rel="stylesheet">
"""

# =========================
# Gradio UI
# ── UI変更: Blocks構造をフェミニン・パステルデザインに再構築 ──
# ── 縦スクロール1カラム / 中央寄せ / 余白たっぷり ──
# =========================
# Gradio 6ではcss/theme/headはmount_gradio_app()で指定
with gr.Blocks() as demo:

    # ── UI変更: ヒーローセクションをトップに配置 ──
    gr.HTML(SENSOR_UI)

    # ── UI変更: セクションタイトル（やわらかいフォント） ──
    gr.Markdown("## - W O R K F L O W -", elem_classes=["section-title"])

    with gr.Tabs():

        # ── UI変更: 収集タブ — 縦並び1カラム ──
        with gr.Tab("収集"):
            label = gr.Dropdown(
                choices=DEFAULT_LABELS,
                value=DEFAULT_LABELS[0],
                label="ラベル"
            )
            btn_c_start = gr.Button("収集開始", size="lg")
            btn_c_stop = gr.Button("収集停止 → 保存", size="lg")
            collect_msg = gr.Markdown("-")
            counts_json = gr.JSON(value={"TOTAL": 0}, label="回数カウンタ")

            btn_c_start.click(collect_start, inputs=[label], outputs=[collect_msg, btn_c_start, btn_c_stop])
            btn_c_stop.click(collect_stop_and_save, inputs=None, outputs=[collect_msg, btn_c_start, btn_c_stop, counts_json])

        # ── UI変更: 学習タブ — スライダー縦並び ──
        with gr.Tab("学習"):
            stline = gr.Markdown("MODEL: not trained")
            window_sec = gr.Slider(0.6, 2.0, value=CFG.window_sec, step=0.1, label="window_sec")
            hop_sec = gr.Slider(0.1, 0.5, value=CFG.hop_sec, step=0.1, label="hop_sec")
            fs_target = gr.Slider(20, 100, value=CFG.fs_target, step=5, label="fs_target")
            feat_mode = gr.Radio(choices=["last", "mean"], value="last", label="state aggregation")
            btn_train = gr.Button("学習", variant="primary", size="lg")
            train_msg = gr.Markdown("-")
            train_cfg = gr.JSON(label="選ばれたハイパラ")
            tail_log = gr.Textbox(lines=6, label="ログ（末尾）")

            btn_train.click(train_click, inputs=[window_sec, hop_sec, fs_target, feat_mode],
                            outputs=[stline, train_cfg, train_msg, tail_log])

        # ── UI変更: 推論タブ — 予測表示中央 ──
        with gr.Tab("推論"):
            infer_state = gr.Markdown("推論: OFF")
            btn_i_start = gr.Button("推論開始", size="lg")
            btn_i_stop = gr.Button("推論停止", size="lg")
            stline2 = gr.Markdown("MODEL: not trained")
            pred_html = gr.HTML("<div style='font-size:24px;font-weight:800;opacity:.6'>-</div>")
            prob_json = gr.JSON(label="確信度（クラス別）")

            btn_i_start.click(infer_start, inputs=None, outputs=[infer_state, pred_html, prob_json, stline2])
            btn_i_stop.click(infer_stop, inputs=None, outputs=[infer_state, pred_html, prob_json, stline2])

            timer_inf = gr.Timer(value=CFG.hop_sec)
            timer_inf.tick(infer_tick, inputs=None, outputs=[pred_html, prob_json, stline2])

        # ── UI変更: 対話タブ ──
        with gr.Tab("対話"):
            gr.Markdown("### 推論結果")
            chat_big = gr.HTML("<div style='font-size:22px;font-weight:800;opacity:.6'>推論がOFFです</div>")
            chat_log = gr.Markdown("(log empty)")
            timer_chat = gr.Timer(value=0.3)
            timer_chat.tick(chat_tick, inputs=None, outputs=[chat_big, chat_log])

    demo.load(fn=None, inputs=None, outputs=None, js=JS_BOOT)


# =========================
# Mount Gradio into FastAPI (SSR OFF)
# =========================
# ── UI変更: Gradio 6ではcss/theme/head をmount_gradio_appに渡す ──
app = gr.mount_gradio_app(
    api, demo, path="/", ssr_mode=False,
    css=CSS,
    head=HEAD,
    theme=gr.themes.Base(
        text_size=gr.themes.sizes.text_md,
        font=["Inter", "Hiragino Kaku Gothic ProN", "Noto Sans JP", "sans-serif"],
    ).set(
        body_text_color="#1a1a1a",
        body_text_color_subdued="#333333",
        block_label_text_color="#222222",
        block_title_text_color="#1a1a1a",
        checkbox_label_text_color="#1a1a1a",
        table_text_color="#1a1a1a",
        link_text_color="#333333",
        color_accent_soft="#e8d5e0",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#1a1a1a",
        input_border_color_dark="#1a1a1a",
    ),
)


# =========================
# Run on Colab (background thread)
# =========================
def run_colab(server_port: int = 7860):
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=server_port, log_level="warning")
    server = uvicorn.Server(config)

    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    time.sleep(1.0)

    # If in colab, show public URL via Gradio share too (simpler UX)
    # Note: We can't "launch" gradio separately because FastAPI+mount is already serving.
    # So: use Colab's port proxy link if available, otherwise open localhost in browser.
    try:
        from google.colab import output
        proxy_url = output.eval_js(f"google.colab.kernel.proxyPort({server_port})")
        print("Open this URL (PC):", proxy_url)
        print("Open the same URL on your smartphone to use accelerometer.")
    except Exception:
        print(f"Server running on http://127.0.0.1:{server_port} (Colab proxy unavailable here).")

# Start
if "google.colab" in sys.modules:
    run_colab(7860)
else:
    # local python run (non-notebook)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
