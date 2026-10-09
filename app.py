"""
Screener Daytrade IDX — "Base datar + pernah tinggi + lonjakan volume pagi"
Data: endpoint screener TradingView (tidak resmi, bisa berubah sewaktu-waktu).
Jalankan:  streamlit run app.py
Bukan penasihat investasi.
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Screener Daytrade IDX", page_icon="📈", layout="wide")

WIB = dt.timezone(dt.timedelta(hours=7))
SCAN_URL = "https://scanner.tradingview.com/indonesia/scan"

# Kolom TradingView -> nama internal. Kolom opsional boleh gagal; app tetap jalan.
CORE_COLS = {
    "name": "kode",
    "description": "nama",
    "close": "harga",
    "open": "open",
    "change": "chg",
    "volume": "volume",
    "average_volume_30d_calc": "vol_avg30",
    "Value.Traded": "value",
    "SMA20": "ma20",
    "SMA50": "ma50",
    "Perf.1M": "perf1m",
    "price_52_week_high": "high52",
    "market_cap_basic": "mcap",
}
OPTIONAL_COLS = {
    "relative_volume_intraday|5": "rvat",  # Relative Volume at Time
    "VWAP": "vwap",
}

# ───────────────────────── PRESET ─────────────────────────
PRESETS = {
    "KETAT (pagi)": dict(min_price=100, max_mcap_t=5.0, ma_lo=0.97, ma_hi=1.03,
                         perf_lo=-8.0, perf_hi=12.0, high52=0.65, use_vol=True, rvol=4.5,
                         value_full_m=7.0, use_chg=True, chg_lo=2.0, chg_hi=7.0, strength=True),
    "LONGGAR (pagi)": dict(min_price=60, max_mcap_t=5.0, ma_lo=0.93, ma_hi=1.07,
                           perf_lo=-12.0, perf_hi=18.0, high52=0.80, use_vol=True, rvol=3.0,
                           value_full_m=3.5, use_chg=True, chg_lo=1.0, chg_hi=10.0, strength=False),
    "WATCHLIST MALAM": dict(min_price=60, max_mcap_t=5.0, ma_lo=0.95, ma_hi=1.05,
                            perf_lo=-10.0, perf_hi=10.0, high52=0.70, use_vol=False, rvol=3.0,
                            value_full_m=0.1, use_chg=False, chg_lo=1.0, chg_hi=10.0, strength=False),
}

# Estimasi porsi kumulatif volume harian per menit sejak 09:00 WIB (pola U, BUKAN data resmi).
# (menit sejak 09:00, porsi kumulatif). Istirahat siang dianggap datar.
CURVE_MON_THU = [(0, 0.03), (15, 0.09), (30, 0.14), (60, 0.24), (90, 0.32), (180, 0.55),
                 (270, 0.55), (330, 0.72), (390, 0.90), (420, 1.0)]
CURVE_FRI = [(0, 0.03), (15, 0.09), (30, 0.14), (60, 0.24), (90, 0.32), (150, 0.52),
             (300, 0.52), (345, 0.70), (390, 0.90), (420, 1.0)]


def expected_fraction(now: dt.datetime) -> float:
    """Perkiraan porsi volume harian yang normalnya sudah terjadi pada jam `now` (WIB)."""
    if now.weekday() >= 5:
        return 1.0
    minutes = (now.hour - 9) * 60 + now.minute
    if minutes < 0 or minutes >= 420:
        return 1.0
    curve = CURVE_FRI if now.weekday() == 4 else CURVE_MON_THU
    xs, ys = zip(*curve)
    return float(np.interp(minutes, xs, ys))


def ara_limit(price: float) -> float:
    if price < 200:
        return 35.0
    if price <= 5000:
        return 25.0
    return 20.0


def tick_size(price: float) -> float:
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2000:
        return 5
    if price < 5000:
        return 10
    return 25


# ───────────────────────── DATA ─────────────────────────
def _scan(columns, cookie):
    payload = {
        "markets": ["indonesia"],
        "symbols": {"query": {"types": []}, "tickers": []},
        "options": {"lang": "en"},
        "filter": [{"left": "type", "operation": "equal", "right": "stock"}],
        "columns": columns,
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, 2000],
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Origin": "https://www.tradingview.com",
        "Referer": "https://www.tradingview.com/",
        "Content-Type": "application/json",
    }
    if cookie:
        headers["Cookie"] = f"sessionid={cookie.strip()}"
    r = requests.post(SCAN_URL, json=payload, headers=headers, timeout=20)
    r.raise_for_status()
    js = r.json()
    if "data" not in js:
        raise ValueError(str(js)[:300])
    return js["data"]


@st.cache_data(ttl=60, show_spinner=False)
def fetch_tradingview(cookie: str = ""):
    """Ambil semua saham IDX. Kalau kolom opsional ditolak, ulangi tanpa kolom itu."""
    tries = [{**CORE_COLS, **OPTIONAL_COLS}, {**CORE_COLS, "VWAP": "vwap"}, dict(CORE_COLS)]
    last_err = None
    for colmap in tries:
        try:
            rows = _scan(list(colmap.keys()), cookie)
            df = pd.DataFrame([r["d"] for r in rows], columns=list(colmap.values()))
            missing = [c for c in OPTIONAL_COLS.values() if c not in df.columns]
            for c in missing:
                df[c] = np.nan
            return df, missing, None
        except Exception as e:  # noqa: BLE001
            last_err = e
    return None, [], f"{type(last_err).__name__}: {last_err}"


def demo_data(n=400, seed=7) -> pd.DataFrame:
    """Data contoh acak untuk mencoba tampilan (bukan data pasar)."""
    rng = np.random.default_rng(seed)
    harga = np.round(rng.lognormal(5.3, 0.9, n)).clip(55, 9000)
    ma50 = harga * rng.normal(1.0, 0.08, n)
    ma20 = ma50 * rng.normal(1.0, 0.04, n)
    high52 = harga * rng.uniform(1.05, 3.0, n)
    vol_avg = rng.lognormal(15, 1.2, n)
    rvat = rng.lognormal(0.1, 0.8, n)
    chg = rng.normal(0.8, 3.5, n)
    df = pd.DataFrame({
        "kode": [f"DM{i:03d}" for i in range(n)],
        "nama": ["Data contoh"] * n,
        "harga": harga,
        "open": harga / (1 + rng.normal(0.01, 0.02, n)),
        "chg": chg,
        "volume": vol_avg * rvat * 0.14,
        "vol_avg30": vol_avg,
        "value": vol_avg * rvat * 0.14 * harga,
        "ma20": ma20,
        "ma50": ma50,
        "perf1m": rng.normal(0, 12, n),
        "high52": high52,
        "mcap": rng.lognormal(27.5, 1.3, n),
        "rvat": rvat,
        "vwap": harga * rng.normal(0.99, 0.01, n),
    })
    return df


# ───────────────────────── LOGIKA SCREENER ─────────────────────────
def apply_rules(df: pd.DataFrame, p: dict, frac: float):
    """Kembalikan (hasil, funnel). Funnel = sisa saham setelah tiap rule."""
    d = df.copy()
    d = d.dropna(subset=["harga", "ma20", "ma50", "high52"])
    # Rasio volume terhadap "normal di jam yang sama". Pakai RVAT TradingView kalau ada.
    est = (d["volume"] / d["vol_avg30"]) / max(frac, 0.01)
    d["rvol"] = d["rvat"].where(d["rvat"].notna(), est)
    d["rvol_src"] = np.where(d["rvat"].notna(), "RVAT", "estimasi")
    d["ma_ratio"] = d["ma20"] / d["ma50"]
    d["dari_puncak"] = (d["harga"] / d["high52"] - 1) * 100
    d["ara"] = d["harga"].apply(ara_limit)
    d["ruang_ara"] = d["ara"] - d["chg"]
    d["tick_pct"] = d["harga"].apply(tick_size) / d["harga"] * 100
    value_min = p["value_full_m"] * 1e9 * frac

    rules = [
        (f"Harga ≥ {p['min_price']:.0f}", d["harga"] >= p["min_price"]),
        (f"Market cap ≤ {p['max_mcap_t']:.1f} T", d["mcap"].fillna(0) <= p["max_mcap_t"] * 1e12),
        (f"MA20/MA50 {p['ma_lo']:.2f}–{p['ma_hi']:.2f}", d["ma_ratio"].between(p["ma_lo"], p["ma_hi"])),
        (f"Return 1 bulan {p['perf_lo']:.0f}% s/d {p['perf_hi']:.0f}%", d["perf1m"].between(p["perf_lo"], p["perf_hi"])),
        (f"Harga ≤ {p['high52']:.2f} × 52W high", d["harga"] <= p["high52"] * d["high52"]),
        (f"Value ≥ Rp{value_min/1e6:,.0f} jt (skala jam)", d["value"].fillna(0) >= value_min),
    ]
    if p["use_vol"]:
        rules.append((f"Volume ≥ {p['rvol']:.1f}x normal jam ini", d["rvol"] >= p["rvol"]))
    if p["use_chg"]:
        rules.append((f"Kenaikan hari ini {p['chg_lo']:.0f}%–{p['chg_hi']:.0f}%", d["chg"].between(p["chg_lo"], p["chg_hi"])))
    if p.get("strength"):
        ok_open = d["harga"] >= d["open"].fillna(0)
        ok_vwap = d["vwap"].isna() | (d["harga"] >= d["vwap"])
        rules.append(("Harga ≥ Open & ≥ VWAP", ok_open & ok_vwap))

    mask = pd.Series(True, index=d.index)
    funnel = [("Semua saham", int(len(d)))]
    for name, cond in rules:
        mask &= cond.fillna(False)
        funnel.append((name, int(mask.sum())))
    return d[mask].sort_values("rvol", ascending=False), funnel


# ───────────────────────── UI ─────────────────────────
now = dt.datetime.now(WIB)
st.title("📈 Screener Daytrade IDX")
st.caption("Base datar 1–3 bulan · pernah tinggi · lonjakan volume pagi · harga belum lari. "
           "Bukan penasihat investasi.")

with st.sidebar:
    st.header("Pengaturan")
    preset_name = st.radio("Preset", list(PRESETS.keys()), index=0)
    base = PRESETS[preset_name]
    with st.expander("Ubah angka rule", expanded=False):
        p = dict(base)
        p["min_price"] = st.number_input("Harga minimum", value=float(base["min_price"]), step=5.0)
        p["max_mcap_t"] = st.number_input("Market cap maks (T)", value=base["max_mcap_t"], step=0.5)
        p["ma_lo"], p["ma_hi"] = st.slider("MA20 ÷ MA50", 0.80, 1.20, (base["ma_lo"], base["ma_hi"]), 0.01)
        p["perf_lo"], p["perf_hi"] = st.slider("Return 1 bulan (%)", -40.0, 40.0, (base["perf_lo"], base["perf_hi"]), 1.0)
        p["high52"] = st.slider("Harga ≤ x × 52W high", 0.30, 1.00, base["high52"], 0.05)
        p["value_full_m"] = st.number_input("Value minimal setara 1 hari penuh (Rp miliar)",
                                            value=base["value_full_m"], step=0.5)
        if base["use_vol"]:
            p["rvol"] = st.slider("Volume ≥ x kali normal jam ini", 1.0, 10.0, base["rvol"], 0.5)
        if base["use_chg"]:
            p["chg_lo"], p["chg_hi"] = st.slider("Kenaikan hari ini (%)", -5.0, 35.0, (base["chg_lo"], base["chg_hi"]), 0.5)
            p["strength"] = st.checkbox("Wajib harga ≥ Open & ≥ VWAP", value=base["strength"])
    watch = st.text_area("Watchlist malam (kode, pisah koma/spasi) → ditandai ⭐ kelas A", "")
    cookie = st.text_input("TradingView sessionid (opsional, untuk data lebih real-time)", type="password")
    demo = st.checkbox("Pakai data contoh (uji tampilan)", value=False)

is_night = preset_name == "WATCHLIST MALAM"
frac = 1.0 if is_night else expected_fraction(now)

c1, c2, c3 = st.columns(3)
c1.metric("Jam (WIB)", now.strftime("%H:%M"))
c2.metric("Porsi volume normal s/d jam ini", f"{frac*100:.0f}%")
c3.metric("Preset", preset_name)
if not is_night and not (9 * 60 + 10 <= now.hour * 60 + now.minute <= 10 * 60 + 30):
    st.info("Preset pagi paling bermakna dijalankan antara 09:15–10:30 WIB (ideal 09:30).")

if st.button("🔍 Scan sekarang", type="primary", width="stretch") or demo:
    if demo:
        df, missing, err = demo_data(), [], None
        st.warning("Mode DATA CONTOH: angka acak, bukan data pasar.")
    else:
        with st.spinner("Mengambil data TradingView…"):
            df, missing, err = fetch_tradingview(cookie)
    if err:
        st.error(f"Gagal mengambil data: {err}")
        st.stop()
    if missing:
        st.caption(f"Kolom tidak tersedia: {', '.join(missing)} → rasio volume memakai estimasi kurva jam.")

    res, funnel = apply_rules(df, p, frac)
    wl = {w.strip().upper() for w in watch.replace(",", " ").split() if w.strip()}

    st.subheader(f"Hasil: {len(res)} saham")
    if len(res) == 0:
        st.write("Kosong. Lihat **Diagnostik** di bawah untuk melihat rule mana yang membuat hasil habis.")
    else:
        out = pd.DataFrame({
            "⭐": res["kode"].isin(wl).map({True: "⭐", False: ""}),
            "Kode": res["kode"],
            "Harga": res["harga"].round(0),
            "Chg %": res["chg"].round(1),
            "Ruang ke ARA %": res["ruang_ara"].round(1),
            "Vol x normal": res["rvol"].round(1),
            "Value (Rp jt)": (res["value"] / 1e6).round(0),
            "Dari 52W high %": res["dari_puncak"].round(0),
            "MA20/MA50": res["ma_ratio"].round(3),
            "Return 1B %": res["perf1m"].round(1),
            "Tick %": res["tick_pct"].round(2),
            "Stockbit": "https://stockbit.com/symbol/" + res["kode"],
        })
        out = out.sort_values(["⭐", "Vol x normal"], ascending=[False, False])
        st.dataframe(out, hide_index=True, width="stretch",
                     column_config={"Stockbit": st.column_config.LinkColumn("Stockbit", display_text="buka")})
        st.download_button("⬇️ Unduh CSV (untuk jurnal)",
                           out.to_csv(index=False).encode("utf-8"),
                           file_name=f"screener_{now:%Y%m%d_%H%M}.csv", mime="text/csv")
        if (res["rvol_src"] == "estimasi").any():
            st.caption("‘Vol x normal’ = estimasi dari volume ÷ rata-rata 30 hari ÷ porsi normal jam ini.")

    with st.expander("Diagnostik: sisa saham setelah tiap rule"):
        st.dataframe(pd.DataFrame(funnel, columns=["Rule", "Sisa saham"]), hide_index=True,
                     width="stretch")

st.divider()
st.markdown(
    "**Setelah saham muncul (cek manual 5 menit):** notasi/UMA/berita → running trade (pasar reguler?, "
    "pembeli menyambar offer?) → Value÷Freq vs kemarin → orderbook (spread, tembok offer) → broker summary "
    "1M/3M dari catatan malam.  \n"
    "**Catatan:** MA60 Stockbit diganti SMA50 TradingView; data TradingView untuk IDX bisa tertunda. "
    "Saham yang baru keluar papan pemantauan khusus sering lolos secara palsu."
)
