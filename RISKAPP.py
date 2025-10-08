# streamlit_app.py
# ──────────────────────────────────────────────────────────────────────────────
# Streamlit NAIP Q&A with overlay + tidy summary panel (RMRS-GTR-315 framing)
#
# Privacy for your OpenAI API key:
# - Preferred: put OPENAI_API_KEY in .streamlit/secrets.toml like:
#       [general]
#       OPENAI_API_KEY = "sk-..."
# - Or set OS env var OPENAI_API_KEY before running.
# - Or paste the key into the password box in the app; it stays in session_state only.
#
# Run:  streamlit run streamlit_app.py
#
# Requires:
#   pip install streamlit geopandas rasterio pystac-client planetary-computer shapely rioxarray requests pillow numpy pandas openai matplotlib
#   (Note: On Windows, preinstall GDAL/GEOS/Fiona deps if needed; use conda-forge for ease.)
# ──────────────────────────────────────────────────────────────────────────────

import os, io, base64, json, textwrap, tempfile, contextlib
from io import BytesIO
from typing import Optional, Tuple

import streamlit as st
import numpy as np
from PIL import Image
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, Point, box, mapping
import geopandas as gpd
import pystac_client, planetary_computer
from openai import OpenAI
import matplotlib.pyplot as plt

# ── App config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NAIP Q&A – RMRS-GTR-315", layout="wide")

# ── Secrets / API key handling (privacy-first) ────────────────────────────────
def _get_openai_key() -> Optional[str]:
    # 1) Streamlit secrets
    key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
    if key:
        return key.strip()
    # 2) Environment
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    # 3) Session (user-provided, masked)
    return st.session_state.get("_OPENAI_API_KEY", None)

def _ensure_openai_client() -> Optional[OpenAI]:
    k = _get_openai_key()
    if not k:
        return None
    return OpenAI(api_key=k)

# ── Cached STAC client ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_stac_client():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

# ── Image helpers ─────────────────────────────────────────────────────────────
def pct_scale_to_u8(rgb_arr: np.ndarray) -> np.ndarray:
    if rgb_arr.dtype == np.uint8:
        return rgb_arr
    out = []
    for b in range(rgb_arr.shape[0]):
        band = rgb_arr[b].astype(np.float32)
        lo, hi = np.nanpercentile(band, [0.0, 99.5])
        if not np.isfinite(lo): lo = 0.0
        if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
        band = (np.clip((band - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)
        out.append(band)
    return np.stack(out, axis=0)

def find_best_naip_item(lon: float, lat: float, start: str, end: str, pad_deg: float = 0.001):
    catalog = get_stac_client()
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [lon - pad_deg, lat - pad_deg],
            [lon + pad_deg, lat - pad_deg],
            [lon + pad_deg, lat + pad_deg],
            [lon - pad_deg, lat + pad_deg],
            [lon - pad_deg, lat - pad_deg],
        ]]
    }
    items = catalog.search(
        collections=["naip"],
        intersects=aoi,
        datetime=f"{start}/{end}",
    ).item_collection()
    if not items or len(items) == 0:
        return None, aoi
    aoi_shape = shape(aoi)
    items_sorted = sorted(items, key=lambda it: shape(it.geometry).intersection(aoi_shape).area, reverse=True)
    return items_sorted[0], aoi

def crop_naip_at_point(lon: float, lat: float, zoom_m: int, start: str, end: str) -> Tuple[Optional[Image.Image], Optional[str], Optional[str]]:
    item, _ = find_best_naip_item(lon, lat, start=start, end=end)
    if item is None:
        return None, None, None
    href = None
    if "image" in item.assets:
        href = item.assets["image"].href
    else:
        for a in item.assets.values():
            if a.href.lower().endswith((".tif", ".tiff")):
                href = a.href
                break
    if href is None:
        return None, None, None

    with rasterio.Env():
        with rasterio.open(href) as src:
            pt = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs=4326).to_crs(src.crs)
            cx, cy = pt.geometry.iloc[0].x, pt.geometry.iloc[0].y
            crop_geom = [mapping(box(cx - zoom_m, cy - zoom_m, cx + zoom_m, cy + zoom_m))]
            data, _ = mask(src, crop_geom, crop=True)
            if data.shape[0] >= 3:
                rgb = data[:3, :, :]
            else:
                rgb = np.repeat(data[0:1, :, :], 3, axis=0)
            rgb_u8 = pct_scale_to_u8(rgb)
            pil = Image.fromarray(np.transpose(rgb_u8, (1, 2, 0)))
            return pil, item.id, href

# ── VLM Q&A ───────────────────────────────────────────────────────────────────
def ask_image_question(pil_image: Image.Image, question: str, system_preamble: Optional[str], model: str, temperature: float, client: OpenAI) -> str:
    buf = BytesIO()
    pil_image.save(buf, format="PNG")
    img_data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    sys_msg = system_preamble or "You are a careful remote sensing analyst. Answer concisely and only from the image."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": img_data_url}}
            ]}
        ],
        temperature=temperature
    )
    return str(resp.choices[0].message.content or "").strip()

# ── Rendering tidy overlay ────────────────────────────────────────────────────
def _safe_json_loads(txt: str):
    s = txt.strip()
    if s.startswith("```"):
        lines = [ln for ln in s.splitlines() if not ln.strip().startswith("```")]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except Exception:
        return {}

def _wrap(ax, label, text, width=56, ystart=None, lh=1.25, bullet=None, fontsize=9):
    if text is None or text == "":
        return ystart
    wrapped = textwrap.fill(str(text), width=width, break_long_words=False, break_on_hyphens=False)
    prefix = (bullet + " ") if bullet else ""
    ax.text(0.02, ystart, f"{label} {prefix}{wrapped}", va="top", ha="left", fontsize=fontsize, family="monospace")
    return ystart - 0.04 * (wrapped.count("\n") + 1)

def render_overlay_and_summary(pil_image: Image.Image, answer_json_text: str,
                               lon: float, lat: float, naip_id: str) -> Image.Image:
    data = _safe_json_loads(answer_json_text)

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.08)

    axL = fig.add_subplot(gs[0, 0])
    axL.imshow(pil_image)
    axL.axis("off")
    axL.set_title(f"NAIP @ {lat:.6f}, {lon:.6f}\n{naip_id}", fontsize=9)

    axR = fig.add_subplot(gs[0, 1])
    axR.axis("off")
    axR.set_title("Image-Grounded Fuel Summary", fontsize=11)

    y = 0.98
    fw = data.get("framework", "RMRS-GTR-315")
    axR.text(0.02, y, f"Framework: {fw}", va="top", ha="left", fontsize=9, family="monospace"); y -= 0.045

    zones = data.get("zones", {})

    # Around home
    zA = zones.get("around_home", {})
    fuA = zA.get("fuels", {})
    axR.text(0.02, y, "Around home", va="top", ha="left", fontsize=10, fontweight="bold", family="monospace"); y -= 0.04
    y = _wrap(axR, "  Fuels.primary:", fuA.get("primary"), ystart=y)
    y = _wrap(axR, "  continuity:", fuA.get("continuity"), ystart=y)
    y = _wrap(axR, "  height_class:", fuA.get("height_class"), ystart=y)
    y = _wrap(axR, "  canopy_over_roof:", fuA.get("canopy_over_roof"), ystart=y)
    y = _wrap(axR, "  surface_litter:", fuA.get("surface_litter"), ystart=y)
    y = _wrap(axR, "  receptive_surfaces:", ", ".join(fuA.get("receptive_surfaces", [])) or None, ystart=y)
    y = _wrap(axR, "  barriers:", ", ".join(fuA.get("barriers", [])) or None, ystart=y)
    y = _wrap(axR, "  notes:", fuA.get("notes"), ystart=y)
    y = _wrap(axR, "  assessment:", zA.get("concise_assessment"), ystart=y)
    recA = zA.get("recommended_actions", [])
    if recA:
        for r in recA:
            y = _wrap(axR, "  action:", r, ystart=y, bullet="•")

    y -= 0.02

    # Rest of image
    zB = zones.get("rest_of_image", {})
    fuB = zB.get("fuels", {})
    axR.text(0.02, y, "Rest of image", va="top", ha="left", fontsize=10, fontweight="bold", family="monospace"); y -= 0.04
    y = _wrap(axR, "  Fuels.primary:", fuB.get("primary"), ystart=y)
    y = _wrap(axR, "  continuity:", fuB.get("continuity"), ystart=y)
    y = _wrap(axR, "  height_class:", fuB.get("height_class"), ystart=y)
    y = _wrap(axR, "  canopy_density:", fuB.get("canopy_density"), ystart=y)
    y = _wrap(axR, "  slope_cues:", fuB.get("slope_cues"), ystart=y)
    y = _wrap(axR, "  barriers:", ", ".join(fuB.get("barriers", [])) or None, ystart=y)
    y = _wrap(axR, "  notes:", fuB.get("notes"), ystart=y)
    y = _wrap(axR, "  assessment:", zB.get("concise_assessment"), ystart=y)
    recB = zB.get("recommended_actions", [])
    if recB:
        for r in recB:
            y = _wrap(axR, "  action:", r, ystart=y, bullet="•")

    y -= 0.02
    y = _wrap(axR, "Overall:", data.get("overall_summary"), ystart=y)
    axR.text(0.02, 0.02, f"Uncertainty: {data.get('uncertainty', '')}", va="bottom", ha="left", fontsize=9, family="monospace")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()

# ── Default system preamble & question template ───────────────────────────────
DEFAULT_SYSTEM = (
    "You are a wildfire risk analyst. Use terminology and framing from USDA Forest Service "
    "RMRS-GTR-315 (Scott, Thompson, Calkin 2013). Stick strictly to what is visible in the image. "
    "Do not invent weather, moisture, or modeled outputs. Base all assessments solely on visible fuels, "
    "topography cues/shadows, access, and vegetation condition. Be concise, structured, and neutral."
)

DEFAULT_PROMPT = r"""
Using only visual evidence in this NAIP crop (~0.6–1 m RGB), **describe the fuels and defensible space conditions**:

1. **Immediately around the home** (direct adjacency to walls, roof, deck, fence, driveway)
2. **Across the rest of the image** (broader surroundings, landscape context)

Focus on visible fuel type, arrangement, continuity, and any structure-adjacent vulnerabilities. 
Avoid assumptions beyond what is clearly visible.

Output **strictly as JSON**:

{
  "framework": "RMRS-GTR-315",
  "zones": {
    "around_home": {
      "fuels": {
        "primary": "<grass|shrub|tree_litter|mixed|bare|hardscape>",
        "continuity": "<none|discontinuous|continuous>",
        "height_class": "<herb|shrub|crown|mixed>",
        "canopy_over_roof": "<none|minor|moderate|substantial>",
        "surface_litter": "<none|thin|moderate|deep>",
        "receptive_surfaces": ["<mulch|wood_fence|wood_pile|deck|plantings_against_wall|none>"],
        "barriers": ["<driveway|patio|rock|lawn|bare_soil|none>"],
        "notes": "<=25 words describing proximity to walls/vents/deck and any ember traps>"
      },
      "concise_assessment": "<=25 words summarizing flame contact or ember vulnerability near structure>",
      "recommended_actions": [
        "<=12 words e.g., replace mulch with rock>",
        "<=12 words e.g., limb branches off roof>",
        "<=12 words e.g., move wood pile away>"
      ]
    },
    "rest_of_image": {
      "fuels": {
        "primary": "<grass|shrub|tree_litter|mixed|bare|hardscape>",
        "continuity": "<none|discontinuous|continuous>",
        "height_class": "<herb|shrub|crown|mixed>",
        "canopy_density": "<open|moderate|closed>",
        "slope_cues": "<upslope_from_home|downslope_from_home|cross_slope|flat|unclear>",
        "barriers": ["<road|One way in driveway|Two ways in driveway|trail|lawn|rock|fuel_break|none>"],
        "notes": "<=25 words on apparent fire spread pathways toward or away from home>"
      },
      "concise_assessment": "<=25 words summarizing likely fire/ember approach from surrounding fuels>",
      "recommended_actions": [
        "<=12 words e.g., break up shrub bands>",
        "<=12 words e.g., maintain short green lawn>",
        "<=12 words e.g., limb trees 2–3 m>"
      ]
    }
  },
  "overall_summary": "<=30 words comparing around-home vs rest-of-image priorities for reducing home ignition potential>",
  "uncertainty": <float 0-1>
}

Constraints:
- Keep text fields within the word limits.
- Base statements only on visible evidence (fuels, structure adjacency, shadows/topography cues).
- Omit any field with no clear visual evidence.
"""

# ── Sidebar: configuration ────────────────────────────────────────────────────
st.sidebar.header("Configuration")
with st.sidebar:
    st.markdown("**OpenAI API key (private)**")
    if not _get_openai_key():
        key_input = st.text_input("Paste OPENAI_API_KEY (kept in session only)", type="password")
        if key_input:
            st.session_state["_OPENAI_API_KEY"] = key_input.strip()
    model = st.text_input("Model", value="gpt-4o")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
    st.divider()
    lon = st.number_input("Longitude", value=-105.308319, format="%.6f")
    lat = st.number_input("Latitude", value=40.655514, format="%.6f")
    zoom_m = st.slider("Crop half-size (meters)", 50, 400, 100, step=10)
    date_start = st.text_input("Date start (YYYY-MM-DD)", value="2018-01-01")
    date_end   = st.text_input("Date end (YYYY-MM-DD)", value="2024-01-01")
    st.divider()
    system_preamble = st.text_area("System preamble", value=DEFAULT_SYSTEM, height=120)
    question = st.text_area("Question prompt", value=DEFAULT_PROMPT, height=360)
    run_btn = st.button("Run NAIP Q&A", type="primary", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("NAIP Q&A • RMRS-GTR-315 Overlay")
st.caption("Loads NAIP near the point, asks a VLM for JSON summary, and renders a tidy panel.")

colL, colR = st.columns([1, 1])
client = _ensure_openai_client()

if run_btn:
    if client is None:
        st.error("OpenAI API key is missing. Add it in .streamlit/secrets.toml, set the env var, or paste it (masked) in the sidebar.")
        st.stop()

    with st.spinner("Fetching NAIP and running analysis..."):
        pil, naip_id, cog_href = crop_naip_at_point(lon, lat, zoom_m=zoom_m, start=date_start, end=date_end)
        if pil is None:
            st.error("No NAIP scene found for the given point/date range.")
            st.stop()

        answer = ask_image_question(pil, question, system_preamble, model, temperature, client)

    with colL:
        st.subheader("NAIP crop")
        st.image(pil, use_column_width=True, caption=f"NAIP @ {lat:.6f}, {lon:.6f} • {naip_id}")
        st.code(f"COG: {cog_href}", language="text")
        st.subheader("Raw model output")
        st.code(answer, language="json")

    # Render overlay
    with st.spinner("Rendering overlay panel..."):
        overlay_img = render_overlay_and_summary(pil_image=pil, answer_json_text=answer, lon=lon, lat=lat, naip_id=naip_id)

    with colR:
        st.subheader("Overlay + Tidy Summary")
        st.image(overlay_img, use_column_width=True)
        # Download
        buf = io.BytesIO()
        overlay_img.save(buf, format="PNG")
        st.download_button("Download overlay PNG", data=buf.getvalue(), file_name=f"naip_overlay_{lat:.6f}_{lon:.6f}.png".replace("-","m").replace(".","p"), mime="image/png")

# Footer notes
st.markdown("---")
st.markdown("**Privacy note:** The API key is read from `st.secrets` or environment if available. If you paste it in the sidebar, it's stored only in the current `st.session_state` and is not written to disk.")
