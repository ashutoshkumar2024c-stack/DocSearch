"""Look and feel of the Streamlit app: one CSS block with colour variables for light and dark mode,
plus small HTML helpers (badges, cards, tables). Streamlit's own widgets are restyled through
their stable `data-testid` / `data-baseweb` hooks so dark mode covers the whole page."""

from __future__ import annotations

import html

LIGHT = {
    "bg": "#f4f6fb", "surface": "#ffffff", "surface2": "#f0f2f8", "text": "#172033", "muted": "#5f6b85",
    "border": "#e2e6f0", "primary": "#4f46e5", "primary2": "#0ea5e9", "shadow": "0 1px 2px rgba(16,24,40,.06), 0 4px 14px rgba(16,24,40,.05)",
    "hero_a": "#4f46e5", "hero_b": "#0ea5e9", "code": "#eef1f8",
}
DARK = {
    "bg": "#0b0f1a", "surface": "#141a29", "surface2": "#1b2336", "text": "#e8ebf5", "muted": "#9aa6c0",
    "border": "#27304a", "primary": "#8b93ff", "primary2": "#38bdf8", "shadow": "0 1px 2px rgba(0,0,0,.5), 0 6px 20px rgba(0,0,0,.35)",
    "hero_a": "#3730a3", "hero_b": "#0369a1", "code": "#10162a",
}

# route -> (label, light bg, light fg, dark bg, dark fg)
ROUTES = {
    "native": ("Text layer", "#dcfce7", "#166534", "#12351f", "#86efac"),
    "ocr": ("OCR", "#fef3c7", "#92400e", "#3b2a08", "#fcd34d"),
    "hybrid": ("Text + OCR", "#ede9fe", "#5b21b6", "#2a1d4d", "#c4b5fd"),
    "visual": ("Vision", "#fce7f3", "#9d174d", "#3d1530", "#f9a8d4"),
    "parser": ("Parser", "#dbeafe", "#1e40af", "#14284d", "#93c5fd"),
}


def route_label(route: str) -> str:
    return ROUTES.get(route, (route,))[0]


def css(dark: bool) -> str:
    v = DARK if dark else LIGHT
    route_rules = "\n".join(
        f".ds-r-{k} {{ background:{(c[3] if dark else c[1])} !important; color:{(c[4] if dark else c[2])} !important; }}"
        for k, c in ROUTES.items())
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {{ --bg:{v['bg']}; --surface:{v['surface']}; --surface2:{v['surface2']}; --text:{v['text']}; --muted:{v['muted']};
        --border:{v['border']}; --primary:{v['primary']}; --primary2:{v['primary2']}; --shadow:{v['shadow']}; --code:{v['code']}; }}
html, body, .stApp, [class*="css"] {{ font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }}
.stApp {{ background: var(--bg) !important; color: var(--text) !important; }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stHeader"] * {{ color: var(--muted) !important; }}
.block-container {{ padding-top: 1.6rem !important; max-width: 1180px; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div[data-testid="stMarkdownContainer"] {{ color: var(--text); }}
.stApp .ds-muted, .stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * {{ color: var(--muted) !important; }}
a {{ color: var(--primary) !important; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: var(--surface) !important; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarContent"] {{ padding-top: 1rem; }}

/* inputs */
.stApp input, .stApp textarea, .stApp [data-baseweb="input"], .stApp [data-baseweb="base-input"],
.stApp [data-baseweb="select"] > div, .stApp [data-baseweb="textarea"] {{
  background: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important; border-radius: 10px !important; }}
.stApp [data-baseweb="select"] * {{ color: var(--text) !important; }}
[data-baseweb="popover"] *, [data-baseweb="menu"], ul[role="listbox"], li[role="option"] {{ background: var(--surface) !important; color: var(--text) !important; }}
li[role="option"]:hover {{ background: var(--surface2) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--surface2) !important; border: 1.5px dashed var(--border) !important; border-radius: 14px !important; }}
[data-testid="stFileUploaderDropzone"] * {{ color: var(--muted) !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; }}
[data-testid="stFileUploader"] section + div * {{ color: var(--text) !important; }}

/* buttons */
.stApp .stButton > button, .stApp .stDownloadButton > button {{
  background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; font-weight: 500; box-shadow: none; transition: all .15s; }}
.stApp .stButton > button:hover {{ border-color: var(--primary) !important; color: var(--primary) !important; transform: translateY(-1px); }}
.stApp .stButton > button[kind="primary"], .stApp .stButton > button[data-testid="stBaseButton-primary"] {{
  background: linear-gradient(135deg, var(--primary), var(--primary2)) !important; color: #fff !important; border: 0 !important; }}
.stApp .stButton > button[kind="primary"] *, .stApp .stButton > button[data-testid="stBaseButton-primary"] * {{ color:#fff !important; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ background: transparent !important; color: var(--muted) !important; border-radius: 10px 10px 0 0; padding: 8px 14px; }}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * {{ color: var(--primary) !important; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--primary) !important; }}
.stTabs [data-baseweb="tab-border"] {{ background: var(--border) !important; }}

/* expanders, chat, alerts, status */
[data-testid="stExpander"] details, [data-testid="stStatus"] details, [data-testid="stStatusWidget"] {{
  background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; box-shadow: none; }}
[data-testid="stExpander"] summary, [data-testid="stStatus"] summary {{ color: var(--text) !important; }}
[data-testid="stChatMessage"] {{ background: var(--surface) !important; border: 1px solid var(--border); border-radius: 16px; padding: 14px 18px; margin-bottom: 10px; box-shadow: var(--shadow); }}
[data-testid="stChatMessage"] * {{ color: var(--text); }}
[data-testid="stChatInput"] {{ background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 16px !important; }}
[data-testid="stChatInput"] textarea {{ background: transparent !important; }}
[data-testid="stBottom"], [data-testid="stBottom"] > div {{ background: var(--bg) !important; }}
[data-testid="stAlert"] {{ border-radius: 12px; border: 1px solid var(--border); }}
[data-testid="stAlert"] * {{ color: inherit; }}
pre, code, [data-testid="stCode"] {{ background: var(--code) !important; color: var(--text) !important; border-radius: 10px; }}
hr {{ border-color: var(--border) !important; }}
[data-testid="stSlider"] [role="slider"] {{ background: var(--primary) !important; }}
[data-testid="stToggle"] label span {{ color: var(--text) !important; }}
[data-testid="stImage"] img {{ border-radius: 12px; border: 1px solid var(--border); }}
[data-testid="stMarkdownContainer"] table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
[data-testid="stMarkdownContainer"] th {{ background: var(--surface2); color: var(--text); text-align: left; }}
[data-testid="stMarkdownContainer"] th, [data-testid="stMarkdownContainer"] td {{ border: 1px solid var(--border); padding: 6px 10px; }}

/* ---------- custom components ---------- */
.ds-hero {{ background: linear-gradient(120deg, {v['hero_a']}, {v['hero_b']}); border-radius: 20px; padding: 26px 30px 22px; color: #fff;
  box-shadow: var(--shadow); margin-bottom: 18px; }}
.stApp .ds-hero, .stApp .ds-hero * {{ color: #fff !important; }}
.ds-hero .eyebrow {{ font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; opacity: .85; font-weight: 600; }}
.ds-hero h1 {{ margin: 4px 0 6px; font-size: 2.1rem; font-weight: 700; letter-spacing: -.02em; }}
.ds-hero p {{ margin: 0 0 16px; opacity: .92; font-size: 1rem; max-width: 760px; }}
.ds-steps {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.ds-step {{ background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.25); border-radius: 999px; padding: 5px 12px; font-size: .8rem; font-weight: 500; backdrop-filter: blur(4px); }}
.ds-step b {{ opacity: .75; margin-right: 6px; }}
.ds-cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0 18px; }}
.ds-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; box-shadow: var(--shadow); }}
.ds-card .k {{ font-size: .75rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.ds-card .v {{ color: var(--text); font-size: 1.7rem; font-weight: 700; margin-top: 2px; }}
.ds-card .s {{ font-size: .8rem; color: var(--muted); }}
.stApp .ds-card .k, .stApp .ds-card .s {{ color: var(--muted); }}
.ds-badge {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600; margin: 0 6px 4px 0; background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }}
.stApp .ds-badge {{ color: inherit; }}
.stApp .ds-badge.plain {{ color: var(--muted); }}
{route_rules}
.ds-ok {{ background:#dcfce7 !important; color:#166534 !important; border-color: transparent !important; }}
.ds-bad {{ background:#fee2e2 !important; color:#991b1b !important; border-color: transparent !important; }}
.ds-doc {{ background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; }}
.ds-doc .n {{ color: var(--text); font-weight: 600; font-size: .9rem; word-break: break-all; }}
.ds-doc .m {{ color: var(--muted); font-size: .78rem; margin-bottom: 4px; }}
.ds-strip {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0 12px; }}
.ds-pg {{ width: 26px; height: 26px; border-radius: 7px; font-size: .68rem; font-weight: 600; display: flex; align-items: center; justify-content: center; }}
.ds-pg.sel {{ outline: 2px solid var(--primary); outline-offset: 1px; }}
.ds-table {{ width: 100%; border-collapse: collapse; font-size: .88rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
.ds-table th {{ background: var(--surface2); text-align: left; padding: 8px 12px; font-weight: 600; color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .05em; }}
.ds-table td {{ color: var(--text); padding: 8px 12px; border-top: 1px solid var(--border); vertical-align: top; }}
.ds-reason {{ color: var(--text); background: var(--surface2); border-left: 3px solid var(--primary); border-radius: 8px; padding: 6px 10px; margin: 4px 0; font-size: .85rem; }}
.ds-model {{ font-size: .74rem; color: var(--muted); margin-top: 6px; }}
.ds-empty {{ text-align:center; padding: 28px 10px; color: var(--muted); }}
.ds-empty .big {{ font-size: 2.2rem; }}
</style>
"""


def badge(text: str, kind: str = "") -> str:
    cls = f"ds-badge ds-r-{kind}" if kind in ROUTES else f"ds-badge {kind or 'plain'}"
    return f'<span class="{cls}">{html.escape(str(text))}</span>'


def route_badge(route: str, count: int | None = None) -> str:
    label = route_label(route) + (f" · {count}" if count is not None else "")
    return badge(label, route)


def hero(steps: list[str]) -> str:
    chips = "".join(f'<span class="ds-step"><b>{i}</b>{html.escape(s)}</span>' for i, s in enumerate(steps, 1))
    return ('<div class="ds-hero"><div class="eyebrow">Hybrid OCR-RAG pipeline</div><h1>DocSearch</h1>'
            '<p>Intelligent document parsing and semantic search. Every page is classified, parsed with the '
            'right tool, indexed, and answered with citations.</p>'
            f'<div class="ds-steps">{chips}</div></div>')


def stat_cards(items: list[tuple[str, str, str]]) -> str:
    return '<div class="ds-cards">' + "".join(
        f'<div class="ds-card"><div class="k">{html.escape(k)}</div><div class="v">{html.escape(str(v))}</div>'
        f'<div class="s">{html.escape(s)}</div></div>' for k, v, s in items) + "</div>"


def table(rows: list[dict], raw_cols: tuple = ()) -> str:
    """HTML table. Values in `raw_cols` are trusted HTML (badges); all others are escaped."""
    if not rows:
        return ""
    cols = list(rows[0])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(
        f"<td>{r[c] if c in raw_cols else html.escape('' if r[c] is None else str(r[c]))}</td>" for c in cols) + "</tr>"
        for r in rows)
    return f'<table class="ds-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def page_strip(routes: list[str], selected: int | None = None) -> str:
    cells = []
    for i, r in enumerate(routes, 1):
        sel = " sel" if selected == i else ""
        cells.append(f'<div class="ds-pg ds-r-{r}{sel}" title="page {i}: {html.escape(route_label(r))}">{i}</div>')
    return '<div class="ds-strip">' + "".join(cells) + "</div>"


def doc_card(name: str, doc_type: str, pages: int, chunks: int, routes: dict) -> str:
    chips = "".join(route_badge(k, n) for k, n in routes.items())
    return (f'<div class="ds-doc"><div class="n">{html.escape(name)}</div>'
            f'<div class="m">{html.escape(doc_type)} · {pages} page(s) · {chunks} passages</div>{chips}</div>')
