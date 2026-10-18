
import streamlit as st
import pandas as pd
from src.hybrid_model import recommend_hybrid, SUGGESTED_PROMPTS

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Scent Sense | AI Fragrance Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================
# PERFORMANCE: CACHE REPEATED RECOMMENDATIONS
# ==================================================
@st.cache_data(show_spinner=False, max_entries=100)
def cached_recommendations(query, top_k):
    """Reuse results for identical queries instead of recomputing."""
    return recommend_hybrid(query, top_k=top_k)


# ==================================================
# PREMIUM LUXURY THEME
# ==================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600;700&display=swap');

    :root {
        --bg: #10100f;
        --panel: #191917;
        --gold: #d6b77a;
        --gold-light: #ead5a4;
        --text: #f5f0e7;
        --muted: #a5a096;
        --border: rgba(214,183,122,0.19);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(16,16,15,0.95);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: var(--text) !important;
        font-family: 'Playfair Display', Georgia, serif !important;
    }

    p, label, li {
        color: #d4cec2;
    }

    .hero-panel {
        background:
            radial-gradient(ellipse at 90% 10%, rgba(214,183,122,0.12), transparent 45%),
            linear-gradient(135deg, #1c1b18, #121211);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: clamp(26px, 5vw, 55px);
        margin: 12px 0 32px;
    }

    .eyebrow, .section-label, .result-kicker {
        color: var(--gold);
        font-size: 0.72rem;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        font-weight: 700;
    }

    .hero-title {
        color: var(--text);
        font-family: 'Playfair Display', Georgia, serif;
        font-size: clamp(3rem, 7vw, 5.4rem);
        line-height: 1.05;
        letter-spacing: -2px;
        margin: 18px 0;
    }

    .hero-title span {
        color: var(--gold);
        font-style: italic;
    }

    .hero-description, .muted-copy {
        color: var(--muted);
        line-height: 1.8;
    }

    .hero-description {
        font-size: 1rem;
        max-width: 650px;
    }

    .section-heading {
        font-family: 'Playfair Display', Georgia, serif;
        color: var(--text);
        font-size: 1.8rem;
        margin: 8px 0;
    }

    .stTextArea textarea {
        background: #1a1917 !important;
        color: var(--text) !important;
        border: 1px solid rgba(214,183,122,0.27) !important;
        border-radius: 15px !important;
        font-size: 1rem !important;
        padding: 16px !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--gold) !important;
        box-shadow: 0 0 0 1px var(--gold) !important;
    }

    .stButton > button {
        background: #1c1b18;
        color: #e9dfcb;
        border: 1px solid var(--border);
        border-radius: 12px;
        min-height: 46px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: var(--gold);
        background: #27241e;
        color: var(--gold-light);
    }

    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #e4ca91, #c5a365) !important;
        color: #171510 !important;
        border: none !important;
        border-radius: 12px !important;
        min-height: 53px;
        font-weight: 700 !important;
        box-shadow: 0 5px 25px rgba(214,183,122,0.12);
    }

    .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #f0dbab, #d6b77a) !important;
        color: #171510 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border-color: var(--border) !important;
        border-radius: 18px !important;
    }

    .perfume-name {
        font-family: 'Playfair Display', Georgia, serif;
        color: var(--text);
        font-size: 1.65rem;
        line-height: 1.3;
        overflow-wrap: anywhere;
    }

    .brand-name {
        color: var(--gold);
        font-size: 0.87rem;
        font-weight: 600;
        margin: 5px 0 15px;
    }

   

    .feature-chip {
        display: inline-block;
        padding: 6px 10px;
        margin: 4px 4px 4px 0;
        border-radius: 30px;
        border: 1px solid rgba(214,183,122,0.24);
        background: rgba(214,183,122,0.07);
        color: #e3d1ac;
        font-size: 0.76rem;
    }

    .metric-box {
        background: #151513;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 15px;
        min-height: 75px;
        margin-top: 12px;
    }

    .metric-value {
        color: var(--gold-light);
        font-size: 1rem;
        font-weight: 600;
        margin-top: 7px;
        overflow-wrap: anywhere;
    }

    [data-testid="stExpander"] {
        border-color: var(--border);
        border-radius: 12px;
        background: rgba(255,255,255,0.015);
    }

    hr {
        border-color: rgba(214,183,122,0.15) !important;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }

    .footer {
        text-align: center;
        color: #777268;
        font-size: 0.78rem;
        line-height: 1.8;
        padding-top: 14px;
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-title {
            letter-spacing: -1px;
        }

        .hero-panel {
            border-radius: 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# SESSION STATE
# ==================================================
if "query" not in st.session_state:
    st.session_state.query = ""

if "active_query" not in st.session_state:
    st.session_state.active_query = ""

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

# ==================================================
# HERO SECTION
# ==================================================
st.markdown(
    """
    <div class="hero-panel">
        <div class="eyebrow">THE ART OF PERSONAL FRAGRANCE</div>
        <div class="hero-title">
            Find your<br><span>signature scent.</span>
        </div>
        <div class="hero-description">
            Tell us how you want to feel, where you're going, or what
            you love. Your AI fragrance consultant will discover scents
            that fit your mood, style and budget.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================
# SUGGESTED PROMPTS
# ==================================================
st.markdown(
    '<div class="section-label">01 / DISCOVER YOUR MOOD</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">Where should we begin?</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted-copy">Choose a starting point or describe '
    'your ideal fragrance in your own words.</p>',
    unsafe_allow_html=True,
)

prompt_columns = st.columns(3)

for i, (label, prompt) in enumerate(SUGGESTED_PROMPTS):
    with prompt_columns[i % 3]:
        if st.button(label, key=f"prompt_{i}", use_container_width=True):
            st.session_state.query = prompt
            st.session_state.active_query = prompt
            st.rerun()

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ==================================================
# CONSULTATION FORM
# ==================================================
st.markdown(
    '<div class="section-label">02 / YOUR FRAGRANCE BRIEF</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-heading">Describe your next obsession.</div>',
    unsafe_allow_html=True,
)

with st.form("fragrance_form"):
    query = st.text_area(
        "Describe your ideal fragrance",
        key="query",
        height=120,
        placeholder=(
            "e.g. A seductive woody fragrance for date night, "
            "not too sweet, suitable for winter evenings."
        ),
        label_visibility="collapsed",
    )

    left, right = st.columns([1, 1])

    with left:
        top_k = st.slider(
            "Number of fragrances to discover",
            min_value=3,
            max_value=10,
            value=st.session_state.top_k,
        )

    with right:
        st.markdown(
            """
            <div style="padding-top:13px;color:#a5a096;
                        font-size:0.82rem;line-height:1.8;">
                <span style="color:#d6b77a;">✦</span> Mood matching<br>
                <span style="color:#d6b77a;">✦</span> Budget-aware discovery<br>
                <span style="color:#d6b77a;">✦</span> Fragrance profile matching
            </div>
            """,
            unsafe_allow_html=True,
        )

    submitted = st.form_submit_button(
        "✦  Find my signature scent",
        type="primary",
        use_container_width=True,
    )

if submitted:
    st.session_state.active_query = query.strip()
    st.session_state.top_k = top_k

# ==================================================
# DISPLAY HELPERS
# ==================================================
def display_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()




def feature_chips(value):
    if not isinstance(value, str) or not value.strip():
        return ""

    if value == "No direct feature match":
        return ""

    features = [
        item.strip()
        for item in value.replace("|", ",").split(",")
        if item.strip()
    ][:8]

    return "".join(
        f'<span class="feature-chip">{feature}</span>'
        for feature in features
    )


def metric_box(label, value):
    return f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """

# ==================================================
# RECOMMENDATIONS
# ==================================================
active_query = st.session_state.active_query

if active_query:
    st.divider()

    st.markdown(
        '<div class="section-label">03 / YOUR PERSONAL EDIT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-heading">Curated for your taste.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="muted-copy">Your brief: “{active_query}”</p>',
        unsafe_allow_html=True,
    )

    try:
        with st.spinner("Exploring fragrance profiles..."):
            results = cached_recommendations(
                active_query,
                st.session_state.top_k,
            )

        if results is None or results.empty:
            st.warning(
                "No close matches found. Try a broader description "
                "or relax your budget or seasonal preferences."
            )

        else:
            st.markdown(
                f"""
                <div style="margin-bottom:22px;">
                    <span style="color:#d6b77a;font-size:1.2rem;">✦</span>
                    <strong style="color:#ead5a4;">
                        {len(results)} fragrances discovered
                    </strong>
                    <div style="font-size:0.82rem;color:#a5a096;margin-top:5px;">
                        Ranked by how closely they match your brief.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for rank, (_, row) in enumerate(results.iterrows(), 1):
                name = display_value(row.get("name")) or "Unnamed fragrance"
                brand = display_value(row.get("brand")) or "Unknown brand"
                description = display_value(row.get("description"))
                matched = display_value(row.get("matched_features"))
                season = display_value(row.get("season_estimate"))
                sweetness = display_value(row.get("sweetness_estimate"))
                score = row.get("final_score")

                with st.container(border=True):
                    header_left, header_right = st.columns([5, 2])

                    with header_left:
                        st.markdown(
                            f'<div class="result-kicker">YOUR MATCH #{rank:02d}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="perfume-name">{name}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="brand-name">{brand}</div>',
                            unsafe_allow_html=True,
                        )

                    
                    if matched and matched != "No direct feature match":
                        st.markdown(
                            '<div class="section-label">WHY IT MATCHES</div>',
                            unsafe_allow_html=True,
                        )
                        chips = feature_chips(matched)
                        if chips:
                            st.markdown(chips, unsafe_allow_html=True)
                    else:
                        st.markdown(
                            '<p class="muted-copy">This fragrance ranked well '
                            'by text similarity, although no direct characteristic '
                            'match was identified.</p>',
                            unsafe_allow_html=True,
                        )

                    metric_left, metric_right = st.columns(2)

                    with metric_left:
                        st.markdown(
                            metric_box(
                                "Estimated season",
                                season.title() if season else "Not available",
                            ),
                            unsafe_allow_html=True,
                        )

                    with metric_right:
                        sweetness_text = (
                            f"{sweetness} / 4"
                            if sweetness
                            else "Not available"
                        )
                        st.markdown(
                            metric_box(
                                "Estimated sweetness",
                                sweetness_text,
                            ),
                            unsafe_allow_html=True,
                        )

                    if description:
                        with st.expander("Explore fragrance details"):
                            st.write(description)

                    try:
                        if pd.notna(score):
                            st.caption(
                                f"Internal ranking score: {float(score):.3f} "
                                "· Not a probability or customer rating."
                            )
                    except (ValueError, TypeError):
                        pass

            st.info(
                "Fragrance impressions are subjective. Season and sweetness "
                "are estimates based on catalogue information."
            )

    except Exception as exc:
        st.error("We couldn't generate recommendations.")
        with st.expander("Technical details"):
            st.code(str(exc))
        st.info(
            "Check the catalogue, dependencies and recommender configuration "
            "if the issue continues."
        )

else:
    st.markdown(
        """
        <div style="border:1px solid rgba(214,183,122,0.18);
                    background:#171715;border-radius:18px;
                    padding:28px;margin-top:20px;">
            <div style="font-size:1.7rem;margin-bottom:12px;">✧</div>
            <div style="font-family:'Playfair Display',Georgia,serif;
                        font-size:1.45rem;color:#f5f0e7;margin-bottom:10px;">
                Start with a feeling, not a perfume name.
            </div>
            <div style="color:#a5a096;line-height:1.9;">
                “Clean and fresh for Mumbai's heat.”<br>
                “Woody, mysterious and memorable for date night.”<br>
                “Elegant and subtle for the office, under ₹3,000.”<br>
                “Warm vanilla, but not overwhelmingly sweet.”
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==================================================
# FOOTER
# ==================================================
st.divider()

st.markdown(
    """
    <div class="footer">
        <span style="color:#d6b77a;letter-spacing:3px;">SCENT SENSE</span>
        <br>
        An experimental AI fragrance discovery studio.
        <br>
        Recommendations depend on the available catalogue and model estimates.
    </div>
    """,
    unsafe_allow_html=True,
)
