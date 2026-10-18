import pandas as pd
import streamlit as st

from src.hybrid_model import recommend_hybrid


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Scent Sense | AI Fragrance Consultant",
    page_icon="🌸",
    layout="wide",
)

# ---------------------------------------------------------
# Suggested prompts
# ---------------------------------------------------------
SUGGESTIONS = [
    (
        "🌙 Date night",
        "I have a date tonight. I want something seductive, masculine, "
        "and memorable for a first impression.",
    ),
    (
        "🍋 Fresh and clean",
        "I want a fresh, clean fragrance with citrus and green notes "
        "for everyday use.",
    ),
    (
        "🖤 Mysterious",
        "I want a mysterious, sophisticated fragrance with woody, "
        "smoky, and spicy characteristics.",
    ),
    (
        "🌹 Romantic",
        "I want a romantic fragrance with floral, vanilla, sweet, "
        "and musky characteristics.",
    ),
    (
        "☀️ Summer",
        "I want a refreshing fragrance for hot summer days with "
        "citrus and aquatic characteristics.",
    ),
    (
        "✨ Elegant",
        "I want an elegant, premium-smelling fragrance with woody, "
        "amber, and musky characteristics.",
    ),
]


def apply_prompt(prompt):
    """Put a suggested prompt into the input and run the recommendation."""
    st.session_state["query_input"] = prompt
    st.session_state["active_query"] = prompt


def is_present(value):
    """Safely check whether a value is available."""
    if value is None:
        return False

    try:
        return bool(pd.notna(value))
    except (TypeError, ValueError):
        return False


def display_value(value, fallback="Not available"):
    """Convert missing values into readable text."""
    if not is_present(value):
        return fallback

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return fallback

    return text


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🌸 Scent Sense")
st.subheader("Your personal AI fragrance consultant")

st.markdown(
    """
    **Tell me about the fragrance you're looking for.**

    It could be for a date, a college day, a special occasion, or simply
    the impression you want to create. I'll use your description to find
    fragrances that best match your preferences.
    """
)

st.divider()

# ---------------------------------------------------------
# Suggested prompt buttons
# ---------------------------------------------------------
st.markdown("#### Not sure where to start?")

st.caption("Choose a situation to get started, or write your own request below.")

columns = st.columns(3)

for index, (label, prompt) in enumerate(SUGGESTIONS):
    with columns[index % 3]:
        st.button(
            label,
            key=f"suggestion_{index}",
            use_container_width=True,
            on_click=apply_prompt,
            args=(prompt,),
        )

st.write("")

# ---------------------------------------------------------
# Natural-language input
# ---------------------------------------------------------
st.markdown("#### Tell me what you're looking for")

st.text_input(
    "Describe your ideal fragrance",
    key="query_input",
    placeholder=(
        "e.g. I want a seductive fragrance for a date tonight "
        "under ₹3,000..."
    ),
    label_visibility="collapsed",
)

if st.button(
    "✨ Find My Fragrance",
    type="primary",
    use_container_width=True,
):
    st.session_state["active_query"] = st.session_state.get(
        "query_input", ""
    ).strip()

st.caption(
    "You can mention the occasion, mood, fragrance characteristics, "
    "season, or budget in your own words."
)

st.divider()

# ---------------------------------------------------------
# Generate recommendations
# ---------------------------------------------------------
active_query = st.session_state.get("active_query", "")

if active_query:
    st.markdown("### 💬 Your fragrance consultation")

    st.markdown(f"**You said:** {active_query}")

    st.write(
        "I'm looking for fragrances that match your description "
        "and the characteristics you've mentioned."
    )

    with st.spinner(
        "🌸 Analysing fragrance profiles and finding your matches..."
    ):
        try:
            recommendations = recommend_hybrid(
                active_query,
                top_k=5,
            )

            if recommendations.empty:
                st.warning(
                    "I couldn't find any fragrances matching all the "
                    "current filters. Try a broader description or "
                    "remove a restriction such as your budget."
                )

            else:
                st.success(
                    f"I found {len(recommendations)} fragrance matches "
                    "for you!"
                )

                for rank, (_, perfume) in enumerate(
                    recommendations.iterrows(),
                    start=1,
                ):
                    st.divider()

                    name = display_value(
                        perfume.get("name"),
                        "Unnamed fragrance",
                    )

                    brand = display_value(
                        perfume.get("brand"),
                        "Brand not listed",
                    )

                    matched = display_value(
                        perfume.get("matched_features"),
                        "Overall text similarity",
                    )

                    st.markdown(f"### {rank}. {name}")
                    st.markdown(f"**Brand:** {brand}")

                    # Recommendation explanation
                    st.markdown("**Why I'm recommending this**")

                    if matched != "No direct feature match":
                        st.write(
                            f"This perfume's available description "
                            f"matches these characteristics from your "
                            f"request: **{matched}**. These matches help "
                            f"explain why it ranked among your "
                            f"recommendations."
                        )
                    else:
                        st.write(
                            "This perfume ranked based on overall text "
                            "and semantic similarity to your request. "
                            "The available data did not identify a "
                            "clear direct characteristic match."
                        )

                    # Season estimate
                    season = perfume.get("season_estimate")

                    if is_present(season):
                        st.write(f"**Estimated season:** {season}")

                    # Sweetness estimate
                    sweetness = perfume.get("sweetness_estimate")

                    if is_present(sweetness):
                        try:
                            st.write(
                                f"**Estimated sweetness:** "
                                f"{float(sweetness):.0f}/4"
                            )
                        except (ValueError, TypeError):
                            pass

                    # Price, if actually available in the dataset
                    price = perfume.get("price_inr")

                    if is_present(price):
                        try:
                            st.write(
                                f"**Listed price:** "
                                f"₹{float(price):,.0f}"
                            )
                        except (ValueError, TypeError):
                            st.write(f"**Listed price:** {price}")

                    # Additional catalogue details
                    description = display_value(
                        perfume.get("description"),
                        "",
                    )

                    with st.expander("View fragrance details"):
                        if description:
                            st.write(description)
                        else:
                            st.write(
                                "No additional fragrance description "
                                "is available in the current catalogue."
                            )

                        st.caption(
                            "Fragrance characteristics and seasonal "
                            "estimates depend on the available catalogue "
                            "data and experimental heuristics."
                        )

                    # Internal ranking score
                    score = perfume.get("final_score")

                    if is_present(score):
                        try:
                            st.caption(
                                f"Internal ranking score: "
                                f"{float(score):.3f}"
                            )
                        except (ValueError, TypeError):
                            pass

                st.divider()

                st.markdown("#### Want a different recommendation?")

                st.write(
                    "Try changing the occasion, mood, fragrance "
                    "characteristics, or budget to explore other "
                    "possibilities."
                )

        except Exception as error:
            st.error("Something went wrong while finding fragrances.")
            st.code(str(error))
            st.info(
                "Check that your virtual environment is active, "
                "the required packages are installed, and the "
                "catalogue files are available."
            )

else:
    st.markdown("### 🌷 Your next favourite fragrance starts here")

    st.write(
        "Describe the impression you want to create, and Scent Sense "
        "will search the fragrance catalogue for suitable matches."
    )

    st.markdown(
        """
        **You could say:**

        - “I need a fragrance for a first date.”
        - “I want something fresh for Mumbai's hot weather.”
        - “Recommend something woody, spicy, and masculine.”
        - “I'd prefer something elegant and not too sweet.”
        """
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()

st.caption(
    "Scent Sense is an experimental AI fragrance recommendation system. "
    "Its ranking scores are not probabilities, and fragrance "
    "preferences are subjective."
)
