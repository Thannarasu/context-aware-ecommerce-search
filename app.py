
import streamlit as st

from src.block2_intent.gemini_query_understanding import analyze_query


st.set_page_config(
    page_title="Context-Aware E-commerce Search",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Context-Aware E-commerce Search")
st.caption(
    "M.Tech research prototype | Gemini-powered query chunking "
    "and intent classification"
)

st.markdown(
    """
Enter a shopping query in natural language. The system will extract
semantic chunks, classify the primary intent, and identify missing
information before product retrieval.
"""
)

examples = [
    "I need a lightweight waterproof backpack for weekend hiking under $80.",
    "airpods pro case without keychain",
    "navy blue silk fabric",
    "I need something timeless that works with any outfit, "
    "but not too thick because I get warm easily.",
]

selected = st.selectbox(
    "Try an example query",
    ["Enter my own query"] + examples,
)

if selected == "Enter my own query":
    default_query = ""
else:
    default_query = selected

query = st.text_area(
    "Shopping query",
    value=default_query,
    height=100,
    placeholder="Describe the product you are looking for...",
)

run = st.button("Analyze query", type="primary", use_container_width=True)

if run:
    if not query.strip():
        st.warning("Please enter a shopping query.")
    else:
        try:
            with st.spinner("Gemini is analyzing your query..."):
                result = analyze_query(query)

            st.success("Query analysis completed.")

            st.subheader("1. Primary intent")
            st.metric("Intent", result["primary_intent"])
            st.write(result["intent_explanation"])

            left, right = st.columns(2)

            with left:
                st.subheader("2. Extracted semantic chunks")
                if result["chunks"]:
                    st.dataframe(
                        [
                            {
                                "Chunk text": chunk["text"],
                                "Label": chunk["label"],
                                "Explanation": chunk["explanation"],
                            }
                            for chunk in result["chunks"]
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No reliable chunks were extracted.")

            with right:
                st.subheader("3. Clarification")
                if result["needs_clarification"]:
                    st.warning("More information may be required.")
                else:
                    st.success("No clarification was flagged.")

                if result["missing_information"]:
                    st.write("Potentially missing information:")
                    for item in result["missing_information"]:
                        st.markdown(f"- {item}")
                else:
                    st.write("No missing information was identified.")

            with st.expander("View raw structured response"):
                st.json(result)

            st.download_button(
                "Download analysis as JSON",
                data=__import__("json").dumps(
                    {"query": query, **result},
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="query_analysis.json",
                mime="application/json",
            )

        except Exception as exc:
            st.error(
                "Query analysis failed. Check your API key, model access, "
                "network connection and error details."
            )
            st.exception(exc)

st.divider()
st.caption(
    "Research prototype: predictions are model-generated and require "
    "evaluation against manually reviewed annotations."
)
