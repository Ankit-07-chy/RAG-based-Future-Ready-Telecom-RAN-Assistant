import sys
from pathlib import Path

# Ensure the repository root is on the Python import path when Streamlit runs this file.
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from src.rag_chain import RAGChain
from src.evaluator import evaluate_single_response
from datetime import datetime

@st.cache_resource
def load_rag_chain():
    return RAGChain()


def main():
    st.set_page_config(
        page_title="Telecom RAG Query Interface",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Telecom RAG — FastAPI + Streamlit Interface")
    st.markdown(
        """
        **FastAPI + Streamlit Query Interface** for Telecom RAG.

        - Enter a telecom question.
        - The system retrieves the most relevant document chunks.
        - Optionally generate a natural answer from retrieved content.
        """
    )

    with st.sidebar:
        st.header("Query settings")
        k = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=5)
        use_hybrid = st.checkbox("Use hybrid retrieval", value=True)
        use_llm = st.checkbox("Generate answer text from retrieved context", value=False)
        run_eval = st.checkbox("Evaluate this response (faithfulness metrics)", value=True)
        reference_answer = st.text_area(
            "Optional ground-truth answer (for accuracy scoring)",
            height=80,
            placeholder="Paste expected answer to score accuracy during evaluation.",
        )
        refresh = st.button("Reload knowledge base")

    query = st.text_area("Enter your query", value="What is MIMO in 5G?", height=140)

    if refresh:
        st.cache_resource.clear()
        st.success("Knowledge base reload requested. Refresh the page if needed.")

    if st.button("Run query"):
        if not query.strip():
            st.warning("Please enter a query before running.")
        else:
            placeholder = st.empty()
            placeholder.info("Loading the RAG knowledge base and retrieving results...")
            try:
                chain = load_rag_chain()
                retrieved_docs = chain.retrieve(query=query, k=k, use_hybrid=use_hybrid)
                response = chain.process_query(
                    query=query,
                    k=k,
                    use_hybrid=use_hybrid,
                    use_llm=use_llm,
                )

                placeholder.success("Query completed")

                st.subheader("Answer")
                st.write(response.answer)

                st.subheader("Reasoning / Notes")
                st.write(response.reasoning)

                st.subheader("Retrieved Sources")
                if response.sources:
                    st.table(response.sources)
                else:
                    st.write("No sources were retrieved.")

                if run_eval:
                    st.subheader("Response Evaluation")
                    eval_metrics = evaluate_single_response(
                        query=query,
                        answer=response.answer,
                        reasoning=response.reasoning,
                        retrieved_docs=retrieved_docs,
                        reference_answer=reference_answer.strip() or None,
                    )
                    st.json(eval_metrics)

                st.markdown(
                    f"**Query type:** {response.query_type.value}  \n"
                    f"**Retrieved chunks:** {response.retrieved_chunks}  \n"
                    f"**Confidence:** {response.confidence:.2f}  \n"
                    f"**Timestamp:** {datetime.now().isoformat()}"
                )

            except Exception as exc:
                placeholder.error(f"Query failed: {exc}")

    st.sidebar.markdown("---")
    st.sidebar.write(
        "**Usage**: build the vector store first with `python main.py --mode pipeline`, then run this app with `streamlit run src/streamlit_app.py`."
    )


if __name__ == "__main__":
    main()
