import streamlit as st
import asyncio
from dotenv import load_dotenv
import sys
import os

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.orchestrator.workflow import build_evaluator_workflow, GraphState

# Load environment variables
load_dotenv()

st.set_page_config(page_title="ArXiv Paper Evaluator", page_icon="📄", layout="wide")

st.title("📄 Agentic Research Paper Evaluator")
st.markdown("Enter an arXiv URL to evaluate the paper for consistency, grammar, novelty, and facts.")

# Input: Text box for the arXiv URL
with st.container(border=True):
    url_input = st.text_input("ArXiv URL", placeholder="https://arxiv.org/abs/xxxx.xxxxx")
    submit_button = st.button("Evaluate Paper", type="primary")

if submit_button:
    if not url_input:
        st.warning("Please enter an ArXiv URL.")
    else:
        # Initialize the LangGraph app
        app = build_evaluator_workflow()
        
        initial_state = GraphState(
            url=url_input,
            arxiv_id="",
            raw_text=None,
            chunks=[],
            consistency_eval=None,
            grammar_eval=None,
            novelty_eval=None,
            fact_check_eval=None,
            authenticity_eval=None,
            final_report=None,
            errors=[]
        )

        async def run_workflow():
            cumulative_state = dict(initial_state)
            
            # Progress: Use st.status to show LangGraph nodes as they execute
            with st.status("Evaluating Paper...", expanded=True) as status:
                try:
                    # Stream updates from the graph
                    async for event in app.astream(initial_state):
                        for node_name, node_update in event.items():
                            st.write(f"⚙️ Executed node: **{node_name.replace('_', ' ').capitalize()}**")
                            # Merge updates into cumulative state
                            cumulative_state.update(node_update)
                            
                    status.update(label="Evaluation Complete!", state="complete", expanded=False)
                    return cumulative_state
                except Exception as e:
                    status.update(label=f"Evaluation failed: {e}", state="error", expanded=True)
                    return None

        # Run the async workflow
        final_state = asyncio.run(run_workflow())
        
        if final_state:
            if final_state.get("errors"):
                st.error("Errors encountered during evaluation:")
                for err in final_state["errors"]:
                    st.error(err)
            else:
                st.success("Analysis complete!")
                
                st.divider()
                
                # Output: Multi-column dashboard showing the scores
                st.subheader("Key Metrics")
                
                # Extract scores safely
                consistency_eval = final_state.get("consistency_eval") or {}
                grammar_eval = final_state.get("grammar_eval") or {}
                novelty_eval = final_state.get("novelty_eval") or {}
                fact_check_eval = final_state.get("fact_check_eval") or {}
                authenticity_eval = final_state.get("authenticity_eval") or {}

                consistency_score = consistency_eval.get("consistency_score", "N/A")
                grammar_score = grammar_eval.get("grammar_score", "N/A")
                novelty_score = novelty_eval.get("novelty_score", "N/A")
                fact_score = fact_check_eval.get("fact_score", "N/A")
                
                # Depending on how fabrication risk is output in your workflow
                # Fact check node has fabrication_risk_score
                # Authenticity node has fabrication_probability
                risk_score = fact_check_eval.get("fabrication_risk_score", "N/A")
                authenticity_prob = authenticity_eval.get("fabrication_probability", "N/A")
                
                final_fabrication_risk = authenticity_prob if authenticity_prob != "N/A" else risk_score

                # Layout using columns with border=True cards
                cols = st.columns(4)
                
                with cols[0]:
                    st.metric(label="Consistency", value=f"{consistency_score}/10" if consistency_score != "N/A" else "N/A", border=True)
                with cols[1]:
                    st.metric(label="Grammar", value=f"{grammar_score}/10" if grammar_score != "N/A" else "N/A", border=True)
                with cols[2]:
                    st.metric(label="Novelty", value=f"{novelty_score}/10" if novelty_score != "N/A" else "N/A", border=True)
                with cols[3]:
                    st.metric(label="Factuality", value=f"{fact_score}/10" if fact_score != "N/A" else "N/A", border=True)
                
                # Highlight fabrication risk
                with st.container(border=True):
                    st.metric(label="Fabrication Risk Score", value=f"{final_fabrication_risk}%" if final_fabrication_risk != "N/A" else "N/A")

                st.divider()
                
                # Output: Full Markdown area for the 'Judgement Report'
                st.subheader("Judgement Report")
                
                report_content = final_state.get("final_report")
                
                if report_content:
                    with st.container(border=True):
                        st.markdown(report_content)
                else:
                    # Fallback to reading the file if it wasn't captured in state
                    arxiv_id = final_state.get("arxiv_id")
                    if arxiv_id:
                        report_path = f"reports/Judgement_Report_{arxiv_id}.md"
                        if os.path.exists(report_path):
                            with open(report_path, "r", encoding="utf-8") as f:
                                with st.container(border=True):
                                    st.markdown(f.read())
                        else:
                            st.info("No report content found.")
                    else:
                        st.info("No report content found.")
