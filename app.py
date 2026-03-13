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
            
            # Define node order for progress bar
            nodes = ["scrape", "decompose", "consistency", "grammar", "novelty", "fact_checker", "authenticity", "report"]
            total_nodes = len(nodes)
            
            # Progress: Use st.status and progress bar
            with st.status("Initializing Evaluation...", expanded=True) as status:
                progress_bar = st.progress(0, text="Preparing...")
                time_display = st.empty()
                start_time = asyncio.get_event_loop().time()
                
                # State to control the continuous timer
                tracking_state = {"is_running": True}
                
                async def update_timer():
                    while tracking_state["is_running"]:
                        elapsed = asyncio.get_event_loop().time() - start_time
                        time_display.markdown(f"⏱️ **Time Spent:** {elapsed:.1f}s")
                        await asyncio.sleep(0.1)

                # Start the continuous timer task
                timer_task = asyncio.create_task(update_timer())
                
                try:
                    # Stream updates from the graph
                    node_count = 0
                    async for event in app.astream(initial_state):
                        for node_name, node_update in event.items():
                            node_count += 1
                            progress = min(node_count / total_nodes, 1.0)
                            
                            # Update progress bar
                            progress_bar.progress(progress, text=f"Executing Node: **{node_name.replace('_', ' ').capitalize()}**...")
                            
                            # Merge updates into cumulative state
                            cumulative_state.update(node_update)
                            
                    # Stop the timer task
                    tracking_state["is_running"] = False
                    await timer_task
                    
                    status.update(label="Evaluation Complete!", state="complete", expanded=False)
                    progress_bar.empty()
                    time_display.empty()
                    st.info(f"✅ Total Time Taken: {asyncio.get_event_loop().time() - start_time:.1f} seconds")
                    return cumulative_state
                except Exception as e:
                    tracking_state["is_running"] = False
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
                
                # --- HIGHLIGHT: VERDICT & OVERALL SCORE ---
                verdict = final_state.get("final_verdict", "N/A")
                overall_score = final_state.get("overall_score", "N/A")
                
                st.divider()
                
                v_col1, v_col2 = st.columns([2, 1])
                with v_col1:
                    if verdict == "Accept":
                        st.success(f"### Recommendation: {verdict}")
                    elif verdict == "Minor Revisions":
                        st.info(f"### Recommendation: {verdict}")
                    elif verdict == "Major Revisions":
                        st.warning(f"### Recommendation: {verdict}")
                    else:
                        st.error(f"### Recommendation: {verdict}")
                
                with v_col2:
                    st.metric("Overall Quality Score", f"{overall_score:.1f}/100" if isinstance(overall_score, (int, float)) else "N/A", border=True)

                st.divider()
                
                # --- DETAILED SCORES ---
                st.subheader("Detailed Metrics")
                
                # Extract scores safely
                consistency_eval = final_state.get("consistency_eval") or {}
                grammar_eval = final_state.get("grammar_eval") or {}
                novelty_eval = final_state.get("novelty_eval") or {}
                fact_check_eval = final_state.get("fact_check_eval") or {}
                authenticity_eval = final_state.get("authenticity_eval") or {}

                consistency_score = consistency_eval.get("consistency_score", "N/A")
                grammar_rating = grammar_eval.get("grammar_rating", "N/A")
                novelty_index = novelty_eval.get("novelty_index", "N/A")
                accuracy_score = fact_check_eval.get("accuracy_score", "N/A")
                
                risk_score = fact_check_eval.get("fabrication_risk_score", "N/A")
                authenticity_prob = authenticity_eval.get("fabrication_probability", "N/A")
                final_fabrication_risk = authenticity_prob if authenticity_prob != "N/A" else risk_score

                # Layout using columns
                cols = st.columns(4)
                with cols[0]:
                    st.metric(label="Consistency", value=f"{consistency_score}/100" if consistency_score != "N/A" else "N/A", border=True)
                with cols[1]:
                    st.metric(label="Grammar Rating", value=str(grammar_rating), border=True)
                with cols[2]:
                    st.metric(label="Accuracy Score", value=f"{accuracy_score}/100" if accuracy_score != "N/A" else "N/A", border=True)
                with cols[3]:
                    st.metric(label="Fabrication Risk", value=f"{final_fabrication_risk}%" if final_fabrication_risk != "N/A" else "N/A", border=True)
                
                # Novelty Index & Executive Summary
                st.markdown(f"**Novelty Index:** {novelty_index}")
                
                with st.expander("View Executive Summary"):
                    # Find summary from report or state
                    st.write(final_state.get("consistency_eval", {}).get("summary", "No summary available."))

                st.divider()
                
                # --- FACT CHECK LOG ---
                st.subheader("Fact Check Log")
                claims = fact_check_eval.get("claims_evaluated", [])
                if claims:
                    for claim in claims:
                        with st.container(border=True):
                            c1, c2 = st.columns([0.1, 0.9])
                            icon = "✅" if claim['verdict'] == "supported" else "❌" if claim['verdict'] == "contradicted" else "⚠️"
                            c1.markdown(f"### {icon}")
                            c2.markdown(f"**Claim:** {claim['claim']}")
                            c2.markdown(f"*Verdict:* {claim['verdict'].capitalize()} | *Confidence:* {claim['confidence'].capitalize()}")
                            c2.info(f"**Evidence:** {claim['evidence']}")
                else:
                    st.info("No claims were extracted for fact-checking.")

                st.divider()
                
                # --- FULL REPORT ---
                st.subheader("Complete Judgement Report")
                report_content = final_state.get("final_report")
                if report_content:
                    st.markdown(report_content)
                else:
                    st.info("No detailed report content found.")
