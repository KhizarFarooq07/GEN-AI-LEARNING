"""
Streamlit UI for LinkedIn Content Curation System.

Requirements:
- Input box for topic
- Display final LinkedIn post
- Show generated image
- Debug panel with:
  - Planner plan
  - Tools executed
  - Execution order (parallel/sequential)

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"
st.set_page_config(
    page_title="LinkedIn Content Curation",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .execution-plan {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .step-box {
        background-color: white;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        border-radius: 0.25rem;
    }
    
    .step-completed {
        border-left-color: #2ca02c;
    }
    
    .step-failed {
        border-left-color: #d62728;
    }
    
    .post-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    
    .post-headline {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #0066cc;
    }
    
    .post-body {
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        color: #333;
    }
    
    .post-hashtags {
        color: #0066cc;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .post-cta {
        background-color: #0066cc;
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.25rem;
        display: inline-block;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def execute_curation(topic: str) -> Optional[dict]:
    """Execute content curation via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/execute",
            json={"topic": topic},
            timeout=600  # 10 minute timeout for image generation
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Image generation may be taking a while.")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def format_execution_plan(plan: dict) -> dict:
    """Format execution plan for display."""
    steps_by_id = {}
    for step in plan.get("steps", []):
        steps_by_id[step["step_id"]] = step
    
    # Group steps by parallelization
    parallel_groups = []
    processed = set()
    
    for step in plan.get("steps", []):
        if step["step_id"] in processed:
            continue
        
        # Find all steps with same dependencies (can run in parallel)
        group = [step]
        for other_step in plan.get("steps", []):
            if (other_step["step_id"] not in processed and 
                other_step["step_id"] != step["step_id"] and
                other_step.get("depends_on") == step.get("depends_on")):
                group.append(other_step)
        
        parallel_groups.append(group)
        for s in group:
            processed.add(s["step_id"])
    
    return {
        "total_steps": len(plan.get("steps", [])),
        "parallel_groups": parallel_groups,
        "all_steps": plan.get("steps", [])
    }


def display_step(step: dict):
    """Display a single execution step."""
    status_icon = {
        "completed": "✅",
        "failed": "❌",
        "running": "⏳",
        "pending": "⏭️"
    }.get(step.get("status"), "❓")
    
    depends = step.get("depends_on", [])
    depends_str = f"depends on steps {depends}" if depends else "independent (parallel)"
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.write(status_icon)
    with col2:
        st.write(f"**Step {step['step_id']}: {step['tool'].upper()}**  \n{depends_str}")
    with col3:
        exec_time = step.get("execution_time", 0)
        st.write(f"{exec_time:.2f}s")


# ============================================================================
# MAIN UI
# ============================================================================

def main():
    # Header
    st.markdown("# 📝 LinkedIn Content Curation System")
    st.markdown("*Agentic system for generating professional LinkedIn posts with images*")
    st.divider()
    
    # Check API connection
    if not check_api_health():
        st.error(
            "⚠️ **API not running!**\n\n"
            "Please start the API server with:\n"
            "`python3 api.py`"
        )
        st.stop()
    
    st.success("✅ Connected to API")
    st.divider()
    
    
    # Main content area
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        st.markdown("### 📥 Input")
        
        # Topic input
        if "input_topic" not in st.session_state:
            st.session_state.input_topic = ""
        
        topic = st.text_area(
            "Enter your LinkedIn post topic:",
            value=st.session_state.input_topic,
            height=100,
            placeholder="e.g., Recent trends in AI for cloud infrastructure...",
            key="topic_input"
        )
        st.session_state.input_topic = topic
        
        col_submit, col_clear = st.columns(2)
        with col_submit:
            submit_button = st.button(
                "🚀 Generate Content",
                type="primary",
                key="submit_btn",
                use_container_width=True
            )
        with col_clear:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    
    with col2:
        st.markdown("### ℹ️ Info")
        st.info(
            "**System Components:**\n\n"
            "1. 🔍 Search\n"
            "2. 📊 Summarize\n"
            "3. 🎨 Image Gen\n"
            "4. ✍️ Generate\n"
            "5. ✏️ Edit",
            icon="ℹ️"
        )
    
    st.divider()
    
    # Process submission
    if submit_button and topic.strip():
        with st.spinner("🔄 Processing your request..."):
            result = execute_curation(topic)
        
        if result:
            st.session_state.last_result = result
    
    # Display results if available
    if "last_result" in st.session_state:
        result = st.session_state.last_result
        
        # Create tabs for different views
        tab_post, tab_image, tab_debug, tab_raw = st.tabs(
            ["📄 Post", "🖼️ Image", "🔧 Debug", "📋 Raw Data"]
        )
        
        # ============================================================================
        # TAB 1: POST
        # ============================================================================
        with tab_post:
            st.markdown("### Generated LinkedIn Post")
            
            if result.get("post"):
                post = result["post"]
                
                # Display post in LinkedIn-style format
                st.markdown(f"""
                <div class="post-box">
                    <div class="post-headline">{post.get('headline', '')}</div>
                    <div class="post-body">{post.get('body', '')}</div>
                    <div class="post-hashtags">{' '.join([f'#{tag}' for tag in post.get('hashtags', [])])}</div>
                    <div class="post-cta">{post.get('call_to_action', 'Join the conversation')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Copy button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    post_text = f"{post.get('headline', '')}\n\n{post.get('body', '')}\n\n{' '.join([f'#{tag}' for tag in post.get('hashtags', [])])}"
                    st.download_button(
                        "📋 Copy Post",
                        post_text,
                        file_name="linkedin_post.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("No post generated")
        
        # ============================================================================
        # TAB 2: IMAGE
        # ============================================================================
        with tab_image:
            st.markdown("### Generated Image")
            
            if result.get("image_url"):
                image_url = result["image_url"]
                
                if image_url.startswith("file://"):
                    # Local file
                    file_path = image_url.replace("file://", "").split("?")[0]
                    try:
                        st.image(file_path, use_column_width=True, caption="Generated with Openjourney + Accelerators")
                    except:
                        st.error(f"Could not load image from: {file_path}")
                else:
                    # URL
                    st.image(image_url, use_column_width=True, caption="Generated Image")
            else:
                st.warning("No image generated")
        
        # ============================================================================
        # TAB 3: DEBUG
        # ============================================================================
        with tab_debug:
            st.markdown("### 🔧 Execution Debug Info")
            
            plan = result.get("plan", {})
            formatted_plan = format_execution_plan(plan)
            
            # Execution Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Steps", formatted_plan["total_steps"])
            with col2:
                st.metric("Total Time", f"{result.get('total_time', 0):.2f}s")
            with col3:
                search_results = result.get("search_results", [])
                st.metric("Search Results", len(search_results))
            with col4:
                st.metric("Parallel Groups", len(formatted_plan["parallel_groups"]))
            
            st.divider()
            
            # Execution Plan
            st.markdown("#### 📋 Execution Plan & Dependencies")
            
            for i, group in enumerate(formatted_plan["parallel_groups"], 1):
                if len(group) > 1:
                    st.markdown(f"**⚡ Parallel Execution {i}** (steps run simultaneously):")
                    cols = st.columns(len(group))
                    for col, step in zip(cols, group):
                        with col:
                            st.markdown(f"""
                            <div class="step-box step-{step.get('status')}">
                                <b>Step {step['step_id']}</b><br>
                                {step['tool'].upper()}<br>
                                <small>{step.get('status').upper()}</small><br>
                                <small>{step.get('execution_time', 0):.2f}s</small>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    step = group[0]
                    depends = step.get("depends_on", [])
                    if depends:
                        st.markdown(f"**⏭️ Sequential Execution {i}** (waits for steps {depends}):")
                    else:
                        st.markdown(f"**▶️ Execution {i}** (independent):")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        status_icon = {
                            "completed": "✅",
                            "failed": "❌",
                            "running": "⏳",
                            "pending": "⏭️"
                        }.get(step.get("status"), "❓")
                        st.write(f"{status_icon} Step {step['step_id']}")
                    with col2:
                        st.write(f"**{step['tool'].upper()}** - {step.get('execution_time', 0):.2f}s")
            
            st.divider()
            
            # Tools Executed
            st.markdown("#### 🔨 Tools Executed")
            
            tools_executed = []
            for step in formatted_plan["all_steps"]:
                if step.get("status") == "completed":
                    tools_executed.append({
                        "step": step["step_id"],
                        "tool": step["tool"].upper(),
                        "time": f"{step.get('execution_time', 0):.2f}s"
                    })
            
            if tools_executed:
                tools_df = st.dataframe(tools_executed, use_container_width=True)
            else:
                st.info("No tools executed yet")
            
            st.divider()
            
            # Execution Log
            st.markdown("#### 📝 Execution Log")
            
            execution_log = result.get("execution_log", [])
            log_text = "\n".join(execution_log)
            
            st.code(log_text, language="text")
        
        # ============================================================================
        # TAB 4: RAW DATA
        # ============================================================================
        with tab_raw:
            st.markdown("### 📋 Raw API Response")
            
            # Search results
            if result.get("search_results"):
                with st.expander("🔍 Search Results", expanded=False):
                    for i, result_item in enumerate(result["search_results"], 1):
                        st.markdown(f"**{i}. {result_item.get('title', 'N/A')}**")
                        st.markdown(f"Source: {result_item.get('source', 'N/A')}")
                        st.markdown(f"{result_item.get('snippet', '')}")
                        st.divider()
            
            # Full JSON
            with st.expander("📦 Full API Response (JSON)", expanded=False):
                st.json(result)
    
    st.divider()
    
    # Footer
    st.markdown("""
    ---
    **LinkedIn Content Curation System** | Powered by FastAPI + Streamlit  
    🤖 Agents: Planner • Searcher • Summarizer • ImageGen • Generator • Editor
    """)


if __name__ == "__main__":
    main()
