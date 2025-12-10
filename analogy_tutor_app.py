"""
Analogy Tutor - Learn Any Topic Through Your Interests (Multi-Model)

Streamlit app that teaches technical concepts using analogies grounded in your
personal interests. Supports:
- Gemini 2.0 Flash
- OpenAI gpt-4o
- Claude 2.1

API keys should be set in a .env file:
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
"""

import os
import random
from typing import Dict, Optional, List, Tuple

import streamlit as st
from dotenv import load_dotenv

# Model clients
import google.generativeai as genai
from openai import OpenAI
import anthropic


# ENV / CONFIG
load_dotenv()

# No default interests – user adds their own
DEFAULT_INTERESTS: List[str] = []

AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gpt-4o",
    "claude-2.1",
]

def save_key_to_env(key: str, value: str) -> None:
    """Append or update a key in the .env file and runtime environment."""
    env_path = ".env"

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    updated = False
    for i, line in enumerate(lines):
        if line.startswith(key + "="):
            lines[i] = f"{key}={value}\n"
            updated = True
            break

    if not updated:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    os.environ[key] = value

# API SETUP (clean UI)

def setup_apis():
    keys = {
        "gemini": os.getenv("GEMINI_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "claude": os.getenv("ANTHROPIC_API_KEY"),
    }

    model_to_key = {
        "gemini-2.0-flash": ("gemini", "GEMINI_API_KEY"),
        "gpt-4o": ("openai", "OPENAI_API_KEY"),
        "claude-2.1": ("claude", "ANTHROPIC_API_KEY"),
    }

    chosen_model = st.session_state.get("chosen_model", AVAILABLE_MODELS[0])
    model_key, env_var = model_to_key[chosen_model]
    current_key = keys[model_key]

    toggle_key = f"{env_var}_change_toggle"
    updated_flag = f"{env_var}_updated"

    # Initialize state keys
    st.session_state.setdefault(toggle_key, False)
    st.session_state.setdefault(updated_flag, False)

    st.sidebar.subheader("🔐 API Key for Selected Model")

    # 1) No API key set yet
    if not current_key:
        st.sidebar.warning(f"⚠️ {env_var} is missing for {chosen_model}.")

        with st.sidebar.form(f"{env_var}_form"):
            new_key = st.text_input(f"Enter {env_var}:", type="password")
            submitted = st.form_submit_button("✅ Set API Key")

        if submitted:
            if new_key:
                save_key_to_env(env_var, new_key)
                keys[model_key] = new_key
                st.sidebar.success(f"{env_var} saved successfully!")
                st.rerun()
            else:
                st.sidebar.error("Please enter a key before saving.")
        return keys

    # 2) API key already exists
    st.sidebar.success(f"{env_var} is already set.")

    if st.session_state[updated_flag]:
        st.sidebar.success(f"{env_var} updated successfully!")
        st.session_state[updated_flag] = False

    st.session_state[toggle_key] = st.sidebar.checkbox(
        "Change API Key", value=st.session_state[toggle_key]
    )

    if st.session_state[toggle_key]:
        with st.sidebar.form(f"{env_var}_change_form"):
            new_key = st.text_input(f"Enter new {env_var}:", type="password")
            submitted = st.form_submit_button("✅ Update API Key")

        if submitted:
            if new_key:
                save_key_to_env(env_var, new_key)
                keys[model_key] = new_key
                st.session_state[updated_flag] = True
                st.session_state[toggle_key] = False
                st.rerun()
            else:
                st.sidebar.error("Please enter a new key before updating.")

    # Configure model SDKs
    if keys["gemini"]:
        genai.configure(api_key=keys["gemini"])

    return keys

# WEIGHTED RANDOM PICKING
def weighted_random_choice(
    interests: List[str],
    weights: Dict[str, float],
    avoid_interest: Optional[str] = None,
) -> Tuple[str, str]:

    if not interests:
        raise ValueError("No interests available.")

    candidates: List[str] = []
    candidate_weights: List[float] = []

    for interest in interests:
        if avoid_interest == interest and len(interests) > 1:
            continue
        w = float(weights.get(interest, 1.0))
        candidates.append(interest)
        candidate_weights.append(max(w, 0.0))

    if not candidates:
        candidates = interests[:]
        candidate_weights = [weights.get(i, 1.0) for i in candidates]

    total = sum(candidate_weights)

    if total <= 0:
        chosen = random.choice(candidates)
        return chosen, "All weights were zero. Selected randomly."

    r = random.random() * total
    upto = 0.0
    chosen = candidates[-1]

    for interest, w in zip(candidates, candidate_weights):
        upto += w
        if r <= upto:
            chosen = interest
            break

    reason = "Weighted random selection."
    return chosen, reason


def pick_interest(topic: str, interests: List[str]):
    last_interest = st.session_state.get("last_interest_domain")
    weights = st.session_state.interest_weights

    chosen, reason = weighted_random_choice(interests, weights, avoid_interest=last_interest)
    st.session_state.last_interest_domain = chosen
    return chosen, reason


# ANALOGY GENERATION
def generate_analogy(topic: str, interest: str, model_name: str):
    prompt = f"""
You are an expert analogy tutor. Explain "{topic}" using the domain "{interest}".

Follow EXACTLY this structure:
1. One short reason why "{interest}" is a good analogy domain.
2. A 3-sentence analogy.
3. 5–8 bullet mappings (topic → analogy).
4. A self-check question + answer.
"""

    try:
        if model_name == "gemini-2.0-flash":
            m = genai.GenerativeModel("gemini-2.0-flash")
            return m.generate_content(prompt).text

        elif model_name == "gpt-4o":
            client = OpenAI()
            out = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            return out.choices[0].message.content

        elif model_name == "claude-2.1":
            client = anthropic.Anthropic()
            out = client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=800,
                prompt=prompt,
            )
            return out["completion"]

    except Exception as e:
        st.error(f"Error generating analogy: {e}")
        return None



# FOLLOW-UP GENERATION
def generate_followup(topic, interest, analogy_text, followup_q, model_name):
    prompt = f"""
Continue the SAME analogy domain: "{interest}".

Original analogy:
\"\"\"{analogy_text}\"\"\"

Learner follow-up question:
\"\"\"{followup_q}\"\"\"

Answer using the SAME analogy domain. Do not switch domains.
"""

    try:
        if model_name == "gemini-2.0-flash":
            m = genai.GenerativeModel("gemini-2.0-flash")
            return m.generate_content(prompt).text

        elif model_name == "gpt-4o":
            client = OpenAI()
            out = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            return out.choices[0].message.content

        elif model_name == "claude-2.1":
            client = anthropic.Anthropic()
            out = client.completions.create(
                model="claude-2.1",
                max_tokens_to_sample=600,
                prompt=prompt,
            )
            return out["completion"]

    except Exception as e:
        st.error(f"Error generating follow-up: {e}")
        return None



# SESSION STATE
def init_session_state():
    st.session_state.setdefault("user_interests", [])
    st.session_state.setdefault("interest_weights", {})
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("chosen_model", AVAILABLE_MODELS[0])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_interest_domain", None)
    # nonce to reset "Add interest" widgets
    st.session_state.setdefault("add_interest_nonce", 0)
    # follow-up input + clear flag
    st.session_state.setdefault("followup_question", "")
    st.session_state.setdefault("clear_followup", False)


def add_to_history(result):
    st.session_state.history.insert(0, result)
    st.session_state.history = st.session_state.history[:10]

# SIDEBAR
def render_sidebar():
    st.sidebar.title("⚙️ Settings")

    st.sidebar.subheader("🧠 Choose Model")
    st.session_state.chosen_model = st.sidebar.selectbox(
        "Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(st.session_state.chosen_model)
    )

    setup_apis()

    st.sidebar.subheader("🎯 Your Interests (Weighted)")
    st.sidebar.caption("Add interests that analogies will be drawn from.")

    # Existing interests
    for idx, interest in enumerate(st.session_state.user_interests):
        col1, col2, col3 = st.sidebar.columns([3, 2, 1])
        col1.write(f"• {interest}")
        current_w = int(st.session_state.interest_weights.get(interest, 1))
        new_w = col2.number_input(
            "Weight",
            min_value=0,
            step=1,
            value=current_w,
            key=f"weight_{idx}",
        )
        st.session_state.interest_weights[interest] = int(new_w)

        if col3.button("✕", key=f"remove_{idx}"):
            st.session_state.user_interests.pop(idx)
            st.session_state.interest_weights.pop(interest, None)
            st.rerun()

    # Add new interest (integer weight, text clears after Add)
    with st.sidebar.form("add_interest_form"):
        nonce = st.session_state.add_interest_nonce

        new_interest = st.text_input(
            "Add new interest:",
            key=f"new_interest_{nonce}",
        )
        w = st.number_input(
            "Initial weight",
            min_value=0,
            step=1,
            value=1,
            key=f"new_interest_weight_{nonce}",
        )

        submitted = st.form_submit_button("➕ Add")
        if submitted:
            if new_interest:
                if new_interest not in st.session_state.user_interests:
                    st.session_state.user_interests.append(new_interest)
                    st.session_state.interest_weights[new_interest] = int(w)
            # bump nonce so next run uses fresh widget keys (clears inputs)
            st.session_state.add_interest_nonce += 1
            st.rerun()

    # Reset
    if st.sidebar.button("🔄 Reset Interests"):
        st.session_state.user_interests = []
        st.session_state.interest_weights = {}
        st.session_state.last_interest_domain = None
        st.rerun()

# RESULTS UI
def render_history():
    if not st.session_state.history:
        return

    with st.expander("📖 Learning History"):
        for i, item in enumerate(st.session_state.history):
            st.markdown(
                f"**{i+1}. Topic:** {item['topic']}  \n"
                f"**Interest:** {item['interest_domain']}  \n"
                f"**Model:** {item['source']}"
            )
            if i < len(st.session_state.history) - 1:
                st.divider()


def render_analogy_result(result):
    st.success("✅ Analogy generated!")
    st.info(f"Using interest domain: **{result['interest_domain']}**")
    st.subheader(f"📚 Topic: {result['topic']}")
    st.markdown(result["analogy_text"])
    st.caption(f"Model: {result['source']}")

# MAIN APP
def main():
    st.set_page_config(page_title="Analogy Tutor", page_icon="🎓", layout="wide")

    # Tighten sidebar space
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.3rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    render_sidebar()

    st.title("🎓 Analogy Tutor")
    st.write("Learn any topic through an interest of your own! 👩🏻‍💻")

    st.subheader("🤔 What topic would you like explained?")
    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(
            "Enter topic",
            placeholder="e.g., gradient descent, backpropagation...",
            label_visibility="collapsed",
        )

    with col2:
        teach = st.button("🎯 Teach me", type="primary", use_container_width=True)

    if teach:
        if not topic.strip():
            st.warning("Please enter a topic.")
        elif not st.session_state.user_interests:
            st.error("Please add at least ONE interest in the sidebar first.")
        else:
            chosen, _ = pick_interest(topic, st.session_state.user_interests)
            with st.spinner(f"Generating analogy using: {chosen}"):
                analogy = generate_analogy(topic, chosen, st.session_state.chosen_model)

            if analogy:
                result = {
                    "topic": topic,
                    "interest_domain": chosen,
                    "analogy_text": analogy,
                    "source": st.session_state.chosen_model,
                }
                st.session_state.last_result = result
                add_to_history(result)
                # clear follow-up box on new analogy
                st.session_state.clear_followup = True
            else:
                st.error("Failed to generate analogy.")

    if st.session_state.last_result:
        result = st.session_state.last_result
        render_analogy_result(result)

        # Regenerate with different interest
        if st.button("🔁 Regenerate with another interest"):
            if not st.session_state.user_interests:
                st.error("Add interests first.")
            else:
                st.session_state.last_interest_domain = result["interest_domain"]
                chosen, _ = pick_interest(result["topic"], st.session_state.user_interests)

                with st.spinner(f"Regenerating using: {chosen}"):
                    new_text = generate_analogy(
                        result["topic"],
                        chosen,
                        st.session_state.chosen_model,
                    )

                if new_text:
                    new_result = {
                        "topic": result["topic"],
                        "interest_domain": chosen,
                        "analogy_text": new_text,
                        "source": st.session_state.chosen_model,
                    }
                    st.session_state.last_result = new_result
                    add_to_history(new_result)
                    # clear follow-up when we regenerate
                    st.session_state.clear_followup = True
                    st.rerun()

        
        # FOLLOW-UP QUESTION SECTION (simple + reliable)        
        st.subheader("💬 Ask a follow-up question")

        # Clear the input *before* rendering widget if flag is set
        if st.session_state.get("clear_followup", False):
            st.session_state.followup_question = ""
            st.session_state.clear_followup = False

        follow_q = st.text_input(
            "Your question:",
            key="followup_question",
        )

        if st.button("💡 Explain"):
            if not follow_q.strip():
                st.warning("Enter a follow-up question.")
            else:
                with st.spinner("Answering using same analogy…"):
                    out = generate_followup(
                        result["topic"],
                        result["interest_domain"],
                        result["analogy_text"],
                        follow_q,
                        st.session_state.chosen_model,
                    )
                if out:
                    st.markdown("### ✅ Follow-up Answer")
                    st.markdown(out)
                    # clear box after answering (no rerun — keep answer visible)
                    st.session_state.clear_followup = True
                else:
                    st.error("Failed to generate follow-up.")

    st.divider()
    render_history()


if __name__ == "__main__":
    main()
