# 🎓 AI Concept Tutor  
Learn any complex topic through personalized analogies based on your interests.

🌐 **Live App:** https://ai-concept-tutor.streamlit.app/

AI Concept Tutor is an interactive learning tool that transforms difficult technical concepts into easy-to-understand analogies based on what *you* love — whether it's pop culture, cooking, sports, or anything else you add as an interest.  
Powered by **GPT-4o**, **Gemini 2.0 Flash**, and **Claude 2.1**, this app generates intuitive explanations and supports follow-up questions using the *same analogy domain* for deeper understanding.

---

## ✨ Features

### 🔹 Personalized Analogy Generation  
Enter any topic (e.g., "backpropagation" or "eigenvalues") and the app explains it using a domain you understand.

### 🔹 Weighted Interest Selection  
Assign higher weight to topics you prefer, letting the system choose them more often.

### 🔹 No Repetition Logic  
Prevents the same interest domain from being selected twice in a row.

### 🔹 Multi-Model Support  
Choose between:
- **GPT-4o**
- **Gemini 2.0 Flash**
- **Claude 2.1**

### 🔹 Follow-Up Questions  
Ask deeper questions and get explanations using **the same analogy domain**, preserving continuity.

### 🔹 Interest Management UI  
Add, remove, or weight interests dynamically through the sidebar.

---

## 🚀 Deployment  
This app is deployed on **Streamlit Cloud**:  
👉 **https://ai-concept-tutor.streamlit.app/**

To run locally:

```bash
pip install -r requirements.txt
streamlit run analogy_tutor_app.py
