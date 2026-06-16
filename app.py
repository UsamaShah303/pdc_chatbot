"""
FinAI Nexus RAG Chatbot - Premium UI
Parallel & Distributed Computing CCP
Student: Syed Usama Ali Shah | ID: 61585
"""

import gradio as gr
from rag import FinAIRagChatbot

print("Loading FinAI Nexus RAG Chatbot...")
chatbot = FinAIRagChatbot()
print("✅ Chatbot ready!")


def respond(message, history):
    if not message.strip():
        return "Please ask a question about Pakistani finance or investments."
    result = chatbot.answer(message)
    answer = result['answer']
    sources = result['sources']
    if sources:
        answer += "\n\n---\n📚 **Sources:** " + " · ".join(sources[:3])
    return answer


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

body, .gradio-container {
    background: #0A0E1A !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 1000px !important;
    margin: 0 auto !important;
    padding: 24px !important;
}

/* ── HERO ── */
.hero {
    position: relative;
    background: linear-gradient(135deg, #0D1B3E 0%, #1a2a6c 40%, #0D9488 100%);
    border-radius: 24px;
    padding: 40px 36px 36px;
    margin-bottom: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(13,148,136,0.3) 0%, transparent 70%);
    border-radius: 50%;
}

.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(46,117,182,0.25) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-content { position: relative; z-index: 1; }

.hero-top {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
}

.hero-icon {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #0D9488, #2E75B6);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 28px;
    box-shadow: 0 8px 24px rgba(13,148,136,0.4);
    flex-shrink: 0;
}

.hero-title {
    font-size: 2em !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    margin: 0 !important;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 0.95em !important;
    color: rgba(202,220,252,0.8) !important;
    margin: 4px 0 0 !important;
    font-weight: 400;
}

.badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 100px;
    padding: 5px 14px;
    font-size: 0.78em;
    font-weight: 500;
    color: rgba(255,255,255,0.85) !important;
    backdrop-filter: blur(8px);
    transition: all 0.2s;
}

.badge:hover {
    background: rgba(255,255,255,0.15);
    border-color: rgba(255,255,255,0.3);
}

/* ── STATS ── */
.stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
}

.stat {
    background: linear-gradient(135deg, #0D1B3E, #111827);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 18px 16px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    position: relative;
    overflow: hidden;
}

.stat::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}

.stat.s1::before { background: linear-gradient(90deg, #2E75B6, #60A5FA); }
.stat.s2::before { background: linear-gradient(90deg, #0D9488, #34D399); }
.stat.s3::before { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
.stat.s4::before { background: linear-gradient(90deg, #F59E0B, #FCD34D); }

.stat:hover { transform: translateY(-2px); border-color: rgba(255,255,255,0.15); }

.stat-num {
    font-size: 1.7em;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1;
    margin-bottom: 6px;
}

.stat-lbl {
    font-size: 0.74em;
    color: rgba(255,255,255,0.45);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── CHAT AREA ── */
.chat-wrap {
    background: #0D1B3E;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    overflow: hidden;
    margin-bottom: 16px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}

.chat-header {
    background: linear-gradient(90deg, #0D1B3E, #1a2a6c);
    padding: 16px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    display: flex;
    align-items: center;
    gap: 10px;
}

.chat-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    display: inline-block;
}

.d1 { background: #FF5F57; }
.d2 { background: #FFBD2E; }
.d3 { background: #28C840; }

.chat-title {
    font-size: 0.88em;
    font-weight: 600;
    color: rgba(255,255,255,0.7) !important;
    margin-left: 6px;
}

/* ── GRADIO OVERRIDES ── */
.message-wrap, #chatbot {
    background: transparent !important;
}

.message.user {
    background: linear-gradient(135deg, #2E75B6, #1a5a9e) !important;
    color: white !important;
    border-radius: 18px 18px 4px 18px !important;
    border: none !important;
    box-shadow: 0 4px 16px rgba(46,117,182,0.3) !important;
}

.message.bot {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(255,255,255,0.9) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px 18px 18px 4px !important;
}

.input-wrap, .input-row {
    background: rgba(255,255,255,0.04) !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    padding: 12px !important;
}

textarea, input[type="text"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: white !important;
    padding: 12px 16px !important;
    font-size: 0.95em !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #0D9488 !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.2) !important;
    outline: none !important;
}

textarea::placeholder { color: rgba(255,255,255,0.3) !important; }

button.primary {
    background: linear-gradient(135deg, #0D9488, #2E75B6) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 16px rgba(13,148,136,0.3) !important;
}

button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(13,148,136,0.4) !important;
}

/* ── EXAMPLES ── */
.examples-section {
    margin-bottom: 20px;
}

.examples-label {
    font-size: 0.82em;
    font-weight: 600;
    color: rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.example-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.example-pill {
    background: rgba(46,117,182,0.12);
    border: 1px solid rgba(46,117,182,0.25);
    border-radius: 100px;
    padding: 7px 16px;
    font-size: 0.83em;
    color: rgba(202,220,252,0.85) !important;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}

.example-pill:hover {
    background: rgba(46,117,182,0.25);
    border-color: rgba(46,117,182,0.5);
    color: white !important;
    transform: translateY(-1px);
}

/* ── FOOTER ── */
.footer {
    background: linear-gradient(135deg, #0D1B3E, #111827);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 16px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.footer-left { font-size: 0.82em; color: rgba(255,255,255,0.45); line-height: 1.5; }
.footer-right { font-size: 0.82em; color: rgba(255,255,255,0.35); text-align: right; }
.footer-right span { color: #F59E0B; font-weight: 600; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

footer { display: none !important; }
"""

with gr.Blocks(
    title="FinAI Nexus — AI Financial Advisor",
    theme=gr.themes.Base(),
    css=CSS,
) as demo:

    # ── Hero ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="hero">
      <div class="hero-content">
        <div class="hero-top">
          <div class="hero-icon">🤖</div>
          <div>
            <div class="hero-title">FinAI Nexus</div>
            <div class="hero-subtitle">AI-Powered Financial Advisor for Pakistani Investors</div>
          </div>
        </div>
        <div class="badges">
          <span class="badge">🦙 LLaMA 3.1 8B</span>
          <span class="badge">📄 RAG Pipeline</span>
          <span class="badge">🔍 FAISS Vector Search</span>
          <span class="badge">📚 1,426 Knowledge Chunks</span>
          <span class="badge">⚡ Groq Inference</span>
          <span class="badge">🇵🇰 Pakistan Market</span>
        </div>
      </div>
    </div>
    """)

    # ── Stats ─────────────────────────────────────────────────────
    gr.HTML("""
    <div class="stats">
      <div class="stat s1">
        <div class="stat-num">1,426</div>
        <div class="stat-lbl">Knowledge Chunks</div>
      </div>
      <div class="stat s2">
        <div class="stat-num">7</div>
        <div class="stat-lbl">PDF Documents</div>
      </div>
      <div class="stat s3">
        <div class="stat-num">384</div>
        <div class="stat-lbl">Vector Dimensions</div>
      </div>
      <div class="stat s4">
        <div class="stat-num">~1s</div>
        <div class="stat-lbl">Response Time</div>
      </div>
    </div>
    """)

    # ── Chat ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="chat-header" style="background:linear-gradient(90deg,#0D1B3E,#1a2a6c);
         padding:14px 20px; border-radius:20px 20px 0 0;
         border:1px solid rgba(255,255,255,0.07); border-bottom:none;
         display:flex; align-items:center; gap:8px;">
      <span class="chat-dot d1"></span>
      <span class="chat-dot d2"></span>
      <span class="chat-dot d3"></span>
      <span class="chat-title" style="margin-left:8px;">FinAI Nexus Chat — Pakistani Financial Assistant</span>
    </div>
    """)

    gr.ChatInterface(
        fn=respond,
        type="messages",
        chatbot=gr.Chatbot(
            height=440,
            placeholder="👋 Assalamu Alaikum! I am FinAI Nexus — your AI financial advisor for Pakistani investments. Ask me about PSX, mutual funds, Islamic finance and more!",
            type="messages",
            show_label=False,
            bubble_full_width=False,
            avatar_images=(
                "https://api.dicebear.com/7.x/initials/svg?seed=U&backgroundColor=2E75B6",
                "https://api.dicebear.com/7.x/bottts/svg?seed=finai&backgroundColor=0D9488"
            ),
        ),
        textbox=gr.Textbox(
            placeholder="💬  Ask about PSX, mutual funds, Islamic finance, portfolio tips...",
            container=False,
            scale=7,
            lines=1,
        ),
        examples=[
            "What is PSX and how can I invest in it?",
            "How do I read a financial statement?",
            "What are my rights as a shareholder?",
            "What is NAV in mutual funds?",
            "What is a Sukuk bond in Islamic finance?",
            "How does CDC account work in Pakistan?",
            "What is an IPO and how to apply?",
            "What is capital gains tax in Pakistan?",
            "How to start investing with PKR 1000?",
            "What is the difference between equity and income fund?",
        ],
    )

    # ── Footer ────────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer">
      <div class="footer-left">
        🎓 <strong style="color:rgba(255,255,255,0.7)">Iqra University Karachi</strong><br>
        Syed Usama Ali Shah (61585) · Parallel & Distributed Computing CCP
      </div>
      <div class="footer-right">
        Powered by <span>LLaMA 3.1 8B</span> + <span>RAG</span> + <span>FAISS</span><br>
        <span style="color:rgba(255,255,255,0.25)">Fine-tuned on Pakistani Financial Data</span>
      </div>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()