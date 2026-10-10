"""Local, code-drawn visuals and concise copy for the presentation homepage."""

import gradio as gr

BRAND_HEADER = """
<header class="sg-brandbar">
  <div class="sg-brand"><span class="sg-brand-icon" aria-hidden="true">◉</span>
    <span>SightGuard<small>视界 · 视觉辅助感知</small></span></div>
  <div class="sg-brand-meta"><span class="sg-dot"></span> LOCAL EXPERIENCE
    <span class="sg-version">研究原型 / 0.4</span></div>
</header>
"""

HERO_COPY = """
<div class="sg-intro">
  <div class="sg-eyebrow"><span></span> VISION BEYOND THE VISIBLE</div>
  <h1>看见前方。<br><span>感知更进一步。</span></h1>
  <p>让连续的视觉线索，成为更有意义的提醒。<br>
  SightGuard 探索视觉感知与风险事件筛选，<br class="sg-desktop-break">
  为视障辅助研究提供一种新的观察方式。</p>
</div>
"""

HERO_VISUAL = """
<div class="sg-optic" role="img" aria-label="视觉感知概念图，非实时检测画面">
  <div class="sg-optic-top"><span class="sg-dot"></span> PERCEPTION FIELD
    <span>CONCEPT VISUAL</span></div>
  <svg viewBox="0 0 600 460" fill="none" aria-hidden="true">
    <defs>
      <radialGradient id="sg-aura"><stop stop-color="#68efc3" stop-opacity=".24"/>
        <stop offset="1" stop-color="#68efc3" stop-opacity="0"/></radialGradient>
      <linearGradient id="sg-lens" x1="180" y1="110" x2="430" y2="350"
        gradientUnits="userSpaceOnUse"><stop stop-color="#94ffdc"/>
        <stop offset=".5" stop-color="#4897aa"/>
        <stop offset="1" stop-color="#287566"/></linearGradient>
      <linearGradient id="sg-beam"><stop stop-color="#8cf6cf" stop-opacity="0"/>
        <stop offset=".5" stop-color="#8cf6cf" stop-opacity=".65"/>
        <stop offset="1" stop-color="#8cf6cf" stop-opacity="0"/></linearGradient>
    </defs>
    <circle cx="300" cy="223" r="213" fill="url(#sg-aura)"/>
    <g stroke="#2e514b" stroke-width=".6" opacity=".5">
      <path d="M0 378H600M0 408H600M0 448H600M240 330L30 460M270 330L160 460
        M300 330V460M330 330L440 460M360 330L570 460"/>
      <path d="M40 110H120M480 110H560M40 330H120M480 330H560"/>
    </g>
    <g class="sg-orbit" transform-origin="300 223">
      <circle cx="300" cy="223" r="178" stroke="#416861" stroke-dasharray="2 12"/>
      <circle cx="300" cy="223" r="155" stroke="#63caae" stroke-opacity=".4"
        stroke-dasharray="145 828" stroke-width="2"/>
    </g>
    <ellipse cx="300" cy="223" rx="217" ry="66" transform="rotate(-27 300 223)"
      stroke="#64a89e" stroke-opacity=".35"/>
    <circle cx="300" cy="223" r="130" fill="#0d211e" fill-opacity=".8"
      stroke="url(#sg-lens)" stroke-width="1.2"/>
    <g stroke="url(#sg-lens)" opacity=".55">
      <ellipse cx="300" cy="223" rx="101" ry="130"/>
      <ellipse cx="300" cy="223" rx="57" ry="130"/>
      <ellipse cx="300" cy="223" rx="130" ry="40"/>
      <ellipse cx="300" cy="223" rx="130" ry="83"/>
      <path d="M170 223H430M300 93V353"/>
    </g>
    <path d="M203 223Q300 123 397 223Q300 323 203 223Z" fill="#0a211d"
      fill-opacity=".65" stroke="#9fffe0" stroke-width="1.5"/>
    <circle cx="300" cy="223" r="39" fill="#101f1d" stroke="#81f5cb"/>
    <circle cx="300" cy="223" r="22" fill="#7fefc6" fill-opacity=".15" stroke="#b1ffe5"/>
    <circle cx="300" cy="223" r="8" fill="#c5ffec"/>
    <path d="M94 223H506" stroke="url(#sg-beam)" stroke-width="1.5"/>
    <g stroke="#8ae9c8" stroke-width="1.4">
      <path d="M156 139V119H176M424 119H444V139M156 307V327H176M424 327H444V307"/>
      <path d="M402 178H461L489 150M206 279H131L99 304" stroke-opacity=".5"/>
    </g>
    <circle cx="402" cy="178" r="4" fill="#a2ffdc"/>
    <circle cx="206" cy="279" r="4" fill="#a2ffdc"/>
    <g fill="#8db4aa" font-family="Consolas, monospace" font-size="9" letter-spacing="2">
      <text x="455" y="138">VISUAL INPUT</text><text x="48" y="324">EVENT SIGNAL</text>
      <text x="261" y="400">SIGHT / GUARD</text>
    </g>
  </svg>
  <div class="sg-optic-bottom"><span>视觉输入 → 连续跟踪 → 事件提醒</span><span>01 / 03</span></div>
</div>
"""

CAPABILITIES = """
<section class="sg-capabilities">
  <div class="sg-section-heading"><div><span class="sg-eyebrow">DESIGNED TO PERCEIVE</span>
    <h2>从画面，到值得关注的变化。</h2></div>
    <span class="sg-section-caption">感知 · 理解 · 提醒</span></div>
  <div class="sg-feature-grid">
    <article class="sg-feature"><span class="sg-feature-index">01 / PERCEPTION</span>
      <div class="sg-feature-icon" aria-hidden="true">⌖</div><h3>持续观察</h3>
      <p>检测画面中的目标，关联连续帧中的变化，让单次观察延伸为持续的视觉线索。</p>
      <span class="sg-feature-tag">目标检测与跟踪</span></article>
    <article class="sg-feature"><span class="sg-feature-index">02 / ASSESSMENT</span>
      <div class="sg-feature-icon" aria-hidden="true">◎</div><h3>关注关键变化</h3>
      <p>结合画面位置、尺度与运动线索，形成可解释的风险评分和事件状态。</p>
      <span class="sg-feature-tag">图像风险线索</span></article>
    <article class="sg-feature"><span class="sg-feature-index">03 / AWARENESS</span>
      <div class="sg-feature-icon" aria-hidden="true">≋</div><h3>让提醒更有意义</h3>
      <p>筛选持续事件、识别等级变化，抑制同等级重复提醒，让演示聚焦重要信息。</p>
      <span class="sg-feature-tag">事件管理与提醒抑制</span></article>
  </div>
</section>
<div class="sg-journey"><span class="sg-eyebrow">EXPLORE THE EXPERIENCE</span>
  <p><b>实时体验</b> 查看摄像头感知画面 <span>↗</span>
  <b>视频回放</b> 探索已保存的视觉结果 <span>↗</span>
  <b>策略对比</b> 观察不同提醒策略</p></div>
"""

FOOTER = """
<footer class="sg-footer"><div><strong>SightGuard / 视界</strong>
  <span>探索更有意义的视觉辅助感知。</span></div>
  <p>研究原型 · 补充盲杖与导盲犬。图像风险评分不代表碰撞概率或安全保证。
  合成示例仅用于流程验证，真实效果仍需独立验证。</p></footer>
"""


def add_overview() -> tuple[gr.Button, gr.Button]:
    """Build the presentation entry without loading a model or opening a camera."""
    with gr.Row(elem_classes="sg-landing-hero"):
        with gr.Column(scale=1, min_width=300, elem_classes="sg-hero-copy"):
            gr.HTML(HERO_COPY)
            with gr.Row(elem_classes="sg-hero-actions"):
                live_button = gr.Button("开启实时体验  ↗", variant="primary", min_width=170)
                replay_button = gr.Button("探索视频回放  →", min_width=170)
            gr.HTML(
                '<div class="sg-hero-footnote"><span>本地处理</span><i></i>'
                "<span>可解释事件</span><i></i><span>按需启动摄像头</span></div>"
            )
        with gr.Column(scale=1, min_width=300, elem_classes="sg-hero-art"):
            gr.HTML(HERO_VISUAL)
    gr.HTML(CAPABILITIES)
    return live_button, replay_button


def section_heading(kicker: str, title: str, description: str) -> str:
    """Render fixed, application-authored section introductions."""
    import html

    return (
        '<div class="sg-page-heading">'
        f'<span class="sg-eyebrow">{html.escape(kicker)}</span>'
        f"<h2>{html.escape(title)}</h2><p>{html.escape(description)}</p></div>"
    )
