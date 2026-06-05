<template>
  <!--
    Layout (fills the right panel, no scroll):
    ┌─────────────────────────────────────────────────────┐
    │  [Score gauge]  [Cert seal ── VS ── Cont seal]  [Metrics] │  ← top-row (fixed height)
    ├─────────────────────────────────────────────────────┤
    │  [Cert doc preview]          [Contract doc preview] │  ← bottom-row (fills rest)
    └─────────────────────────────────────────────────────┘
  -->
  <div class="result-root">

    <!-- TOP ROW -->
    <div class="top-row">

      <!-- Score block -->
      <div class="score-block card">
        <svg class="gauge-svg" viewBox="0 0 160 95">
          <path d="M 14 82 A 66 66 0 0 1 146 82" fill="none" stroke="#e2e8f0" stroke-width="13" stroke-linecap="round"/>
          <path d="M 14 82 A 66 66 0 0 1 146 82" fill="none" :stroke="result.verdict_color"
            stroke-width="13" stroke-linecap="round"
            :stroke-dasharray="`${arcLen} 999`" style="transition:stroke-dasharray 1s ease"/>
          <text x="80" y="73" text-anchor="middle" font-size="28" font-weight="700" :fill="result.verdict_color">{{ result.score }}</text>
          <text x="80" y="89" text-anchor="middle" font-size="11" fill="#94a3b8">점</text>
        </svg>
        <div class="verdict-badge" :style="badgeStyle">{{ result.verdict }}</div>
        <p class="verdict-desc">{{ verdictDesc }}</p>
      </div>

      <!-- Seals comparison -->
      <div class="seals-block card">
        <div class="seal-item">
          <p class="seal-label">인감 증명서</p>
          <div class="seal-img-wrap">
            <img :src="`data:image/png;base64,${result.certificate_seal}`" class="seal-img" alt="인감 증명서 도장" />
          </div>
        </div>
        <div class="vs-col">
          <span class="vs-text">VS</span>
          <div class="match-line" :style="{ background: result.verdict_color }"/>
        </div>
        <div class="seal-item">
          <p class="seal-label">계약서</p>
          <div class="seal-img-wrap">
            <img :src="`data:image/png;base64,${result.contract_seal}`" class="seal-img" alt="계약서 도장" />
          </div>
        </div>
      </div>

      <!-- Metrics block -->
      <div class="metrics-block card">
        <p class="block-title">세부 분석</p>
        <div class="metrics">
          <div v-for="(val, key) in result.details" :key="key" class="metric">
            <div class="metric-header">
              <span class="metric-name">{{ metricLabel(key) }}</span>
              <span class="metric-val" :style="{ color: barColor(val) }">{{ val }}<small>점</small></span>
            </div>
            <div class="bar-bg">
              <div class="bar-fill" :style="{ width: val + '%', background: barColor(val) }"/>
            </div>
          </div>
        </div>

      </div>
    </div>

    <!-- BOTTOM ROW: full-document previews -->
    <div class="bottom-row">
      <div class="preview-card card">
        <p class="block-title">인감 증명서 — 도장 위치</p>
        <div class="preview-img-wrap">
          <img :src="`data:image/png;base64,${result.certificate_preview}`" class="preview-img" alt="인감 증명서 미리보기"/>
        </div>
      </div>
      <div class="preview-card card">
        <p class="block-title">계약서 — 도장 위치</p>
        <div class="preview-img-wrap">
          <img :src="`data:image/png;base64,${result.contract_preview}`" class="preview-img" alt="계약서 미리보기"/>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({ result: { type: Object, required: true } })

// Arc for r=66 half-circle: π×66 ≈ 207.3
const ARC_TOTAL = 207.3
const arcLen = computed(() => (props.result.score / 100) * ARC_TOTAL)

const badgeStyle = computed(() => ({
  background:   props.result.verdict_color + '22',
  color:        props.result.verdict_color,
  borderColor:  props.result.verdict_color + '55',
}))

const verdictDesc = computed(() => ({
  '패스':   '동일한 인감으로 판단됩니다.',
  '양호':   '대체로 동일한 인감으로 볼 수 있습니다.',
  '주의':   '추가 육안 확인이 필요합니다.',
  '불일치': '다른 인감으로 판단됩니다.',
}[props.result.verdict] ?? ''))


const metricLabels = {
  red_pattern:      '잉크 패턴 일치',
  structural:       '구조적 유사도',
  color_histogram:  '색상 분포',
  feature_matching: '특징점 매칭',
}
const metricLabel = k => metricLabels[k] ?? k

function barColor(v) {
  if (v >= 85) return '#22c55e'
  if (v >= 70) return '#84cc16'
  if (v >= 55) return '#f59e0b'
  return '#ef4444'
}
</script>

<style scoped>
/* ── Root ─────────────────────────────────────────────── */
.result-root {
  display: grid;
  grid-template-rows: auto 1fr;
  gap: .75rem;
  padding: .75rem;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}

.card {
  background: var(--surface);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}

/* ── TOP ROW ──────────────────────────────────────────── */
.top-row {
  display: grid;
  grid-template-columns: 190px 1fr 210px;
  gap: .75rem;
  min-height: 0;
}

/* Score block */
.score-block {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: .85rem .75rem .75rem;
  gap: .4rem;
}
.gauge-svg { width: 100%; max-width: 152px; height: auto; }

.verdict-badge {
  padding: .3rem 1.1rem;
  border-radius: 999px;
  font-weight: 700;
  font-size: 1rem;
  border: 2px solid transparent;
}
.verdict-desc {
  font-size: .74rem;
  color: var(--muted);
  text-align: center;
  line-height: 1.5;
}

/* Seals block */
.seals-block {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: .75rem;
  padding: .85rem 1rem;
}
.seal-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: .4rem;
  flex: 1;
  min-width: 0;
}
.seal-label { font-size: .74rem; font-weight: 600; color: var(--muted); }
.seal-img-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 110px;
}
.seal-img {
  max-width: 100%;
  max-height: 110px;
  object-fit: contain;
  border-radius: 6px;
  border: 1px solid var(--border);
}
.vs-col {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: .4rem;
  flex-shrink: 0;
}
.vs-text { font-size: 1rem; font-weight: 800; color: #94a3b8; }
.match-line { width: 2px; height: 50px; border-radius: 1px; opacity: .5; }

/* Metrics block */
.metrics-block {
  display: flex;
  flex-direction: column;
  padding: .85rem .9rem .75rem;
  gap: .55rem;
}
.block-title {
  font-size: .75rem;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .04em;
  flex-shrink: 0;
}
.metrics { display: flex; flex-direction: column; gap: .6rem; justify-content: space-between; flex: 1; }

.metric-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: .2rem;
}
.metric-name { font-size: .76rem; color: var(--muted); }
.metric-val  { font-size: .82rem; font-weight: 700; }
.metric-val small { font-size: .65rem; font-weight: 400; margin-left: 1px; }

.bar-bg { height: 6px; background: var(--border); border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px; transition: width 1s ease; }


/* ── BOTTOM ROW ───────────────────────────────────────── */
.bottom-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: .75rem;
  min-height: 0;
}

.preview-card {
  display: flex;
  flex-direction: column;
  padding: .75rem;
  gap: .5rem;
  min-height: 0;
}

.preview-img-wrap {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

.preview-img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 6px;
  border: 1px solid var(--border);
}
</style>
