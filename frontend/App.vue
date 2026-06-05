<template>
  <div class="app">
    <header class="header">
      <div class="header-inner">
      <div class="logo">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
            <path d="M8 12h8M12 8v8"/>
        </svg>
        인감 도장 대조 시스템
      </div>
      </div>
    </header>

    <main class="main">
      <section class="card upload-card">
        <h2 class="section-title">파일 업로드</h2>
        <p class="section-sub">인감 증명서와 계약서를 업로드하여 도장을 자동으로 대조합니다.</p>

        <div class="upload-grid">
          <FileUpload
            label="인감 증명서"
            v-model="certFile"
          />
          <FileUpload
            label="계약서"
            v-model="contractFile"
          />
        </div>

        <button
          class="compare-btn"
          :class="{ loading }"
          :disabled="!canCompare"
          @click="compare"
        >
          <span v-if="!loading" class="btn-content">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"/>
              <line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            대조 시작
          </span>
          <span v-else class="btn-content">
            <span class="spinner" />
            분석 중...
          </span>
        </button>
      </section>

        <!-- Error -->
      <div v-if="error" class="error-banner">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10"/>
          <line x1="12" y1="8" x2="12" y2="12"/>
          <line x1="12" y1="16" x2="12.01" y2="16"/>
          </svg>
          {{ error }}
        </div>

        <!-- Result -->
      <ResultDisplay v-if="result && !loading" :result="result" />
      </main>

    <footer class="footer">인감 도장 대조 시스템 &mdash; 결과는 참고용이며 법적 효력이 없습니다.</footer>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import FileUpload from './components/FileUpload.vue'
import ResultDisplay from './components/ResultDisplay.vue'

const certFile     = ref(null)
const contractFile = ref(null)
const loading      = ref(false)
const error        = ref(null)
const result       = ref(null)

const canCompare = computed(() => certFile.value && contractFile.value && !loading.value)

async function compare() {
  error.value  = null
  result.value = null
  loading.value = true

  try {
    const form = new FormData()
    form.append('certificate', certFile.value)
    form.append('contract',    contractFile.value)

    const res = await fetch('/api/compare', { method: 'POST', body: form })
    const data = await res.json()

    if (!res.ok) {
      error.value = data.detail ?? '알 수 없는 오류가 발생했습니다.'
      return
    }

    result.value = data
  } catch (e) {
    error.value = '서버에 연결할 수 없습니다. 백엔드 서버가 실행 중인지 확인해 주세요.'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* Header */
.header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 10;
}

.header-inner {
  max-width: 900px;
  margin: 0 auto;
  padding: 1rem 1.5rem;
}

.logo {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  display: flex;
  align-items: center;
  gap: .5rem;
}

.logo svg { color: var(--primary); }

/* Main */
.main {
  flex: 1;
  max-width: 900px;
  width: 100%;
  margin: 2rem auto;
  padding: 0 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* Card */
.card {
  background: var(--surface);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 2rem;
}

.section-title {
  font-size: 1.2rem;
  font-weight: 700;
  margin-bottom: .3rem;
}

.section-sub {
  font-size: .88rem;
  color: var(--muted);
  margin-bottom: 1.5rem;
}

.upload-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

@media (max-width: 560px) {
  .upload-grid { grid-template-columns: 1fr; }
}

/* Compare button */
.compare-btn {
  width: 100%;
  padding: .9rem;
  background: var(--primary);
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  transition: background .15s, opacity .15s;
}

.compare-btn:hover:not(:disabled) { background: var(--primary-dk); }
.compare-btn:disabled { opacity: .5; cursor: not-allowed; }

.btn-content {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: .5rem;
}

.spinner {
  width: 18px;
  height: 18px;
  border: 2.5px solid rgba(255,255,255,.4);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin .7s linear infinite;
  flex-shrink: 0;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* Error */
.error-banner {
  background: #fef2f2;
  color: #b91c1c;
  border: 1px solid #fecaca;
  border-radius: 8px;
  padding: .65rem .75rem;
  font-size: .78rem;
  display: flex;
  align-items: flex-start;
  gap: .4rem;
  flex-shrink: 0;
}
.error-box svg { flex-shrink: 0; margin-top: 1px; }

/* Legend */
.legend {
  margin-top: auto;
  padding-top: .75rem;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.legend-title {
  font-size: .72rem;
  font-weight: 700;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .05em;
  margin-bottom: .5rem;
}
.legend-row {
  display: flex;
  align-items: center;
  gap: .45rem;
  padding: .25rem 0;
}
.legend-dot {
  width: 9px; height: 9px;
  border-radius: 50%;
  flex-shrink: 0;
}
.legend-label { font-size: .82rem; font-weight: 600; min-width: 40px; }
.legend-range { font-size: .75rem; color: var(--muted); }

/* ── Right panel ───────────────────────────────────────── */
.right-panel {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  background: var(--bg);
}

/* Empty state */
.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  color: var(--muted);
  font-size: .9rem;
  text-align: center;
  line-height: 1.7;
}
</style>