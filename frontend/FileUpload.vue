<template>
  <div
    class="upload-zone"
    :class="{ dragover, 'has-file': modelValue }"
    @dragover.prevent="dragover = true"
    @dragleave.prevent="dragover = false"
    @drop.prevent="onDrop"
    @click="$refs.input.click()"
  >
    <input
      ref="input"
      type="file"
      :accept="accept"
      hidden
      @change="onFileChange"
    />

    <template v-if="!modelValue">
      <div class="icon">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="12" y1="18" x2="12" y2="12"/>
          <line x1="9" y1="15" x2="15" y2="15"/>
        </svg>
      </div>
      <p class="label">{{ label }}</p>
      <p class="hint">PDF, JPG, PNG 파일을 드래그하거나 클릭하세요</p>
    </template>

    <template v-else>
      <div class="file-info">
        <div class="file-icon">
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div>
          <p class="filename">{{ modelValue.name }}</p>
          <p class="filesize">{{ formatSize(modelValue.size) }}</p>
        </div>
        <button class="remove-btn" @click.stop="$emit('update:modelValue', null)">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18"/>
            <line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
      </div>
      <p class="label small">{{ label }}</p>
    </template>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  label:       { type: String,  default: '파일 업로드' },
  accept:      { type: String,  default: '.pdf,.jpg,.jpeg,.png' },
  modelValue:  { default: null },
})
const emit = defineEmits(['update:modelValue'])

const dragover = ref(false)

function onFileChange(e) {
  const file = e.target.files[0]
  if (file) emit('update:modelValue', file)
  e.target.value = ''
}

function onDrop(e) {
  dragover.value = false
  const file = e.dataTransfer.files[0]
  if (file) emit('update:modelValue', file)
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}
</script>

<style scoped>
.upload-zone {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1rem;
  cursor: pointer;
  transition: border-color .2s, background .2s;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: .45rem;
  min-height: 100px;
  justify-content: center;
  background: var(--surface);
  user-select: none;
}

.upload-zone:hover,
.upload-zone.dragover {
  border-color: var(--primary);
  background: #eff6ff;
}

.upload-zone.has-file {
  border-style: solid;
  border-color: var(--primary);
  background: #f0f7ff;
}

.icon { color: #94a3b8; }

.label {
  font-weight: 600;
  font-size: 1rem;
  color: var(--text);
}

.label.small {
  font-size: .8rem;
  color: var(--muted);
  margin-top: .25rem;
}

.hint {
  font-size: .8rem;
  color: var(--muted);
}

.file-info {
  display: flex;
  align-items: center;
  gap: .75rem;
  width: 100%;
  max-width: 320px;
}

.file-icon { color: var(--primary); flex-shrink: 0; }

.filename {
  font-size: .9rem;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 180px;
}

.filesize {
  font-size: .78rem;
  color: var(--muted);
}

.remove-btn {
  margin-left: auto;
  background: none;
  border: none;
  padding: .3rem;
  border-radius: 6px;
  color: var(--muted);
  flex-shrink: 0;
  display: flex;
  align-items: center;
  transition: background .15s, color .15s;
}

.remove-btn:hover {
  background: #fee2e2;
  color: var(--fail);
}
</style>
