import assert from 'node:assert/strict'
import test from 'node:test'
import { parseDialogueForTTS } from '../src/services/tts-dialogue.js'

test('parses ordered speaker turns from Chinese multi-speaker dialogue', () => {
  const parsed = parseDialogueForTTS('\u8001\u5218\uff1a\u9648\u8ff9\u4f60\u597d\uff0c\u6211\u73b0\u5728\u9700\u8981\u95ee\u4f60\u4e00\u4e9b\u95ee\u9898\u3002 \u9648\u8ff9\uff1a\u53ef\u4ee5\u3002')

  assert.equal(parsed.ignorable, false)
  assert.deepEqual(parsed.turns, [
    { speaker: '\u8001\u5218', text: '\u9648\u8ff9\u4f60\u597d\uff0c\u6211\u73b0\u5728\u9700\u8981\u95ee\u4f60\u4e00\u4e9b\u95ee\u9898\u3002' },
    { speaker: '\u9648\u8ff9', text: '\u53ef\u4ee5\u3002' },
  ])
  assert.equal(parsed.speaker, '\u8001\u5218')
  assert.equal(parsed.pureText, '\u9648\u8ff9\u4f60\u597d\uff0c\u6211\u73b0\u5728\u9700\u8981\u95ee\u4f60\u4e00\u4e9b\u95ee\u9898\u3002 \u53ef\u4ee5\u3002')
})

test('keeps single-speaker dialogue compatible with existing TTS callers', () => {
  const parsed = parseDialogueForTTS('\u9648\u8ff9\uff1a\u6211\u6ca1\u6709\u8981\u7ed3\u675f\u751f\u547d\u3002')

  assert.equal(parsed.ignorable, false)
  assert.deepEqual(parsed.turns, [
    { speaker: '\u9648\u8ff9', text: '\u6211\u6ca1\u6709\u8981\u7ed3\u675f\u751f\u547d\u3002' },
  ])
  assert.equal(parsed.pureText, '\u6211\u6ca1\u6709\u8981\u7ed3\u675f\u751f\u547d\u3002')
})
