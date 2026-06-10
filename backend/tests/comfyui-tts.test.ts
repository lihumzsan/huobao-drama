import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildComfyUiAudioViewUrl,
  findFirstOutputAudio,
  parseComfyUiAudioHistoryResponse,
  resolveComfyUiAudioWorkflow,
  selectComfyUiAudioWorkflow,
} from '../src/services/adapters/comfyui-tts.js'

test('selects pure text workflow when voice is not a reference audio source', () => {
  assert.equal(
    selectComfyUiAudioWorkflow({ text: '你好', voice: 'alloy' }),
    'baseaudio/音色/s2-se',
  )
})

test('selects LongCat single speaker clone workflow for reference audio by default', () => {
  assert.equal(
    selectComfyUiAudioWorkflow({ text: '你好', voice: '周星驰.MP3' }),
    'baseaudio/单人/LongCat-one',
  )
})

test('allows model value to select FishS2 clone workflow', () => {
  assert.equal(
    selectComfyUiAudioWorkflow({ text: '你好', voice: '周星驰.MP3', model: 's2-pro-bnb-nf4' }),
    'baseaudio/单人/s2-one',
  )
})

test('resolves LongCat clone workflow with target text and reference audio injected', () => {
  const graph = resolveComfyUiAudioWorkflow({
    workflowKey: 'baseaudio/单人/LongCat-one',
    text: '这是项目注入的新台词。',
    referenceAudioFilename: 'huobao-ref.wav',
  })

  const tts = Object.values(graph).find(node => node.class_type === 'LongCatVoiceCloneTTS')
  assert.ok(tts, 'LongCatVoiceCloneTTS should exist')
  const textSourceId = (tts.inputs.text as [string, number])[0]
  assert.equal(graph[textSourceId].inputs.text, '这是项目注入的新台词。')

  const loadAudio = Object.values(graph).find(node => node.class_type === 'LoadAudio')
  assert.ok(loadAudio, 'LoadAudio should exist')
  assert.equal(loadAudio.inputs.audio, 'huobao-ref.wav')
  assert.equal(Object.prototype.hasOwnProperty.call(loadAudio.inputs, 'upload'), false)
})

test('resolves pure text FishS2 workflow without requiring reference audio', () => {
  const graph = resolveComfyUiAudioWorkflow({
    workflowKey: 'baseaudio/音色/s2-se',
    text: '纯文本语音。',
  })

  const tts = Object.values(graph).find(node => node.class_type === 'FishS2TTS')
  assert.ok(tts, 'FishS2TTS should exist')
  const textSourceId = (tts.inputs.text as [string, number])[0]
  assert.equal(graph[textSourceId].inputs.text, '纯文本语音。')
  assert.equal(Object.values(graph).some(node => node.class_type === 'LoadAudio'), false)
})

test('finds audio outputs from ComfyUI history payloads', () => {
  assert.deepEqual(
    findFirstOutputAudio({
      '7': {
        audio: [
          { filename: 'voice.mp3', subfolder: 'audio', type: 'output' },
        ],
      },
    }),
    { filename: 'voice.mp3', subfolder: 'audio', type: 'output' },
  )
})

test('builds ComfyUI audio view urls', () => {
  assert.equal(
    buildComfyUiAudioViewUrl('http://127.0.0.1:8188/', {
      filename: 'voice.mp3',
      subfolder: 'audio',
      type: 'output',
    }),
    'http://127.0.0.1:8188/view?filename=voice.mp3&subfolder=audio&type=output',
  )
})

test('parses ComfyUI audio history responses', () => {
  assert.deepEqual(parseComfyUiAudioHistoryResponse({}, 'http://127.0.0.1:8188'), {
    status: 'processing',
  })

  assert.deepEqual(
    parseComfyUiAudioHistoryResponse({
      promptA: {
        status: { status_str: 'success', completed: true },
        outputs: {
          '4': {
            audio: [
              { filename: 'voice.mp3', subfolder: 'audio', type: 'output' },
            ],
          },
        },
      },
    }, 'http://127.0.0.1:8188'),
    {
      status: 'completed',
      audioUrl: 'http://127.0.0.1:8188/view?filename=voice.mp3&subfolder=audio&type=output',
    },
  )
})
