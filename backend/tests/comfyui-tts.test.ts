import assert from 'node:assert/strict'
import test from 'node:test'
import {
  buildComfyUiAudioViewUrl,
  ComfyUiTTSAdapter,
  findFirstOutputAudio,
  parseComfyUiAudioHistoryResponse,
  resolveComfyUiAudioWorkflow,
  selectComfyUiAudioWorkflow,
} from '../src/services/adapters/comfyui-tts.js'

function containsValue(value: unknown, expected: string): boolean {
  if (value === expected) return true
  if (Array.isArray(value)) return value.some(item => containsValue(item, expected))
  if (value && typeof value === 'object') {
    return Object.values(value).some(item => containsValue(item, expected))
  }
  return false
}

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

test('selects multi speaker workflows when multiple reference voices are provided', () => {
  assert.equal(
    selectComfyUiAudioWorkflow({
      text: '[speaker_1]: hello\n[speaker_2]: ok',
      voices: ['liu.wav', 'chen.wav'],
    } as any),
    'baseaudio/\u591a\u4eba/LongCat-two',
  )

  assert.equal(
    selectComfyUiAudioWorkflow({
      text: '[speaker_1]: hello\n[speaker_2]: ok',
      voices: ['liu.wav', 'chen.wav'],
      model: 's2-pro-bnb-nf4',
    } as any),
    'baseaudio/\u591a\u4eba/s2-two',
  )

  assert.equal(
    selectComfyUiAudioWorkflow({
      text: '[speaker_1]: hello\n[speaker_2]: ok\n[speaker_3]: good',
      voices: ['liu.wav', 'chen.wav', 'wang.wav'],
    } as any),
    'baseaudio/\u4e09\u4eba/s2-three',
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

test('resolves multi speaker FishS2 workflow with per-speaker reference audio injected', () => {
  const graph = resolveComfyUiAudioWorkflow({
    workflowKey: 'baseaudio/\u4e09\u4eba/s2-three',
    text: '[speaker_1]: A\n[speaker_2]: B\n[speaker_3]: C',
    referenceAudioFilenames: ['liu.wav', 'chen.wav', 'wang.wav'],
    modelPath: 's2-pro-bnb-nf4',
  } as any)

  const tts = Object.values(graph).find(node => node.class_type === 'FishS2MultiSpeakerTTS')
  assert.ok(tts, 'FishS2MultiSpeakerTTS should exist')
  assert.equal(graph[(tts.inputs['num_speakers.speaker_1_audio'] as [string, number])[0]].inputs.audio, 'liu.wav')
  assert.equal(graph[(tts.inputs['num_speakers.speaker_2_audio'] as [string, number])[0]].inputs.audio, 'chen.wav')
  assert.equal(graph[(tts.inputs['num_speakers.speaker_3_audio'] as [string, number])[0]].inputs.audio, 'wang.wav')
})

test('generates multi speaker ComfyUI prompts with uploaded reference audio per speaker', async () => {
  const originalFetch = globalThis.fetch
  let uploadCount = 0
  let promptPosted = false

  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input)

    if (url === 'http://127.0.0.1:8188/upload/image') {
      uploadCount += 1
      return new Response(JSON.stringify({ name: uploadCount === 1 ? 'liu.wav' : 'chen.wav' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8188/prompt') {
      promptPosted = true
      const body = JSON.parse(String(init?.body || '{}'))
      assert.ok(containsValue(body.prompt, '[speaker_1]: first line\n[speaker_2]: second line'))
      assert.ok(containsValue(body.prompt, 'liu.wav'))
      assert.ok(containsValue(body.prompt, 'chen.wav'))
      return new Response(JSON.stringify({ prompt_id: 'prompt-1' }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8188/history/prompt-1') {
      return new Response(JSON.stringify({
        prompt1: {
          status: { status_str: 'success', completed: true },
          outputs: {
            '9': {
              audio: [
                { filename: 'voice.mp3', subfolder: 'audio', type: 'output' },
              ],
            },
          },
        },
      }), {
        status: 200,
        headers: { 'content-type': 'application/json' },
      })
    }

    if (url === 'http://127.0.0.1:8188/view?filename=voice.mp3&subfolder=audio&type=output') {
      return new Response(Buffer.from([1, 2, 3]), {
        status: 200,
        headers: { 'content-type': 'audio/mpeg' },
      })
    }

    throw new Error(`Unexpected fetch: ${url}`)
  }) as typeof fetch

  try {
    const adapter = new ComfyUiTTSAdapter()
    const result = await adapter.generateAudio(
      { provider: 'comfyui', baseUrl: 'http://127.0.0.1:8188', apiKey: '', model: 's2-pro-bnb-nf4' },
      {
        text: 'fallback text',
        voice: 'alloy',
        speakers: [
          { speaker: 'liu', text: 'first line', voice: 'data:audio/wav;base64,AQID' },
          { speaker: 'chen', text: 'second line', voice: 'data:audio/wav;base64,BAUG' },
        ],
      },
    )

    assert.equal(uploadCount, 2)
    assert.equal(promptPosted, true)
    assert.equal(result.audioBuffer?.length, 3)
  } finally {
    globalThis.fetch = originalFetch
  }
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
