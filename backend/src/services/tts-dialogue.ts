export type TTSDialogueTurn = {
  speaker: string
  text: string
}

export type ParsedDialogueForTTS = {
  speaker: string
  pureText: string
  ignorable: boolean
  turns: TTSDialogueTurn[]
}

const SPEAKER_LABEL_RE = /(^|[\s\r\n\u3002\uFF01\uFF1F!?\uFF1B;])([^:\uFF1A\r\n\u3002\uFF01\uFF1F!?\uFF1B;]{1,32})[:\uFF1A]/g
const STAGE_DIRECTION_RE = /[\(\uFF08][^\)\uFF09]*[\)\uFF09]/g
const IGNORE_TTS_SPEAKERS = /^(?:\u73af\u5883\u97f3|\u73af\u5883\u58f0|\u97f3\u6548|\u6548\u679c\u97f3|sfx|sound ?effect|bgm|\u80cc\u666f\u97f3|\u80cc\u666f\u97f3\u4e50|ambient)$/i
const IGNORE_TTS_TEXT = /^(?:\u65e0|\u65e0\u5bf9\u767d|\u65e0\u53f0\u8bcd|\u65e0\u65c1\u767d|\u65e0\u9700\u914d\u97f3|\u65e0\u9700\u5bf9\u767d|none|null|n\/a|na|\u73af\u5883\u97f3|\u73af\u5883\u58f0|\u97f3\u6548|\u6548\u679c\u97f3|\u7eaf\u97f3\u6548|\u7eaf\u73af\u5883\u97f3|\u53ea\u6709\u73af\u5883\u97f3|\u4ec5\u73af\u5883\u97f3|\u80cc\u666f\u97f3|\u80cc\u666f\u97f3\u4e50|bgm|sfx|ambient)$/i
const NARRATOR_SPEAKERS = /^(?:\u65c1\u767d|\u753b\u5916\u97f3|narrator)$/i

export function parseDialogueForTTS(dialogue?: string | null): ParsedDialogueForTTS {
  const raw = dialogue?.trim() || ''
  if (!raw) return { speaker: '', pureText: '', ignorable: true, turns: [] }

  const turns = parseDialogueTurns(raw)
  const pureText = turns.map(turn => turn.text).filter(Boolean).join(' ')
  const speaker = turns[0]?.speaker || ''
  const ignorable = !pureText || turns.every(isIgnorableTurn)

  return { speaker, pureText, ignorable, turns }
}

export function isNarratorSpeaker(speaker?: string | null): boolean {
  return NARRATOR_SPEAKERS.test(String(speaker || '').trim())
}

function parseDialogueTurns(raw: string): TTSDialogueTurn[] {
  const matches = [...raw.matchAll(SPEAKER_LABEL_RE)].map(match => {
    const prefix = match[1] || ''
    const matchStart = match.index || 0
    return {
      speaker: cleanSpeakerName(match[2] || ''),
      labelStart: matchStart + prefix.length,
      contentStart: matchStart + match[0].length,
    }
  })
  if (!matches.length) {
    const text = cleanDialogueText(raw)
    return text ? [{ speaker: '', text }] : []
  }

  const turns: TTSDialogueTurn[] = []
  const leadingText = cleanDialogueText(raw.slice(0, matches[0].labelStart))
  if (leadingText) turns.push({ speaker: '', text: leadingText })

  for (let index = 0; index < matches.length; index += 1) {
    const match = matches[index]
    const contentEnd = matches[index + 1]?.labelStart ?? raw.length
    const text = cleanDialogueText(raw.slice(match.contentStart, contentEnd))

    if (match.speaker || text) {
      turns.push({ speaker: match.speaker, text })
    }
  }

  return turns.filter(turn => turn.text)
}

function isIgnorableTurn(turn: TTSDialogueTurn): boolean {
  return (!!turn.speaker && IGNORE_TTS_SPEAKERS.test(turn.speaker)) || !turn.text || IGNORE_TTS_TEXT.test(turn.text)
}

function cleanSpeakerName(value: string): string {
  return value.replace(STAGE_DIRECTION_RE, '').trim()
}

function cleanDialogueText(value: string): string {
  return value.replace(STAGE_DIRECTION_RE, '').replace(/\s+/g, ' ').trim()
}
