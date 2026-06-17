import { toast } from 'vue-sonner'
import { api } from './useApi'

export function useAgent() {
  const running = ref(false)
  const runningType = ref<string | null>(null)

  async function run(type: string, msg: string, dramaId: number, episodeId: number, onDone?: () => void | Promise<void>) {
    if (running.value) { toast.warning('操作执行中'); return }
    running.value = true
    runningType.value = type
    try {
      const result = await api.post<any>(`/agent/${type}/chat`, {
        message: msg,
        drama_id: dramaId,
        episode_id: episodeId,
      })
      await onDone?.()
      toast.success(result?.text || '完成')
    } catch (err: any) {
      toast.error(err.message)
    } finally {
      running.value = false
      runningType.value = null
    }
  }

  return { running, runningType, run }
}
