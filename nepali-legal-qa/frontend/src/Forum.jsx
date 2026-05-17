import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  ArrowUp,
  ArrowDown,
  MessageSquare,
  Search,
  Tag,
  Plus,
  RefreshCw,
  Scale,
  Sparkles,
  Clock,
} from 'lucide-react'
import { UserProfile } from './GoogleLogin'

const API_BASE = (import.meta.env.VITE_FORUM_API_BASE ?? '').replace(/\/$/, '')
const FORUM_BASE = API_BASE ? `${API_BASE}/api/forum` : '/api/forum'

async function forumRequest(path, options = {}) {
  const token = localStorage.getItem('auth_token')
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers ?? {}),
  }

  if (token) {
    headers.Authorization = `Bearer ${token}`
  }

  const res = await fetch(`${FORUM_BASE}${path}`, {
    ...options,
    headers,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `HTTP ${res.status}`)
  }

  return res.json()
}

function formatRelativeTime(value) {
  const now = Date.now()
  const then = new Date(value).getTime()
  if (Number.isNaN(then)) return 'just now'

  const diffSec = Math.round((now - then) / 1000)
  if (diffSec < 60) return 'just now'

  const diffMin = Math.round(diffSec / 60)
  if (diffMin < 60) return `${diffMin}m ago`

  const diffHours = Math.round(diffMin / 60)
  if (diffHours < 24) return `${diffHours}h ago`

  const diffDays = Math.round(diffHours / 24)
  return `${diffDays}d ago`
}

function scoreLabel(upvotes, downvotes) {
  const score = upvotes - downvotes
  if (score === 0) return '0'
  return score > 0 ? `+${score}` : `${score}`
}

function Skeleton({ className }) {
  return <div className={`animate-pulse rounded-md bg-purple-100/70 ${className}`} />
}

export function Forum({ user, onLogout }) {
  const [questions, setQuestions] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [selectedThread, setSelectedThread] = useState(null)
  const [listLoading, setListLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)
  const [listError, setListError] = useState(null)
  const [detailError, setDetailError] = useState(null)
  const [sort, setSort] = useState('hot')
  const [query, setQuery] = useState('')
  const [tagFilter, setTagFilter] = useState('')
  const [composerOpen, setComposerOpen] = useState(false)
  const [lastSync, setLastSync] = useState(null)
  const [creating, setCreating] = useState(false)
  const [answering, setAnswering] = useState(false)
  const [createError, setCreateError] = useState(null)
  const [answerError, setAnswerError] = useState(null)

  const listAbort = useRef(null)
  const detailAbort = useRef(null)

  const filteredTags = useMemo(() => {
    const set = new Set()
    questions.forEach((q) => q.tags.forEach((tag) => set.add(tag)))
    return Array.from(set).slice(0, 12)
  }, [questions])

  useEffect(() => {
    let active = true

    async function loadList(isSilent = false) {
      listAbort.current?.abort()
      const controller = new AbortController()
      listAbort.current = controller

      if (!isSilent) {
        setListLoading(true)
        setListError(null)
      }

      try {
        const params = new URLSearchParams({
          sort,
          limit: '40',
        })
        if (query.trim()) params.set('q', query.trim())
        if (tagFilter) params.set('tag', tagFilter)

        const data = await forumRequest(`/questions?${params.toString()}`, {
          signal: controller.signal,
        })

        if (!active) return
        setQuestions(data)
        setLastSync(new Date().toISOString())
        setListError(null)

        if (!selectedId && data.length > 0) {
          setSelectedId(data[0].id)
        } else if (selectedId && !data.find((q) => q.id === selectedId)) {
          setSelectedId(data[0]?.id ?? null)
        }
      } catch (err) {
        if (!active || err.name === 'AbortError') return
        setListError(err.message)
      } finally {
        if (!active) return
        setListLoading(false)
      }
    }

    loadList()
    const id = setInterval(() => loadList(true), 5000)

    return () => {
      active = false
      clearInterval(id)
      listAbort.current?.abort()
    }
  }, [sort, query, tagFilter, selectedId])

  useEffect(() => {
    if (!selectedId) {
      setSelectedThread(null)
      return
    }

    let active = true

    async function loadDetail(isSilent = false) {
      detailAbort.current?.abort()
      const controller = new AbortController()
      detailAbort.current = controller

      if (!isSilent) {
        setDetailLoading(true)
        setDetailError(null)
      }

      try {
        const data = await forumRequest(`/questions/${selectedId}`, {
          signal: controller.signal,
        })
        if (!active) return
        setSelectedThread(data)
        setDetailError(null)
      } catch (err) {
        if (!active || err.name === 'AbortError') return
        setDetailError(err.message)
      } finally {
        if (!active) return
        setDetailLoading(false)
      }
    }

    loadDetail()
    const id = setInterval(() => loadDetail(true), 4000)

    return () => {
      active = false
      clearInterval(id)
      detailAbort.current?.abort()
    }
  }, [selectedId])

  const handleCreateQuestion = async (event) => {
    event.preventDefault()
    const form = event.currentTarget
    const formData = new FormData(form)

    const payload = {
      title: formData.get('title')?.toString() ?? '',
      body: formData.get('body')?.toString() ?? '',
      tags: (formData.get('tags')?.toString() ?? '')
        .split(',')
        .map((tag) => tag.trim())
        .filter(Boolean),
    }

    setCreating(true)
    setCreateError(null)

    try {
      const created = await forumRequest('/questions', {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      form.reset()
      setComposerOpen(false)
      setSelectedId(created.id)
      setQuestions((prev) => [created, ...prev])
    } catch (err) {
      setCreateError(err.message)
    } finally {
      setCreating(false)
    }
  }

  const handleCreateAnswer = async (event) => {
    event.preventDefault()
    if (!selectedThread) return

    const form = event.currentTarget
    const formData = new FormData(form)
    const payload = {
      body: formData.get('answer')?.toString() ?? '',
    }

    setAnswering(true)
    setAnswerError(null)

    try {
      const created = await forumRequest(`/questions/${selectedThread.question.id}/answers`, {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      form.reset()
      setSelectedThread((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          question: {
            ...prev.question,
            answer_count: prev.question.answer_count + 1,
          },
          answers: [...prev.answers, created],
        }
      })
    } catch (err) {
      setAnswerError(err.message)
    } finally {
      setAnswering(false)
    }
  }

  const handleVote = async (target, id, value) => {
    try {
      await forumRequest(`/${target}/${id}/vote`, {
        method: 'POST',
        body: JSON.stringify({ value }),
      })
      if (target === 'questions') {
        setQuestions((prev) =>
          prev.map((q) =>
            q.id === id
              ? {
                  ...q,
                  upvotes: value === 1 ? q.upvotes + 1 : q.upvotes,
                  downvotes: value === -1 ? q.downvotes + 1 : q.downvotes,
                }
              : q
          )
        )
        setSelectedThread((prev) => {
          if (!prev || prev.question.id !== id) return prev
          return {
            ...prev,
            question: {
              ...prev.question,
              upvotes: value === 1 ? prev.question.upvotes + 1 : prev.question.upvotes,
              downvotes: value === -1 ? prev.question.downvotes + 1 : prev.question.downvotes,
            },
          }
        })
      } else {
        setSelectedThread((prev) => {
          if (!prev) return prev
          return {
            ...prev,
            answers: prev.answers.map((answer) =>
              answer.id === id
                ? {
                    ...answer,
                    upvotes: value === 1 ? answer.upvotes + 1 : answer.upvotes,
                    downvotes: value === -1 ? answer.downvotes + 1 : answer.downvotes,
                  }
                : answer
            ),
          }
        })
      }
    } catch (err) {
      setDetailError(err.message)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-white via-purple-50/30 to-white text-gray-900">
      <header className="border-b border-purple-100 bg-white/80 backdrop-blur-xl sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-5 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-600 to-indigo-600 flex items-center justify-center text-white shadow-md">
              <Scale className="w-5 h-5" strokeWidth={2.5} aria-hidden="true" />
            </div>
            <div>
              <p className="text-sm font-semibold text-gray-900 font-display">Nyaay Forum</p>
              <p className="text-[11px] text-gray-400">Community legal Q&A</p>
            </div>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm">
            <Link className="text-gray-600 hover:text-gray-900" to="/">Home</Link>
            <Link className="text-gray-600 hover:text-gray-900" to="/about">About</Link>
            <Link className="text-purple-600 font-semibold" to="/forum">Forum</Link>
            <Link className="text-gray-600 hover:text-gray-900" to="/chat">Ask AI</Link>
          </nav>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => setComposerOpen((open) => !open)}
              className="hidden sm:inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-full bg-purple-600 text-white hover:bg-purple-700 transition-colors focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
            >
              <Plus className="w-4 h-4" aria-hidden="true" /> Ask a question
            </button>
            {user && <UserProfile user={user} onLogout={onLogout} />}
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8 grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] gap-8">
        <section className="space-y-6">
          <div className="rounded-2xl border border-purple-100 bg-white shadow-sm p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <p className="text-xs font-semibold tracking-[0.2em] text-purple-500 uppercase">Live forum</p>
                <h1 className="text-3xl font-semibold text-gray-900 mt-2 font-display">Ask legal questions, share real answers.</h1>
                <p className="text-sm text-gray-500 mt-2 max-w-xl">Post anything related to Nepali law. Community members and AI-assisted responses help you move forward fast.</p>
              </div>
              <Link
                to="/chat"
                className="inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-full border border-purple-200 text-purple-700 hover:border-purple-300 hover:bg-purple-50 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
              >
                <Sparkles className="w-4 h-4" aria-hidden="true" /> Ask AI now
              </Link>
            </div>
            <div className="mt-5 grid gap-3 md:grid-cols-[1fr_auto]">
              <label className="relative">
                <span className="sr-only">Search questions</span>
                <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" aria-hidden="true" />
                <input
                  type="search"
                  name="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Search questions, keywords, or laws"
                  className="w-full h-11 rounded-xl border border-gray-200 pl-10 pr-4 text-sm text-gray-700 placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                  autoComplete="off"
                />
              </label>
              <div className="flex items-center gap-2">
                {['hot', 'new', 'top'].map((value) => (
                  <button
                    key={value}
                    type="button"
                    onClick={() => setSort(value)}
                    className={`h-11 px-4 rounded-xl text-xs font-semibold border transition-colors focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white ${
                      sort === value
                        ? 'bg-purple-600 text-white border-purple-600'
                        : 'border-gray-200 text-gray-600 hover:border-gray-300'
                    }`}
                  >
                    {value === 'hot' ? 'Hot' : value === 'new' ? 'New' : 'Top'}
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-3 text-xs text-gray-500">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-50 border border-purple-100">
                <Clock className="w-3 h-3" aria-hidden="true" />
                <span>Last sync {lastSync ? formatRelativeTime(lastSync) : 'just now'}</span>
              </div>
              {tagFilter && (
                <button
                  type="button"
                  onClick={() => setTagFilter('')}
                  className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-purple-200 text-purple-700 hover:bg-purple-50 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                >
                  <Tag className="w-3 h-3" aria-hidden="true" /> #{tagFilter} (clear)
                </button>
              )}
            </div>
          </div>

          {composerOpen && (
            <form onSubmit={handleCreateQuestion} className="rounded-2xl border border-purple-200 bg-white shadow-sm p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 font-display">Ask a new question</h2>
                <button
                  type="button"
                  onClick={() => setComposerOpen(false)}
                  className="text-xs text-gray-500 hover:text-gray-700 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                >
                  Close
                </button>
              </div>
              <label className="block">
                <span className="text-xs font-semibold text-gray-600">Title</span>
                <input
                  name="title"
                  type="text"
                  required
                  minLength={8}
                  maxLength={140}
                  placeholder="Example: How to file for divorce in Nepal?"
                  className="mt-2 w-full h-11 rounded-xl border border-gray-200 px-3 text-sm text-gray-700 placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                  autoComplete="off"
                />
              </label>
              <label className="block">
                <span className="text-xs font-semibold text-gray-600">Details</span>
                <textarea
                  name="body"
                  required
                  minLength={20}
                  maxLength={4000}
                  rows={5}
                  placeholder="Share context, location, and anything already tried."
                  className="mt-2 w-full rounded-xl border border-gray-200 px-3 py-2 text-sm text-gray-700 placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                />
              </label>
              <label className="block">
                <span className="text-xs font-semibold text-gray-600">Tags (comma separated)</span>
                <input
                  name="tags"
                  type="text"
                  placeholder="labor, divorce, property"
                  className="mt-2 w-full h-11 rounded-xl border border-gray-200 px-3 text-sm text-gray-700 placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                  autoComplete="off"
                />
              </label>
              {createError && (
                <div className="text-sm text-red-600">{createError}</div>
              )}
              <div className="flex flex-wrap items-center gap-3">
                <button
                  type="submit"
                  disabled={creating}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-purple-600 text-white text-xs font-semibold hover:bg-purple-700 disabled:opacity-60 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                >
                  {creating ? 'Posting...' : 'Post question'}
                </button>
                <p className="text-xs text-gray-400">Keep it factual. No personal identifiers.</p>
              </div>
            </form>
          )}

          <div className="rounded-2xl border border-purple-100 bg-white shadow-sm">
            <div className="flex items-center justify-between px-5 py-4 border-b border-purple-50">
              <div className="flex items-center gap-2 text-xs text-gray-500">
                <MessageSquare className="w-4 h-4" aria-hidden="true" />
                <span>{questions.length} questions</span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setSort('hot')
                  setQuery('')
                  setTagFilter('')
                }}
                className="inline-flex items-center gap-2 text-xs text-gray-500 hover:text-gray-700 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
              >
                <RefreshCw className="w-4 h-4" aria-hidden="true" /> Reset
              </button>
            </div>

            {listLoading ? (
              <div className="p-5 space-y-4">
                {[...Array(4)].map((_, idx) => (
                  <div key={idx} className="grid grid-cols-[auto_1fr] gap-4">
                    <Skeleton className="w-10 h-16" />
                    <div className="space-y-2">
                      <Skeleton className="h-4 w-3/4" />
                      <Skeleton className="h-3 w-1/2" />
                      <Skeleton className="h-3 w-1/3" />
                    </div>
                  </div>
                ))}
              </div>
            ) : listError ? (
              <div className="p-6 text-sm text-red-600">
                {listError}
                <button
                  type="button"
                  onClick={() => setSort('hot')}
                  className="ml-3 text-xs text-gray-500 underline focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                >
                  Retry
                </button>
              </div>
            ) : questions.length === 0 ? (
              <div className="p-10 text-center">
                <p className="text-sm text-gray-600">No questions yet.</p>
                <button
                  type="button"
                  onClick={() => setComposerOpen(true)}
                  className="mt-3 inline-flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-full bg-purple-600 text-white focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                >
                  <Plus className="w-4 h-4" aria-hidden="true" /> Start the first thread
                </button>
              </div>
            ) : (
              <div className="divide-y divide-purple-50">
                {questions.map((q) => (
                  <button
                    key={q.id}
                    type="button"
                    onClick={() => setSelectedId(q.id)}
                    className={`w-full text-left px-5 py-4 grid grid-cols-[auto_1fr] gap-4 transition-colors focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white ${
                      selectedId === q.id ? 'bg-purple-50' : 'hover:bg-purple-50/70'
                    }`}
                  >
                    <div className="flex flex-col items-center gap-1">
                      <span className="text-xs font-semibold text-gray-700">{scoreLabel(q.upvotes, q.downvotes)}</span>
                      <span className="text-[10px] text-gray-400">score</span>
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold text-gray-900 leading-snug">{q.title}</h3>
                        <span className="text-[11px] text-gray-400">{formatRelativeTime(q.created_at)}</span>
                      </div>
                      <p className="text-xs text-gray-500 mt-1 max-h-10 overflow-hidden">{q.body}</p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-gray-400">
                        <span>{q.answer_count} answers</span>
                        <span className="text-gray-300">•</span>
                        <span>{q.author?.name ?? 'Guest'}</span>
                        {q.tags.length > 0 && (
                          <span className="flex flex-wrap items-center gap-2">
                            {q.tags.map((tag) => (
                              <span
                                key={tag}
                                className="px-2 py-0.5 rounded-full bg-purple-100/70 text-purple-700"
                              >
                                #{tag}
                              </span>
                            ))}
                          </span>
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </section>

        <aside className="space-y-6">
          <div className="rounded-2xl border border-purple-100 bg-white shadow-sm p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 font-display">Thread</h2>
              <button
                type="button"
                onClick={() => setComposerOpen(true)}
                className="inline-flex sm:hidden items-center gap-2 px-3 py-2 text-xs font-semibold rounded-full border border-purple-200 text-purple-700 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
              >
                <Plus className="w-4 h-4" aria-hidden="true" /> Ask
              </button>
            </div>

            {detailLoading ? (
              <div className="mt-4 space-y-4">
                <Skeleton className="h-5 w-3/4" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-2/3" />
              </div>
            ) : detailError ? (
              <div className="mt-4 text-sm text-red-600">{detailError}</div>
            ) : !selectedThread ? (
              <p className="mt-4 text-sm text-gray-500">Select a question to see details.</p>
            ) : (
              <div className="mt-4 space-y-4">
                <div>
                  <p className="text-xs text-gray-400">Asked {formatRelativeTime(selectedThread.question.created_at)}</p>
                  <h3 className="text-xl font-semibold text-gray-900 mt-1 font-display">{selectedThread.question.title}</h3>
                  <p className="text-sm text-gray-600 mt-2 leading-relaxed">{selectedThread.question.body}</p>
                  <div className="flex flex-wrap items-center gap-2 mt-3 text-xs text-gray-400">
                    <span>By {selectedThread.question.author?.name ?? 'Guest'}</span>
                    {selectedThread.question.tags.map((tag) => (
                      <button
                        key={tag}
                        type="button"
                        onClick={() => setTagFilter(tag)}
                        className="px-2 py-0.5 rounded-full bg-purple-100/70 text-purple-700 hover:bg-purple-200 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                      >
                        #{tag}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => handleVote('questions', selectedThread.question.id, 1)}
                    className="inline-flex items-center justify-center h-10 w-10 rounded-xl border border-gray-200 text-gray-600 hover:text-purple-700 hover:border-purple-300 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                    aria-label="Upvote question"
                  >
                    <ArrowUp className="w-4 h-4" aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    onClick={() => handleVote('questions', selectedThread.question.id, -1)}
                    className="inline-flex items-center justify-center h-10 w-10 rounded-xl border border-gray-200 text-gray-600 hover:text-purple-700 hover:border-purple-300 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                    aria-label="Downvote question"
                  >
                    <ArrowDown className="w-4 h-4" aria-hidden="true" />
                  </button>
                  <span className="text-sm font-semibold text-gray-700">
                    {scoreLabel(selectedThread.question.upvotes, selectedThread.question.downvotes)} score
                  </span>
                </div>

                <div className="border-t border-purple-100 pt-4">
                  <h4 className="text-sm font-semibold text-gray-900">Answers ({selectedThread.answers.length})</h4>
                  <div className="mt-3 space-y-4">
                    {selectedThread.answers.map((answer) => (
                      <div key={answer.id} className="rounded-xl border border-gray-200 p-4">
                        <div className="flex items-start justify-between gap-3">
                          <p className="text-xs text-gray-400">{answer.author?.name ?? 'Guest'}</p>
                          <p className="text-xs text-gray-400">{formatRelativeTime(answer.created_at)}</p>
                        </div>
                        <p className="text-sm text-gray-700 mt-2 leading-relaxed">{answer.body}</p>
                        <div className="mt-3 flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => handleVote('answers', answer.id, 1)}
                            className="inline-flex items-center justify-center h-9 w-9 rounded-lg border border-gray-200 text-gray-600 hover:text-purple-700 hover:border-purple-300 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                            aria-label="Upvote answer"
                          >
                            <ArrowUp className="w-4 h-4" aria-hidden="true" />
                          </button>
                          <button
                            type="button"
                            onClick={() => handleVote('answers', answer.id, -1)}
                            className="inline-flex items-center justify-center h-9 w-9 rounded-lg border border-gray-200 text-gray-600 hover:text-purple-700 hover:border-purple-300 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                            aria-label="Downvote answer"
                          >
                            <ArrowDown className="w-4 h-4" aria-hidden="true" />
                          </button>
                          <span className="text-xs font-semibold text-gray-700">
                            {scoreLabel(answer.upvotes, answer.downvotes)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <form onSubmit={handleCreateAnswer} className="border-t border-purple-100 pt-4 space-y-3">
                  <label className="block">
                    <span className="text-xs font-semibold text-gray-600">Your answer</span>
                    <textarea
                      name="answer"
                      required
                      minLength={10}
                      maxLength={4000}
                      rows={4}
                      placeholder="Share an answer grounded in Nepali law."
                      className="mt-2 w-full rounded-xl border border-gray-200 px-3 py-2 text-sm text-gray-700 placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                    />
                  </label>
                  {answerError && (
                    <div className="text-sm text-red-600">{answerError}</div>
                  )}
                  <button
                    type="submit"
                    disabled={answering}
                    className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-purple-600 text-white text-xs font-semibold hover:bg-purple-700 disabled:opacity-60 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                  >
                    {answering ? 'Posting...' : 'Post answer'}
                  </button>
                </form>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-purple-100 bg-white shadow-sm p-5">
            <h3 className="text-sm font-semibold text-gray-900 font-display">Popular tags</h3>
            <div className="mt-3 flex flex-wrap gap-2">
              {filteredTags.length === 0 && (
                <span className="text-xs text-gray-400">Tags appear once questions are posted.</span>
              )}
              {filteredTags.map((tag) => (
                <button
                  key={tag}
                  type="button"
                  onClick={() => setTagFilter(tag)}
                  className="px-3 py-1 rounded-full border border-purple-200 text-xs text-purple-700 hover:bg-purple-50 focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:ring-offset-2 focus-visible:ring-offset-white"
                >
                  #{tag}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-purple-100 bg-white shadow-sm p-5">
            <h3 className="text-sm font-semibold text-gray-900 font-display">Forum guidelines</h3>
            <ul className="mt-3 space-y-2 text-xs text-gray-500">
              <li>Share verifiable context, not personal data.</li>
              <li>Explain which law or act you are referencing.</li>
              <li>Be respectful and cite sources when possible.</li>
            </ul>
          </div>
        </aside>
      </main>
    </div>
  )
}
