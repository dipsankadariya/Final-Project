import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowDown, ArrowUp, ChevronLeft, MessageSquare, Scale } from 'lucide-react'
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

export function ForumDetail({ user, onLogout }) {
  const { id } = useParams()
  const [thread, setThread] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [comment, setComment] = useState('')
  const [commentError, setCommentError] = useState(null)
  const [commenting, setCommenting] = useState(false)

  const loadThread = async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await forumRequest(`/questions/${id}`)
      setThread(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadThread()
  }, [id])

  const handleVote = async (value) => {
    if (!thread) return
    try {
      await forumRequest(`/questions/${thread.question.id}/vote`, {
        method: 'POST',
        body: JSON.stringify({ value }),
      })
      setThread((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          question: {
            ...prev.question,
            upvotes: value === 1 ? prev.question.upvotes + 1 : prev.question.upvotes,
            downvotes: value === -1 ? prev.question.downvotes + 1 : prev.question.downvotes,
          },
        }
      })
    } catch (err) {
      setError(err.message)
    }
  }

  const handleCommentSubmit = async (event) => {
    event.preventDefault()
    const trimmed = comment.trim()
    if (!trimmed) {
      setCommentError('Comment is required.')
      return
    }

    setCommenting(true)
    setCommentError(null)

    try {
      const created = await forumRequest(`/questions/${id}/answers`, {
        method: 'POST',
        body: JSON.stringify({ body: trimmed }),
      })
      setThread((prev) => {
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
      setComment('')
    } catch (err) {
      setCommentError(err.message)
    } finally {
      setCommenting(false)
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#fff', fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif" }}>
      <header
        style={{
          borderBottom: '1px solid #E8E6E0',
          background: '#fff',
          position: 'sticky',
          top: 0,
          zIndex: 10,
        }}
      >
        <div
          style={{
            maxWidth: 1400,
            margin: '0 auto',
            padding: '0 32px',
            height: 56,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 34,
                height: 34,
                borderRadius: 10,
                background: '#2D2059',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Scale size={16} color="#fff" aria-hidden="true" />
            </div>
            <div>
              <p style={{ margin: 0, fontSize: 14, fontWeight: 600, color: '#1A1A1A', letterSpacing: '-0.01em' }}>Legal Forum</p>
              <p style={{ margin: 0, fontSize: 11, color: '#9B9690' }}>Question detail</p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Link
              to="/forum"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                fontSize: 12,
                fontWeight: 600,
                color: '#2D2059',
                border: '1.5px solid #2D2059',
                borderRadius: 100,
                padding: '6px 14px',
                textDecoration: 'none',
                letterSpacing: '0.01em',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => e.currentTarget.style.background = '#F0EEF9'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              <ChevronLeft size={13} aria-hidden="true" /> Back
            </Link>
            {user && <UserProfile user={user} onLogout={onLogout} />}
          </div>
        </div>
      </header>

      <main
        style={{
          maxWidth: 720,
          margin: '0 auto',
          padding: '40px 32px',
          display: 'flex',
          flexDirection: 'column',
          gap: 40,
        }}
      >
        {loading && <p style={{ fontSize: 14, color: '#9B9690', margin: 0 }}>Loading question…</p>}
        {error && <p style={{ fontSize: 14, color: '#C0392B', margin: 0, background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: 10, padding: '14px 16px' }}>{error}</p>}

        {!loading && thread && (
          <>
            {/* Question post */}
            <article style={{ display: 'flex', gap: 16 }}>
              {/* Avatar */}
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  background: '#D4CAE8',
                  flexShrink: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 18,
                  fontWeight: 700,
                  color: '#2D2059',
                  textTransform: 'uppercase',
                  overflow: 'hidden',
                }}
              >
                {thread.question.author?.picture ? (
                  <img src={thread.question.author.picture} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  (thread.question.author?.name ?? 'G')[0]
                )}
              </div>

              {/* Content */}
              <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div>
                  <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: '#1A1A1A', letterSpacing: '-0.01em', lineHeight: 1.4 }}>
                    {thread.question.title}
                  </h1>
                  {thread.question.body && (
                    <p
                      style={{
                        margin: '12px 0 0',
                        fontSize: 15,
                        color: '#6B6760',
                        whiteSpace: 'pre-wrap',
                        lineHeight: 1.6,
                      }}
                    >
                      {thread.question.body}
                    </p>
                  )}
                </div>

                {/* Metadata and engagement */}
                <div style={{ display: 'flex', alignItems: 'center', gap: 20, paddingTop: 8 }}>
                  <span style={{ fontSize: 13, color: '#B8B4AD', fontWeight: 500 }}>
                    {thread.question.author?.name ?? 'Guest'}
                  </span>
                  <span style={{ fontSize: 13, color: '#B8B4AD' }}>
                    {formatRelativeTime(thread.question.created_at)}
                  </span>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginLeft: 'auto' }}>
                    <button
                      type="button"
                      onClick={() => handleVote(1)}
                      title="Like this post"
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: 4,
                        background: 'none',
                        border: 'none',
                        color: '#B8B4AD',
                        cursor: 'pointer',
                        fontSize: 13,
                        fontFamily: 'inherit',
                        padding: 0,
                        transition: 'color 0.2s',
                      }}
                      onMouseEnter={e => e.currentTarget.style.color = '#2D2059'}
                      onMouseLeave={e => e.currentTarget.style.color = '#B8B4AD'}
                    >
                      👍 {thread.question.upvotes}
                    </button>

                    <span style={{ fontSize: 13, color: '#B8B4AD' }}>
                      💬 {thread.answers.length}
                    </span>
                  </div>
                </div>
              </div>
            </article>

            {/* Divider */}
            <div style={{ height: '1px', background: '#F0EEE8' }} />

            {/* Answers section */}
            <section style={{ display: 'flex', flexDirection: 'column', gap: 32 }}>
              <h2 style={{ margin: 0, fontSize: 16, fontWeight: 700, color: '#1A1A1A', display: 'flex', alignItems: 'center', gap: 8 }}>
                <MessageSquare size={18} aria-hidden="true" />
                {thread.answers.length} {thread.answers.length === 1 ? 'answer' : 'answers'}
              </h2>

              {thread.answers.length > 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                  {thread.answers.map((answer, idx) => (
                    <div key={answer.id} style={{ display: 'flex', gap: 16, paddingBottom: idx < thread.answers.length - 1 ? 24 : 0, borderBottom: idx < thread.answers.length - 1 ? '1px solid #F0EEE8' : 'none' }}>
                      {/* Avatar */}
                      <div
                        style={{
                          width: 40,
                          height: 40,
                          borderRadius: '50%',
                          background: '#D4CAE8',
                          flexShrink: 0,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: 14,
                          fontWeight: 700,
                          color: '#2D2059',
                          textTransform: 'uppercase',
                          overflow: 'hidden',
                        }}
                      >
                        {answer.author?.picture ? (
                          <img src={answer.author.picture} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        ) : (
                          (answer.author?.name ?? 'G')[0]
                        )}
                      </div>

                      {/* Answer body */}
                      <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                          <span style={{ fontSize: 13, color: '#1A1A1A', fontWeight: 600 }}>
                            {answer.author?.name ?? 'Guest'}
                          </span>
                          <span style={{ fontSize: 12, color: '#B8B4AD' }}>
                            {formatRelativeTime(answer.created_at)}
                          </span>
                        </div>
                        <p style={{ margin: 0, fontSize: 14, color: '#6B6760', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                          {answer.body}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {thread.answers.length === 0 && (
                <p style={{ fontSize: 14, color: '#9B9690', margin: 0, padding: '24px 0' }}>No answers yet. Be the first to respond!</p>
              )}
            </section>

            {/* Add answer form */}
            <section style={{ paddingTop: 24, borderTop: '1px solid #F0EEE8' }}>
              <form onSubmit={handleCommentSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 700, color: '#1A1A1A' }}>Add your answer</span>
                  <textarea
                    value={comment}
                    onChange={(event) => setComment(event.target.value)}
                    rows={4}
                    style={{
                      borderRadius: 8,
                      border: '1px solid #E0DDD7',
                      padding: '10px 12px',
                      fontSize: 13,
                      color: '#1A1A1A',
                      outline: 'none',
                      fontFamily: 'inherit',
                      background: '#FAFAF8',
                      resize: 'vertical',
                      transition: 'border-color 0.2s, box-shadow 0.2s',
                      lineHeight: 1.5,
                    }}
                    onFocus={e => {
                      e.currentTarget.style.borderColor = '#2D2059'
                      e.currentTarget.style.boxShadow = '0 0 0 2px rgba(45,32,89,0.08)'
                    }}
                    onBlur={e => {
                      e.currentTarget.style.borderColor = '#E0DDD7'
                      e.currentTarget.style.boxShadow = 'none'
                    }}
                    placeholder="Share your thoughts or answer..."
                    required
                  />
                </label>
                {commentError && (
                  <p style={{ margin: 0, fontSize: 12, color: '#C0392B', background: '#FEF2F2', border: '1px solid #FECACA', borderRadius: 8, padding: '10px 12px' }}>
                    {commentError}
                  </p>
                )}
                <button
                  type="submit"
                  disabled={commenting}
                  style={{
                    alignSelf: 'flex-start',
                    padding: '10px 16px',
                    borderRadius: 8,
                    background: commenting ? '#ADA9C7' : '#2D2059',
                    color: '#fff',
                    fontSize: 13,
                    fontWeight: 700,
                    border: 'none',
                    cursor: commenting ? 'not-allowed' : 'pointer',
                    fontFamily: 'inherit',
                    letterSpacing: '0.01em',
                    transition: 'background 0.2s, transform 0.1s',
                  }}
                  onMouseEnter={e => { if (!commenting) e.currentTarget.style.background = '#1E154A' }}
                  onMouseLeave={e => { if (!commenting) e.currentTarget.style.background = '#2D2059' }}
                  onMouseDown={e => { if (!commenting) e.currentTarget.style.transform = 'scale(0.96)' }}
                  onMouseUp={e => { if (!commenting) e.currentTarget.style.transform = 'scale(1)' }}
                >
                  {commenting ? 'Posting…' : 'Post answer'}
                </button>
              </form>
            </section>
          </>
        )}
      </main>
    </div>
  )
}
