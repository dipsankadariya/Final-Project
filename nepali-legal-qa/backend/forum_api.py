from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from auth import verify_access_token
from forum_db import get_conn, init_db, normalize_tags, utc_now


app = FastAPI(title="Nepali Legal QA Forum", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForumAuthor(BaseModel):
    sub: str
    name: str
    picture: Optional[str] = None


class QuestionCreate(BaseModel):
    title: str = Field(min_length=1, max_length=140)
    body: Optional[str] = Field(default=None, max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=6)


class AnswerCreate(BaseModel):
    body: str = Field(min_length=1, max_length=2000)


class VoteRequest(BaseModel):
    value: int


class QuestionOut(BaseModel):
    id: int
    title: str
    body: str
    tags: list[str]
    author: ForumAuthor
    created_at: str
    updated_at: str
    upvotes: int
    downvotes: int
    score: int
    answer_count: int


class AnswerOut(BaseModel):
    id: int
    question_id: int
    body: str
    author: ForumAuthor
    created_at: str
    upvotes: int
    downvotes: int
    score: int


class QuestionDetail(BaseModel):
    question: QuestionOut
    answers: list[AnswerOut]


def get_user_optional(authorization: Optional[str]) -> ForumAuthor:
    if not authorization:
        return ForumAuthor(sub="guest", name="Guest", picture=None)

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    token_data = verify_access_token(parts[1])
    return ForumAuthor(sub=token_data.sub, name=token_data.name, picture=token_data.picture)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/api/forum/health")
def health():
    return {"status": "ok"}


def row_to_question(row) -> QuestionOut:
    return QuestionOut(
        id=row["id"],
        title=row["title"],
        body=row["body"],
        tags=[t for t in (row["tags"] or "").split(",") if t],
        author=ForumAuthor(
            sub=row["author_sub"],
            name=row["author_name"],
            picture=row["author_picture"],
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        upvotes=row["upvotes"],
        downvotes=row["downvotes"],
        score=row["score"],
        answer_count=row["answer_count"],
    )


def row_to_answer(row) -> AnswerOut:
    return AnswerOut(
        id=row["id"],
        question_id=row["question_id"],
        body=row["body"],
        author=ForumAuthor(
            sub=row["author_sub"],
            name=row["author_name"],
            picture=row["author_picture"],
        ),
        created_at=row["created_at"],
        upvotes=row["upvotes"],
        downvotes=row["downvotes"],
        score=row["score"],
    )


@app.get("/api/forum/questions", response_model=list[QuestionOut])
def list_questions(
    sort: str = Query("hot", pattern="^(hot|new|top)$"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: Optional[str] = None,
    tag: Optional[str] = None,
):
    where = []
    params: list[object] = []

    if q:
        like = f"%{q.strip()}%"
        where.append("(title LIKE ? OR body LIKE ?)")
        params.extend([like, like])

    if tag:
        like = f"%{tag.strip().lower()}%"
        where.append("tags LIKE ?")
        params.append(like)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    if sort == "new":
        order_sql = "ORDER BY created_at DESC"
    elif sort == "top":
        order_sql = "ORDER BY (upvotes - downvotes) DESC, created_at DESC"
    else:
        order_sql = "ORDER BY updated_at DESC"

    sql = f"""
        SELECT *, (upvotes - downvotes) as score
        FROM questions
        {where_sql}
        {order_sql}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    with get_conn() as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()

    return [row_to_question(row) for row in rows]


@app.post("/api/forum/questions", response_model=QuestionOut)
def create_question(req: QuestionCreate, authorization: Optional[str] = Header(None)):
    user = get_user_optional(authorization)
    tags = normalize_tags(req.tags)
    now = utc_now()
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")
    body = (req.body or "").strip()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO questions (title, body, tags, author_sub, author_name, author_picture, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                body,
                ",".join(tags),
                user.sub,
                user.name,
                user.picture,
                now,
                now,
            ),
        )
        question_id = cur.lastrowid
        row = cur.execute(
            "SELECT *, (upvotes - downvotes) as score FROM questions WHERE id = ?",
            (question_id,),
        ).fetchone()

    return row_to_question(row)


@app.get("/api/forum/questions/{question_id}", response_model=QuestionDetail)
def get_question(question_id: int):
    with get_conn() as conn:
        cur = conn.cursor()
        question_row = cur.execute(
            "SELECT *, (upvotes - downvotes) as score FROM questions WHERE id = ?",
            (question_id,),
        ).fetchone()
        if not question_row:
            raise HTTPException(status_code=404, detail="Question not found")

        answer_rows = cur.execute(
            "SELECT *, (upvotes - downvotes) as score FROM answers WHERE question_id = ? ORDER BY created_at ASC",
            (question_id,),
        ).fetchall()

    return QuestionDetail(
        question=row_to_question(question_row),
        answers=[row_to_answer(row) for row in answer_rows],
    )


@app.post("/api/forum/questions/{question_id}/answers", response_model=AnswerOut)
def create_answer(
    question_id: int,
    req: AnswerCreate,
    authorization: Optional[str] = Header(None),
):
    user = get_user_optional(authorization)
    now = utc_now()
    body = req.body.strip()
    if not body:
        raise HTTPException(status_code=400, detail="Comment is required")

    with get_conn() as conn:
        cur = conn.cursor()
        exists = cur.execute("SELECT 1 FROM questions WHERE id = ?", (question_id,)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Question not found")

        cur.execute(
            """
            INSERT INTO answers (question_id, body, author_sub, author_name, author_picture, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question_id,
                body,
                user.sub,
                user.name,
                user.picture,
                now,
            ),
        )
        answer_id = cur.lastrowid
        cur.execute(
            """
            UPDATE questions
            SET answer_count = answer_count + 1, updated_at = ?
            WHERE id = ?
            """,
            (now, question_id),
        )
        row = cur.execute(
            "SELECT *, (upvotes - downvotes) as score FROM answers WHERE id = ?",
            (answer_id,),
        ).fetchone()

    return row_to_answer(row)


def apply_vote(table: str, target_id: int, value: int) -> None:
    if value not in (-1, 1):
        raise HTTPException(status_code=400, detail="Vote value must be 1 or -1")

    up_field = "upvotes" if value == 1 else "downvotes"
    sql = f"UPDATE {table} SET {up_field} = {up_field} + 1 WHERE id = ?"

    with get_conn() as conn:
        cur = conn.cursor()
        updated = cur.execute(sql, (target_id,)).rowcount
        if updated == 0:
            raise HTTPException(status_code=404, detail="Target not found")


@app.post("/api/forum/questions/{question_id}/vote")
def vote_question(
    question_id: int,
    req: VoteRequest,
    authorization: Optional[str] = Header(None),
):
    get_user_optional(authorization)
    apply_vote("questions", question_id, req.value)

    with get_conn() as conn:
        conn.execute("UPDATE questions SET updated_at = ? WHERE id = ?", (utc_now(), question_id))

    return {"status": "ok"}


@app.post("/api/forum/answers/{answer_id}/vote")
def vote_answer(
    answer_id: int,
    req: VoteRequest,
    authorization: Optional[str] = Header(None),
):
    get_user_optional(authorization)
    apply_vote("answers", answer_id, req.value)

    with get_conn() as conn:
        row = conn.execute(
            "SELECT question_id FROM answers WHERE id = ?",
            (answer_id,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE questions SET updated_at = ? WHERE id = ?",
                (utc_now(), row["question_id"]),
            )

    return {"status": "ok"}
