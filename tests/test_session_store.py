"""Session store tests (Stage S-01, Agent A1).

Review gate: 'SQLite fallback exercised by a test (write a session, read it back
through the SQLite path)'.
"""

from curator.session import InMemorySessionStore, SessionAgent, SQLiteSessionStore
from curator.session.schema import PhotoRef, Session, Stage


def _sample_session() -> Session:
    s = Session(user_id=99)
    s.platforms = ["instagram", "hinge"]
    s.vibe = "candid summer"
    s.constraints = ["no_location_names", "max_chars:100"]
    s.photos = [PhotoRef(file_id="f1", width=800, height=1000), PhotoRef(file_id="f2")]
    s.stage = Stage.RECEIVING_PHOTOS
    return s


def test_inmemory_roundtrip():
    store = InMemorySessionStore()
    s = _sample_session()
    store.save(s)
    got = store.load(99)
    assert got is not None and got.user_id == 99
    assert got.platforms == ["instagram", "hinge"]


def test_sqlite_fallback_roundtrip_in_memory_db():
    """Write through the SQLite path and read the full record back."""
    store = SQLiteSessionStore(path=":memory:")
    try:
        original = _sample_session()
        store.save(original)
        loaded = store.load(99)
        assert loaded is not None
        assert loaded.user_id == 99
        assert loaded.platforms == ["instagram", "hinge"]
        assert loaded.vibe == "candid summer"
        assert loaded.constraints == ["no_location_names", "max_chars:100"]
        assert [p.file_id for p in loaded.photos] == ["f1", "f2"]
        assert loaded.photos[0].width == 800
        assert loaded.stage == Stage.RECEIVING_PHOTOS
    finally:
        store.close()


def test_sqlite_persists_across_store_instances(tmp_path):
    """Survive 'restart': a new store instance reads what an old one wrote."""
    db = str(tmp_path / "sessions.sqlite")
    s1 = SQLiteSessionStore(path=db)
    s1.save(_sample_session())
    s1.close()

    s2 = SQLiteSessionStore(path=db)
    try:
        loaded = s2.load(99)
        assert loaded is not None and loaded.vibe == "candid summer"
        assert 99 in s2.all_user_ids()
    finally:
        s2.close()


def test_sqlite_delete():
    store = SQLiteSessionStore(path=":memory:")
    try:
        store.save(_sample_session())
        store.delete(99)
        assert store.load(99) is None
        assert store.all_user_ids() == []
    finally:
        store.close()


def test_session_agent_over_sqlite_lifecycle():
    """SessionAgent works identically over the SQLite backend."""
    store = SQLiteSessionStore(path=":memory:")
    agent = SessionAgent(store=store)
    try:
        agent.init(5)
        agent.update(5, vibe="test")
        assert agent.get(5).vibe == "test"
        agent.expire(5)
        assert agent.get(5) is None
    finally:
        agent.close()
