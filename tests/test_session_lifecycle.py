"""Session lifecycle tests (Stage S-01, Agent A1).

Review gate: 'session lifecycle unit-tested (init -> get -> update -> expire TTL)'.
"""

import time

from curator.session import SessionAgent
from curator.session.schema import PhotoRef, Session, Stage


def test_init_creates_session():
    agent = SessionAgent()
    s = agent.init(user_id=42)
    assert isinstance(s, Session)
    assert s.user_id == 42
    assert s.stage == Stage.NEW
    assert s.platforms == [] and s.photos == []


def test_get_returns_live_session():
    agent = SessionAgent()
    agent.init(7)
    got = agent.get(7)
    assert got is not None and got.user_id == 7


def test_get_missing_returns_none():
    agent = SessionAgent()
    assert agent.get(999) is None


def test_update_mutates_and_persists():
    agent = SessionAgent()
    agent.init(1)
    updated = agent.update(1, vibe="golden hour roadtrip")
    assert updated is not None
    assert updated.vibe == "golden hour roadtrip"
    # persisted: a fresh get sees it
    assert agent.get(1).vibe == "golden hour roadtrip"


def test_update_unknown_field_raises():
    agent = SessionAgent()
    agent.init(1)
    try:
        agent.update(1, not_a_field=True)
        assert False, "expected AttributeError"
    except AttributeError:
        pass


def test_update_bumps_ttl_window():
    agent = SessionAgent()
    s = agent.init(1, ttl_seconds=1000)
    original = s.updated_at
    time.sleep(0.01)
    agent.update(1, vibe="x")
    assert agent.get(1).updated_at > original


def test_expire_removes_session():
    agent = SessionAgent()
    agent.init(5)
    agent.expire(5)
    assert agent.get(5) is None


def test_expire_is_idempotent():
    agent = SessionAgent()
    agent.expire(123)  # no session present; must not raise
    agent.init(123)
    agent.expire(123)
    agent.expire(123)
    assert agent.get(123) is None


def test_ttl_expiry_via_get():
    agent = SessionAgent()
    # 0-second TTL -> immediately stale on next get
    s = agent.init(8, ttl_seconds=0)
    # force updated_at into the past so now > updated_at + 0
    s.updated_at = time.time() - 5
    agent.store.save(s)
    assert agent.get(8) is None  # expired and swept


def test_is_expired_logic():
    s = Session(user_id=1, ttl_seconds=10)
    assert not s.is_expired(now=s.updated_at + 5)
    assert s.is_expired(now=s.updated_at + 11)


def test_sweep_expired_counts():
    agent = SessionAgent()
    a = agent.init(1, ttl_seconds=0)
    b = agent.init(2, ttl_seconds=10_000)
    a.updated_at = time.time() - 100
    agent.store.save(a)
    expired = agent.sweep_expired()
    assert expired == 1
    assert agent.get(2) is not None


def test_add_photo_advances_stage():
    agent = SessionAgent()
    agent.init(1)
    s = agent.add_photo(1, PhotoRef(file_id="abc"))
    assert s.stage == Stage.RECEIVING_PHOTOS
    assert len(s.photos) == 1 and s.photos[0].file_id == "abc"


def test_toggle_platform_multiselect():
    agent = SessionAgent()
    agent.init(1)
    agent.toggle_platform(1, "instagram")
    agent.toggle_platform(1, "tinder")
    assert agent.get(1).platforms == ["instagram", "tinder"]
    agent.toggle_platform(1, "instagram")  # toggle off
    assert agent.get(1).platforms == ["tinder"]
