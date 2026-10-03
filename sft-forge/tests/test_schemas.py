"""Record/Seed behaviour and the training-format views."""

from __future__ import annotations

import pytest

from sft_forge.schemas import STATUS_ACCEPTED, STATUS_REJECTED, Record, Seed


def make_record(**kwargs) -> Record:
    payload = {
        "instruction": "How do I bake bread?",
        "response": "Mix flour, water, salt and yeast, then let it rise.",
        "system": "You are helpful.",
        "domain": "cooking & food",
    }
    payload.update(kwargs)
    return Record(**payload)


def test_seed_roundtrip_and_describe():
    seed = Seed(topic="bread baking", domain="cooking & food", constraints=["be brief"])
    assert seed.id.startswith("seed_")
    restored = Seed.from_dict(seed.to_dict())
    assert restored == seed
    assert "Topic: bread baking" in seed.describe()
    assert "Constraints: be brief" in seed.describe()


def test_seed_id_is_content_addressed_but_overridable():
    assert Seed(topic="a").id == Seed(topic="a").id
    assert Seed(topic="a").id != Seed(topic="b").id
    assert Seed(topic="a", id="custom").id == "custom"


def test_record_messages_and_scores():
    record = make_record()
    assert [m["role"] for m in record.messages] == ["system", "user", "assistant"]
    assert make_record(system=None).messages[0]["role"] == "user"
    record.scores = {"overall": 4.5}
    assert record.overall_score == 4.5
    assert make_record().overall_score is None


def test_record_reject_is_idempotent_and_accumulates_reasons():
    record = make_record()
    record.reject("too_short").reject("too_short").reject("duplicate")
    assert record.status == STATUS_REJECTED
    assert record.reject_reasons == ["too_short", "duplicate"]
    assert not record.is_accepted


def test_record_dict_roundtrip():
    record = make_record(scores={"overall": 4.0}, meta={"topic": "x"}, reject_reasons=["nope"])
    restored = Record.from_dict(record.to_dict())
    assert restored.instruction == record.instruction
    assert restored.scores == record.scores
    assert restored.meta == record.meta
    assert restored.id == record.id
    assert restored.created_at == record.created_at


def test_record_ignores_unknown_fields_on_load():
    record = Record.from_dict({"instruction": "x", "response": "y", "future_field": 1})
    assert record.instruction == "x"


@pytest.mark.parametrize(
    "fmt,key",
    [("chatml", "messages"), ("alpaca", "instruction"), ("sharegpt", "conversations"), ("pairs", "completion")],
)
def test_export_views(fmt, key):
    view = make_record().export_view(fmt)
    assert key in view
    assert view["metadata"]["domain"] == "cooking & food"


def test_export_view_can_drop_system_and_scores():
    chatml = make_record().export_view("chatml", include_system=False)
    assert [m["role"] for m in chatml["messages"]] == ["user", "assistant"]
    pairs = make_record().export_view("pairs", include_system=False)
    assert pairs["prompt"] == "How do I bake bread?"
    assert "metadata" not in make_record().export_view("chatml", include_scores=False)


def test_export_view_rejects_unknown_format():
    with pytest.raises(ValueError, match="unknown export format"):
        make_record().export_view("yaml")


def test_alpaca_view_includes_system_when_present():
    assert make_record().export_view("alpaca")["system"] == "You are helpful."
    assert "system" not in make_record(system=None).export_view("alpaca")


def test_status_constant_flow():
    record = make_record()
    record.status = STATUS_ACCEPTED
    assert record.is_accepted
