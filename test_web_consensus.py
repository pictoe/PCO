from wyzer.core.intent_plan import ExecutionResult, ExecutionSummary
from wyzer.core.web_consensus import (
    try_release_year_consensus,
    try_release_date_consensus,
    try_release_month_year_consensus,
    try_build_consensus_reply,
)


def test_release_year_consensus_hits_with_majority() -> None:
    summary = ExecutionSummary(
        ran=[
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League was released in 2015."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Psyonix launched Rocket League in 2015 on PC and PS4."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Release date: 2015. Some other text."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "It received updates in 2016 and 2017."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League came out in 2015."}),
        ]
    )

    c = try_release_year_consensus("When was Rocket League released?", summary)
    assert c is not None
    assert c["year"] == 2015
    assert c["support"] >= 3
    assert c["total"] >= 3

    reply, met = try_build_consensus_reply("When was Rocket League released?", summary)
    assert met is True
    assert "2015" in (reply or "")


def test_release_date_consensus_hits_with_majority_and_phrase_when_did_release() -> None:
    summary = ExecutionSummary(
        ran=[
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League was released on July 7, 2015."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Release date: July 7, 2015."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League launched July 7, 2015 on PS4 and PC."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "It received updates in 2016 and 2017."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League came out in July 2015."}),
        ]
    )

    c = try_release_date_consensus("when did rocket league release", summary)
    assert c is not None
    assert c["date"] == (2015, 7, 7)

    reply, met = try_build_consensus_reply("when did rocket league release", summary)
    assert met is True
    assert "July" in (reply or "")
    assert "2015" in (reply or "")


def test_release_month_year_consensus_fallback() -> None:
    summary = ExecutionSummary(
        ran=[
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League release date: July 2015."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "It was released July 2015 on PS4 and PC."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League came out in July 2015."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Some unrelated info with 2016 mentioned."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Another page: July 2015."}),
        ]
    )

    c = try_release_month_year_consensus("when did rocket league release", summary)
    assert c is not None
    assert c["month_year"] == (2015, 7)

    reply, met = try_build_consensus_reply("when did rocket league release", summary)
    assert met is True
    assert "July" in (reply or "")
    assert "2015" in (reply or "")


def test_release_year_consensus_skips_non_release_questions() -> None:
    summary = ExecutionSummary(
        ran=[ExecutionResult(tool="web_fetch", ok=True, result={"text": "Rocket League was released in 2015."})]
    )
    assert try_release_year_consensus("Tell me about Rocket League", summary) is None


def test_release_year_consensus_fails_when_conflicting() -> None:
    summary = ExecutionSummary(
        ran=[
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Released in 2015."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Released in 2016."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Released in 2017."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Released in 2018."}),
            ExecutionResult(tool="web_fetch", ok=True, result={"text": "Released in 2019."}),
        ]
    )
    assert try_release_year_consensus("What year was it released?", summary) is None
