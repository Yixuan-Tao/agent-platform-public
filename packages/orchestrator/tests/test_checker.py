# Test values are synthetic fixtures for software behavior.
from __future__ import annotations

from agent_platform.orchestrator.checker import Checker, ReflectionLoop


class PassAfterLLM:
    def __init__(self, fail_first=True):
        self.fail_first = fail_first
        self.calls = 0

    async def chat(self, messages, temperature=0.7, **kw):
        self.calls += 1
        return "FAIL: 缺少细节" if self.fail_first and self.calls == 1 else "PASS"


async def test_checker_pass():
    checker = Checker(PassAfterLLM(fail_first=False))
    passed, reason = await checker.check("q", "a")
    assert passed is True


async def test_reflection_retries_once():
    llm = PassAfterLLM(fail_first=True)
    checker = Checker(llm)
    loop = ReflectionLoop(checker, max_iterations=2)

    async def generator(feedback):
        return "answer-v" + ("2" if feedback else "1")

    answer, attempts = await loop.run("task", generator)
    assert attempts == 2
    assert answer == "answer-v2"


async def test_reflection_stops_after_max():
    class AlwaysFail:
        async def chat(self, messages, temperature=0.7, **kw):
            return "FAIL: 一直不行"

    loop = ReflectionLoop(Checker(AlwaysFail()), max_iterations=2)

    async def generator(feedback):
        return "x"

    answer, attempts = await loop.run("task", generator)
    assert attempts == 2
    assert answer == "x"
