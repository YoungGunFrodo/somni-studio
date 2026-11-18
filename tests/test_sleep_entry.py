"""The sleep narration wraps Somni's real stage functions (in a child process, so the patching
cannot leak into other tests) and must report dream questions and their results in order."""
import json
import subprocess
import sys
import textwrap
import unittest

from somni_studio.settings import STUDIO_ROOT

from .helpers import SOMNI_REPO, require_somni

SCRIPT = textwrap.dedent('''
    import sys
    sys.path[:0] = [{repo!r}, {studio!r}]
    from somni_studio.somni import sleep_entry
    sleep_entry.instrument()
    from sleepai.sleep import dreaming, pipeline

    class M:
        def __init__(self, i, value):
            self.id, self.value = i, value

    class Teacher:
        def dream_questions(self, memory, n):
            return [f"q{{k}} about {{memory.id}}" for k in range(n)]

    asked = dreaming._dream_questions(Teacher(), [M("m1", "a"), M("m2", "b")], 2)
    for (memory, _q), answer in zip(asked, ["a", "x", "b", "b"]):
        dreaming.exact(answer, memory.value)
    assert pipeline.run_dreaming.__name__ == "wrapper"
''')


class SleepEntryTest(unittest.TestCase):
    def test_dreams_are_narrated(self):
        require_somni(self)
        code = SCRIPT.format(repo=str(SOMNI_REPO), studio=str(STUDIO_ROOT))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        marks = [json.loads(line[len("[studio] "):]) for line in out.stdout.splitlines()
                 if line.startswith("[studio] ")]
        self.assertEqual(marks[0], {"phase": "dreaming", "state": "asking", "questions": 4, "memories": ["m1", "m2"]})
        self.assertEqual(marks[1]["state"], "checked")
        self.assertEqual(marks[1]["results"], [["m1", True], ["m1", False], ["m2", True], ["m2", True]])
        self.assertEqual(marks[1]["right"], 3)


if __name__ == "__main__":
    unittest.main()
