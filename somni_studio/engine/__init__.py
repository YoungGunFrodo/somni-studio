"""Running a model for chat, with telemetry the UI turns into light.

All model work happens on one worker thread (worker.py): models are not thread-safe, SQLite
connections belong to the thread that opened them, and one GPU job at a time is all a consumer
card can do well anyway.
"""
