# Solutions

Reference implementations for the tutorial exercises live here, one file per chapter, each
exposing a `summarize_*(payload)` function that `tests/test_tutorials.py` checks against the
frozen output in `tests/expected/`.

Students should try the matching `tutorials/` chapter first, then compare against these files when
they are stuck or when they want to inspect one clean implementation path. Each solution imports
directly from the maintained `src/futureaffinity_reproduction` package rather than duplicating logic,
so it's also the easiest way to see how the pieces are meant to compose.
