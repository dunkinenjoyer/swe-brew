<!--
# SOURCE ATTRIBUTION
# ==================
# The bulk of this prompt is a **verbatim lift** of SWT-Bench's BASE_PROMPT
# (`swt-bench/dataset/swt_bench.py:91-177` in the swt-bench repo at
# github.com/logic-star-ai/SWT-Bench). Localised to Java / JUnit 4:
#
#   - Python demo/file.py → Java demo/BarTest.java in the worked example.
#   - "the issue of the user" framing kept.
#   - Unified-diff explainer, hunk header semantics, and the
#     before/after file-listing convention: kept verbatim.
#
# The only additions on top of BASE_PROMPT are:
#   - A "Primary reference: the code patch" section that hands the LLM
#     the actual gold code patch as the oracle (SWT-Bench's protocol
#     depends on this; we make it explicit in the prompt).
#   - A "Secondary reference: PRD" section (business context, demoted
#     from checklist per user alignment: patch is primary).
#
# Rendered at inference time via ``str.format(**vars)`` with keys:
#   ``code_patch``, ``changed_files_list``, ``changed_count``,
#   ``prd``, ``prd_len``, ``repo_short``, ``max_tests``.
-->

You are a senior Java engineer writing JUnit 4 tests that pin
down the exact behaviour change captured by a code patch.

## Primary reference — the code patch

**The patch below is the ground truth.** Every hunk describes one
concrete behaviour change: a modified method body, a new branch, a
changed return-value shape, an added state-machine transition, an
altered cache-key layout, a new config toggle, a new callback path, a
new invariant. **Your job is to write tests such that, for each of those
hunks, there is at least one test that fails against the pre-patch
codebase and passes after the patch is applied.**

Think of it as: *"if I removed this hunk from the patch, which test
would break?"* — that's the test you should write for that hunk.

### Code patch (git-native unified diff, apply-clean against base_commit)

```diff
{code_patch}
```

### Changed files ({changed_count} total; test files excluded — you write those)

{changed_files_list}

## Secondary reference — the product requirements document

Use this only for **business context and naming**: what real-world
scenario motivated the change, so your tests can be named
descriptively (e.g. ``shouldSubmitTryOnOrderWhenAutoOpenIsTrue`` rather
than ``testCase1``). **Do NOT treat the PRD as a checklist of features to
cover** — the patch already tells you exactly what changed. If a
requirement in the PRD is not reflected in the patch, it is out of
scope for this test file.

### PRD (Chinese, human-authored, ~{prd_len} chars)

```
{prd}
```

## Task
<!-- SWT-Bench BASE_PROMPT lift begins here.
     Original: swt-bench/dataset/swt_bench.py:91-177 (verbatim except for the demo file). -->

Please generate test cases that check whether the implemented solution
above (shown in the code patch) resolves the change captured by that
patch. Present the test cases in unified diff formatting.

The general format of a diff is the unified output format, described as follows.
The unified output format starts with a two-line header, which looks like this:

--- from-file
+++ to-file

Next come one or more hunks of differences; each hunk shows one area where the files differ. Unified format hunks look like this:

@@ from-file-line-numbers to-file-line-numbers @@
 line-from-either-file
 line-from-either-file…

If a hunk contains just one line, only its start line number appears. Otherwise its line numbers look like 'start,count'. An empty hunk is considered to start at the line that follows the hunk.

If a hunk and its context contain two or more lines, its line numbers look like 'start,count'. Otherwise only its end line number appears. An empty hunk is considered to end at the line that precedes the hunk.

The lines common to both files begin with a space character. The lines that actually differ between the two files have one of the following indicator characters in the left print column:

'+' A line was added here to the first file.
'-' A line was removed here from the first file.

Insertion can only be done at the end or beginning of the file, indicated by EOF or BOF respectively.

As an example for a diff, consider the following two versions of the same file, once before and once after a change.
The original version of the file was as follows.
[start of some-module/src/test/java/demo/BarTest.java]
1 package demo;
2
3 import org.junit.Test;
4 import static org.junit.Assert.assertEquals;
5
6 public class BarTest {
7     @Test
8     public void testAdd() {
9         Bar b = new Bar();
10        assertEquals(2, b.add(1, 1));
11    }
12 }
[end of some-module/src/test/java/demo/BarTest.java]

The diff that fixes a bug in ``add`` (expected 3 for ``add(1, 2)``) and adds a new test for ``multiply`` is as follows.
This diff changes the first file into the second file.
```diff
--- a/some-module/src/test/java/demo/BarTest.java
+++ b/some-module/src/test/java/demo/BarTest.java
@@ -8,4 +8,10 @@
     @Test
     public void testAdd() {
         Bar b = new Bar();
-        assertEquals(2, b.add(1, 1));
+        assertEquals(3, b.add(1, 2));
+    }
+
+    @Test
+    public void testMultiply() {
+        Bar b = new Bar();
+        assertEquals(6, b.multiply(2, 3));
     }
 }
```

The new version of the file is as follows.
[start of some-module/src/test/java/demo/BarTest.java]
1 package demo;
2
3 import org.junit.Test;
4 import static org.junit.Assert.assertEquals;
5
6 public class BarTest {
7     @Test
8     public void testAdd() {
9         Bar b = new Bar();
10        assertEquals(3, b.add(1, 2));
11    }
12
13    @Test
14    public void testMultiply() {
15        Bar b = new Bar();
16        assertEquals(6, b.multiply(2, 3));
17    }
18 }
[end of some-module/src/test/java/demo/BarTest.java]

As you can see, you need to indicate the approximate line numbers, function name and the path and file name you want to change,
but there can be as many independent blocks of changes as you need. You may also apply changes to several files.
Apply as much reasoning as you please and see necessary. The format of the solution is fixed and has to follow the unified diff format.
Make sure to implement only test cases and don't try to fix the issue itself.

<!-- SWT-Bench BASE_PROMPT lift ends here. -->

## Java / JUnit 4 constraints

- Test files must live under **``src/test/java/``** (any module) — a filename ending
  in ``Test.java``, ``Tests.java``, ``IT.java``, or ``ITCase.java``.
- Import ``org.junit.Test`` (JUnit 4). Do NOT use Jupiter (JUnit 5).
- If mocking is required, prefer ``org.mockito.Mockito`` (already on classpath).
- **Do not** modify any file outside ``src/test/java/``.
- **Do not** modify or duplicate code from the code patch — you are writing *tests*, not the solution.
- Cap output at {max_tests} distinct test methods across all classes.

### Symbol hygiene

Prefer symbols visible in the code patch and its imports. If you must
reference a symbol from elsewhere in the repository, be sure it exists
at the pre-patch base_commit — the sandbox will run compile-check and
loop back to you if you invent something.

Standard library, JUnit 4 (``org.junit.*``), Mockito
(``org.mockito.*``), and Assertions (``org.junit.Assert.*``) are always
available.

Wrap each diff in a fenced ```` ```diff ```` block. Emit nothing else — no prose
between blocks, no closing summary, no code fences around anything except the diff.
