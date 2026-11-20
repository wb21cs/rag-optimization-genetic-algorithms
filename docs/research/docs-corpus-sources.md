# Docs corpus: sources, licences, extraction, passage IDs

Research for ticket #3. Compiled 2026-11-20. Counts and byte sizes measured with the GitHub
trees API on that date; licences read from each repository's own licence file.

## Summary table

| Corpus | Repo | Docs path | Format | Build tool | Files | Source bytes | ≈ tokens (bytes/4) | Licence | Redistribute extracted text? |
|---|---|---|---|---|---|---|---|---|---|
| Python | `python/cpython` | `Doc/` | reST | Sphinx (`Doc/conf.py`) | 561 `.rst` | 13,532,901 | ≈ 3.4M | PSF-2.0 | Yes, with the PSF notice |
| Docker | `docker/docs` | `content/` | Markdown (+ Hugo shortcodes) | Hugo (`hugo.yaml`, `go.mod`) | 992 `.md` | 7,428,818 | ≈ 1.9M | Apache-2.0 | Yes, with LICENSE/NOTICE; no trademark grant |
| PostgreSQL | `postgres/postgres` | `doc/src/sgml/` | DocBook SGML | DocBook XSLT (`Makefile`) | 429 `.sgml` | 12,476,422 | ≈ 3.1M | PostgreSQL License | Yes, with the copyright notice |
| Git | `git/git` | `Documentation/` | AsciiDoc | asciidoctor (`Makefile`) | 956 `.adoc` | 5,661,269 | ≈ 1.4M | GPL-2.0 (only) | Yes, but derivative must stay GPL-2.0 |

Total: **39,099,410 source bytes ≈ 9.8M tokens ≈ 19k chunks at 512 tokens.**

Method: `gh api repos/<repo>/git/trees/<branch>?recursive=1`, filtered by path prefix and extension,
summing `.size`. Token estimate uses 4 bytes/token, which **overstates** prose density because reST,
SGML and AsciiDoc markup is stripped at extraction — treat 9.8M as an upper bound, not a measurement.

## Per-corpus detail

### Python — `python/cpython`, `Doc/`

- Authoritative source: <https://github.com/python/cpython/tree/main/Doc>. The docs live in the main
  CPython repo; there is no separate docs repo. `Doc/conf.py` confirms Sphinx
  (<https://github.com/python/cpython/blob/main/Doc/conf.py>).
- Licence: CPython's `LICENSE` (<https://github.com/python/cpython/blob/main/LICENSE>) records the
  PSF Licence 2.0 for Python 2.1 onward. The published docs state: "This documentation is licensed
  under the Python Software Foundation License Version 2"
  (<https://docs.python.org/3/license.html>). Redistribution of the docs with the notice is permitted.
- Structure: one `.rst` file per page, `toctree` defines hierarchy, `.. versionadded::` /
  `.. deprecated::` directives are inline. Includes are used for shared snippets, so raw-file parsing
  alone will miss substituted content.
- Extraction: **build, then convert.** `sphinx-build -b text Doc build/text` produces clean text with
  heading underlines and no navigation, and resolves includes/substitutions for you. The `-b html`
  route plus HTML→text is the fallback if the text builder mangles code blocks.
- Version to pin: the `3.13` branch of `cpython` (stable, matches what readers use), pinned by commit
  SHA. `main` documents 3.15-dev and drifts.

### Docker — `docker/docs`, `content/`

- Source: <https://github.com/docker/docs>, Markdown under `content/`, built by Hugo
  (`hugo.yaml` at repo root). Licence: Apache-2.0
  (<https://github.com/docker/docs/blob/main/LICENSE>).
- Apache-2.0 §6 explicitly withholds trademark rights, which matters if the demo uses Docker marks.
- Structure: `content/manuals/<product>/...` with front-matter (`title`, `description`, `weight`) and
  **Hugo shortcodes** (`{{< tabs >}}`, `{{< include >}}`, callouts). Shortcodes carry real content
  (tabbed instructions, notes), so a naive Markdown parse silently drops material. Resolve shortcodes
  or build the site.
- Extraction: **parse Markdown + handle shortcodes**, or `hugo` build + HTML→text. Parsing is cheaper
  and loses only shortcode-rendered content; building is more faithful. Recommended: build, because
  the Docker guides use tabs heavily.
- Version to pin: `docker/docs` has a single `main`; pin a commit SHA. Note that product docs change
  frequently — record the SHA in the corpus manifest and in the report.

### PostgreSQL — `postgres/postgres`, `doc/src/sgml/`

- Source: <https://github.com/postgres/postgres/tree/master/doc/src/sgml>, DocBook SGML built by the
  `Makefile` in that directory with DocBook XSL stylesheets.
- Licence: PostgreSQL License
  (<https://github.com/postgres/postgres/blob/master/COPYRIGHT>), a permissive BSD-style licence that
  explicitly covers "this software and its documentation", requiring only that the copyright notice
  and two paragraphs travel with copies.
- Structure: one `.sgml` file per chapter plus `ref/*.sgml` man-page sources. Entities (`&nbsp;`,
  `&version;`) and `<xref>` links are resolved at build time.
- Extraction: **build HTML** (`make -C doc/src/sgml html`, or use the published
  <https://www.postgresql.org/docs/17/> HTML) then HTML→text. Parsing SGML directly means
  reimplementing entity resolution; not worth it.
- Version: pin the `REL_17_STABLE` branch (or the 17.x tag) by commit SHA. The docs URL embeds the
  major version, so the corpus must record which one.

### Git — `git/git`, `Documentation/`

- Source: <https://github.com/git/git/tree/master/Documentation>, AsciiDoc (`.adoc`) built by
  `asciidoctor` via the `Makefile` into man pages and HTML.
- Licence: GPL-2.0 only — `COPYING` states "the only valid version of the GPL as far as this project
  is concerned is _this_ particular version of the license (ie v2, not v2.2 or v3.x…)"
  (<https://github.com/git/git/blob/master/COPYING>). The docs ship under the same licence.
  Consequence: extracted Git text in a public repo is a GPL-2.0 derivative; the repo would need to
  carry the GPL-2.0 text and the extraction scripts would inherit it. **The clean move is a downloader
  script pinned to a commit, with the text generated locally and never committed.**
- Structure: one `.adoc` per command/topic, `linkgit:` cross-references, `include::` for shared
  fragments, heavy use of definition lists and literal blocks.
- Extraction: **asciidoctor → HTML → text**, or parse `.adoc` directly for headings and literal
  blocks. Git's docs are unusually regular; direct parsing is viable and dependency-free.
- Version: pin a release tag (e.g. `v2.47.0`) or `maint`, by commit SHA.

## Passage identity

Use a two-part ID:

```
<corpus>/<path-without-extension>#<heading-slug>[/<ordinal>]@<sha256-12>
```

Example: `python/library/os#process-management/2@9f2c1a7b3d4e`.

- `<corpus>/<path>#<heading-slug>` is the **stable, human-readable** part; it survives text edits and
  is what gold labels reference.
- `/<ordinal>` disambiguates sibling chunks under one heading.
- `@<sha256-12>` is the **content hash**, used only for cache invalidation and change detection.

Pitfalls to handle explicitly: heading renames (gold labels break — keep a redirect map), duplicate
headings under one page, translated docs (exclude non-English trees), versioned docs (Python
`3.13/` vs `3.14/`, Docker product versions), and Git's many `ref/*.sgml` man-page stubs that
duplicate main pages.

## Redistribution: one rule for all four

Ship **`corpus/fetch.py`** (pinned commit SHAs + checksums) and **`corpus/extract.py`**
(deterministic, version-pinned toolchain), and **do not commit extracted text**. Reasons:

1. Git's GPL-2.0 makes committed text a licensing question you do not need to answer in a course
   repo.
2. 39 MB of source → a repo bloat that every clone pays for.
3. A downloader is reproducible: the manifest records commit SHA + file hashes, so anyone can
   regenerate the exact corpus.

If the marker requires the corpus to be present offline, ship the Docker + PostgreSQL + Python
extracts (permissive) and only the Git *downloader*.

## Recommendation

1. Pin each corpus by **commit SHA**, not branch, in `corpus/manifest.json` (repo, sha, path, licence,
   file count, byte count, extraction tool + version).
2. Extract by **building with each project's own toolchain, then one shared HTML→Markdown normaliser**
   (keep `h1..h6`, fenced code, tables as pipe tables, drop nav/footer/images/ads). Python may skip
   HTML and use `sphinx-build -b text`.
3. Normalise to one Markdown dialect with heading levels preserved and code fences intact — this is
   what the chunker sees, and it makes chunk boundaries predictable across all four corpora.
4. Commit the fetcher + extractor, not the text; record the manifest in the report's environment
   table.

## Confidence / gaps

High: repository paths, default branches, licence identities, build tools, file counts, byte sizes —
all read from the repositories on 2026-11-20. Medium: the token estimate (bytes/4 over markup-heavy
sources) and the chunk count. Low / unverified: whether Docker's Hugo shortcodes can be resolved
without a full site build (measure on the first extraction run); whether Python's `-b text` builder
preserves code-block indentation acceptably (spot-check 20 pages); the exact GPL-2.0 obligations for
a corpus of extracted Git prose (not legal advice — the downloader sidesteps it).
