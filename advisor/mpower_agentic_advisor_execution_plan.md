# mPower Advisor — High-Level Execution Plan

**Status:** Draft for review
**Author:** drafted with Copilot CLI
**Date:** 2026-07-27
**Scope:** Strategy / execution plan. *Not* an implementation or coding plan.

---

## 1. Vision

An **agentic assistant that sits beside every mPower run** and compresses the gap between
"the run finished" and "I know what to do next."

Today an experienced AE/CAD engineer spends hours reading `mpower.log`, cross-referencing
warning codes, hunting for the roll-up counts, and correlating IR hotspots with upstream
input problems. The reference artifact
(`prototype/mpower_result_summary.html`) proves that this analysis is
*reproducible and mostly mechanical* — it just isn't automated.

**End-state capability:**

| Capability | User question answered |
|---|---|
| Run digest / dashboard | "Did my run actually do what I asked, and is the result trustworthy?" |
| Input & setup validation | "What did mPower silently work around because my inputs were incomplete?" |
| Error triage | "It failed / it's wrong — why, and what do I change?" |
| EM/IR debug assistance | "This instance violates. What's causing it and what's the fix?" |
| Performance advisor | "It took 11 hours. Where did the time go and how do I make it faster?" |
| Setup assistant | "How do I set this flow up correctly in the first place?" |

We deliberately start at the top of that table and work down: the digest is the highest
value-per-unit-effort and it *builds the data substrate* that every later capability needs.

---

## 2. Grounding — what was verified in the environment

This plan is calibrated against real artifacts, not assumptions. Findings that drive design:

### 2.1 The log corpus is small and LLM-friendly

Reference run (`BRCM/N3E rts_top`, dynamic vectored, 11h18m, 61.8M nodes):

| Artifact | Size | Lines |
|---|---|---|
| `mpower.log` | 67 KB | 814 |
| `mpower.warning.log` | 543 KB | 3,778 |
| `mpower.lib.log` | 371 KB | 3,071 |
| `mpower.error.log` | 455 B | 4 |
| `mpower.cmd` | 11 KB | — |

**Implication:** the *primary* log for a massive multi-million-instance design fits
comfortably in a single LLM context window. This is a rare and very favorable property —
it means high-quality reasoning is achievable without heavy retrieval engineering on the
log side. Effort goes into *knowledge* and *correlation*, not log chunking.

### 2.2 A structured message catalog already exists (major asset)

`src/dev/tclcommands/ui/mhelp/*.mhelp` — **1,767 message codes**, each with:

```
<message> PWR-258
<key>     PWR-258
<details> This warning occurs when ccsp waveform's first or last point is not
          close to the corresponding leakage value.
<type>    Warning
<text>    CCSP: leakage mismatch. cell: %1, pg pin: %2, ...
<action>  No_Information
```

Plus **390 `.help` command files** in `ghelp/`, and **826 regression `runScript`s** across 33
categories under `regression/short/` that encode canonical, known-good flow usage.

**This is the knowledge base.** We do not have to invent it — we have to *normalize and
enrich* it.

### 2.3 Two significant, cheap-to-fix knowledge gaps

| Gap | Measurement | Impact |
|---|---|---|
| `<action>` field is empty | **976 of 1,733 (56%)** are `No_Information` | The single most useful field for an advisor — "what do I do about it" — is missing for over half of all codes. |
| `<type>` field is not normalized | ~20 distinct spellings (`Warning`, `Warning `, `WARNING`, `warning`, `Error message`, `Message is not used.`, …) | Blocks reliable severity-based triage until normalized. |

Closing the `<action>` gap is arguably the **highest ROI work item in the entire program**,
and it benefits mPower users directly *even if the agent is never built*.

### 2.4 Warning counts are a correctness trap — rule discovered and validated

This was investigated in depth because it is the single easiest place to be badly wrong.

**The trap has three layers:**

1. `mpower.warning.log` is capped at **10 messages per code per emitting context** (per
   liberty file, per phase, per MPI rank). Counting lines there gives 621 for `PWR-258` —
   against a true total of **17,568,309**. That is a **28,000×** under-report.
2. True counts appear only as roll-up lines in `mpower.log`:
   `Info: 10 (out of 17568309) 'PWR-258' messages were displayed [UI]`
3. **A code emits multiple roll-up lines** (`PWR-258` has 6, spread across phases), and some
   are per-rank (`... displayed by rank=0 process`). **Summing them double-counts.**

**Validated extraction rule:** *take the **maximum** "out of N" value per code across all
roll-up lines* — the counters are cumulative, so the largest is the final total.

Verified against the hand-authored reference report — **all 13 codes it lists match exactly**:

| Code | max-rule | Reference report | Naive sum (wrong) |
|---|---:|---:|---:|
| PWR-258 | 17,568,309 | 17,568,309 ✓ | 21,430,060 ✗ |
| PWR-259 | 1,203,979 | 1,203,979 ✓ | 1,430,638 ✗ |
| PWR-170 | 966,759 | 966,759 ✓ | 1,169,418 ✗ |
| EXT-223 | 32,075 | 32,075 ✓ | 63,400 ✗ |
| PWR-245 | 12,619 | 12,619 ✓ | 13,336 ✗ |
| PWR-306 | 1,009 | 1,009 ✓ | 2,018 ✗ |

**This rule is a concrete, transferable deliverable of the planning exercise** — it should
become a Phase 1 requirement with a dedicated regression test.

**And it already demonstrates the value proposition.** Applying the rule surfaced two codes
the hand-authored report **missed entirely**:

- **`PWR-268` — 44,164 occurrences:** *"TWF: Pin %2 of instance %3 is not connected to a net."*
- **`PWR-242` — 469 occurrences:** *"Skipping pulse because of zero timing for instance…"*

`PWR-268` is a 44k-occurrence TWF integrity problem that directly reinforces the report's own
"the TWF looks malformed" conclusion (§2.3 of that report) — and an expert still missed it
while reading the log by hand. **This is the argument for the whole program in one data
point:** mechanical exhaustiveness is exactly what humans lose at scale, and what tooling
provides for free.

### 2.5 Reports are parseable; some artifacts are dangerously large

`*.rpt` files have a stable, self-describing header (generating command, tool version,
design, user, host, date) followed by fixed-column data — cheap and reliable to parse.

But: `vector_annotated_nets.txt` is **772 MB** and the `DB/` directory is **~84 GB**. The
collector must have hard size guards and stream/sample rather than read. The reference
report explicitly notes "The 84 GB DB was not read" — correct behavior to preserve.

### 2.6 Resource telemetry is already emitted

`report_time` / `report_memory` produce parseable checkpoints:

```
Info: Current wall time from the start of mpower: 10h42m5s (CPU time: 6h18m56s.711ms)
Info Memory usage: CURRENT - 56.922GB, PEAK 56.930GB.
```

Stage attribution and the CPU/wall parallel-efficiency signal (6.3h CPU vs 7.8h wall →
"the solver is not parallelizing well") fall straight out of these. Caveat: checkpoints only
exist where the *user's* script called them, so stage attribution is best-effort and the
system must degrade gracefully.

---

## 3. Product concept

**Name (working):** mPower Advisor

**Primary form factor (Phase 1–2):** a standalone command-line tool, run after (or during)
an mPower run, pointed at a run directory:

```
mpower-advisor <run_dir>  →  mpower_result_summary.{html,md,json}
```

**Assumption stated explicitly** (user was unavailable to confirm): standalone CLI first.
Rationale — no dependency on the mPower C++ build/release cycle, works on *historical* run
directories immediately (large existing corpus of runs to validate against), and can be
adopted by AEs without a tool update. Tcl-command and GUI integration are deliberately
deferred to Phase 5 once the analysis content has proven itself.

**Design principles**

1. **Deterministic first, LLM second.** Every number in the report comes from a parser, not
   a model. The LLM explains, correlates, prioritizes, and writes prose — it never invents a
   metric. This is non-negotiable for sign-off credibility.
2. **Cite everything.** Each finding links back to the source file and line. An engineer must
   be able to audit any claim in one click.
3. **Degrade gracefully.** A crashed run, a partial run, a missing `.rpt`, a run with no
   `report_time` calls — all must still produce a useful report.
4. **Never read the elephant.** Size guards on every input.
5. **Offline-capable.** Must be usable in customer environments with no external network and
   strict data-egress rules (see §8 Risks).

---

## 4. Architecture (conceptual)

Five layers. Each is independently useful and independently testable.

```
  run directory (logs, *.rpt, *.tcl, mpower.cmd)
        │
  ┌─────▼──────────────────────────────────────────────┐
  │ L1  COLLECTORS — artifact discovery + normalization │
  │     → "Run Manifest" (single JSON, the contract)    │
  └─────┬──────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────┐
  │ L2  KNOWLEDGE BASE                                  │
  │     mhelp catalog · ghelp commands · regression     │
  │     flows · curated remediation playbooks           │
  └─────┬──────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────┐
  │ L3  ANALYZERS — deterministic rules → Findings      │
  │     completeness · triage · perf · margin · config  │
  └─────┬──────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────┐
  │ L4  REASONING — LLM correlation, ranking, narrative │
  │     + agentic tool-use (Phase 3+)                   │
  └─────┬──────────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────────────────┐
  │ L5  PRESENTATION — HTML · Markdown · JSON · chat    │
  └────────────────────────────────────────────────────┘
```

### L1 — Collectors
Discover and normalize: `mpower.log`, `mpower.error.log`, `mpower.warning.log`,
`mpower.lib.log`, `mpower.cmd`, `lef_def.warnings`, `run.tcl`, and `*.rpt`
(static / dynamic / em / resistance / power / current families). Handles the
`logs_<timestamp>/` + symlink convention, multi-run directories, and MPI/distributed runs
with per-rank logs.

**Output: the Run Manifest** — a single versioned JSON document that everything downstream
consumes. This is the most important architectural decision in the plan: it decouples
parsing from analysis, makes the whole system testable without LLM calls, and lets the
report, the chat agent, and any future GUI panel share one source of truth.

### L2 — Knowledge base
- **Message catalog:** normalize the 1,767 mhelp entries (fix the `<type>` variants), and
  progressively enrich `<action>` with real remediation guidance.
- **Command reference:** the 390 `.help` files → what each command/option does, so the
  advisor can recommend *specific* next commands (e.g. `report_nets -type without_delay`).
- **Flow patterns:** mine `regression/short/**` for canonical, known-good flow orderings —
  the basis for "your flow is missing X" and "X before Y" checks.
- **Playbooks:** curated, human-authored remediation for the top ~100 codes by real-world
  frequency. This is where domain experts (Power_Simulation, Extraction, EM_Thermal per
  `maintainers.yml`) inject knowledge that exists nowhere in the codebase.

### L3 — Analyzers
Deterministic rule modules producing structured **Findings**
(`severity, code, evidence, count, section, remediation`):

- **Completeness/missing-input** — un-annotated nets, missing SPEF/TWF sections, nets
  without arrival time, unreachable nets, missing PG pins, missing timescale.
- **Error triage** — cluster errors, map to catalog, identify "command ignored" silent
  fallbacks (the 4 duplicate-clock errors are the exemplar).
- **Warning roll-up** — apply the validated **max-of-roll-up** rule from §2.4 (never line
  counts, never sums). Dedicated regression test with the 13 known-good reference values.
- **Performance attribution** — stage wall/CPU/memory, % of total, CPU:wall parallel
  efficiency, slot utilization vs available, distributed-enabled-or-not.
- **Margin checks** — worst static/dynamic IR vs configurable spec (e.g. ≤10% of rail),
  EM violations, grid resistance outliers.
- **Config sanity** — "0 decaps in design", "distributed power disabled", "16 of 128 slots",
  option combinations that contradict the analysis type.

### L4 — Reasoning
The Findings + Run Manifest go to the model. Its job:
- **Correlate across domains** — the reference report's best insight is that the ignored
  duplicate-clock errors land on `u_rts_clk_rst_ctrl`, *the same block as the worst dynamic
  IR*. That cross-domain link is exactly what deterministic rules miss and an LLM finds.
- **Rank** by impact-to-the-user, not by raw count.
- **Write** the executive summary and the prioritized "Recommended Next Actions".

Phase 3 adds **tool use**: the agent can call back into the manifest, grep specific logs,
look up catalog entries, or (guarded, opt-in) run read-only mPower query commands on a
saved DB.

### L5 — Presentation
HTML (the reference artifact is the visual spec — dark theme, KPI cards, severity callouts,
numbered action list), Markdown for pasting into tickets, JSON for CI/regression gating.

---

## 5. Phased roadmap & level of effort

LOE in **dev engineer-weeks (EW)** unless stated otherwise, assuming engineers familiar with
mPower flows. A "team" of 1–2 devs + fractional domain-expert time is assumed throughout.
Domain-expert effort is quoted separately as **expert-weeks** and is *not* added to the dev
total — the two come from different staffing pools.

> **Basis of estimate.** These are top-down figures calibrated against the artifacts
> inspected in §2, not a bottom-up task decomposition. Three conventions matter when reading
> them:
>
> 1. **EW is dev effort, not elapsed time.** Phase 0 is largely gated on collecting run
>    directories from colleagues — calendar time that is not charged here.
> 2. **Phases 1–3 assume LLM-assisted development.** Parsers, report renderers and prompt
>    scaffolding are exactly the work that compresses hardest; the same scope hand-written
>    would be roughly 2× these numbers.
> 3. **Dev-weeks and expert-weeks are not interchangeable.** Knowledge enrichment and the
>    Phase 2 blind-review evaluation are domain-expert time.
>
> Phases 4 and 5 are the least compressible and carry the widest uncertainty — Phase 4 needs
> genuine PI domain modeling, and Phase 5's Tcl/GUI2 integration is bound to the product
> build and regression cycle. Together they are **~65–70% of the dev total.** If a
> bottom-up re-estimate is needed for funding, do it on those two and not on Phases 0–3.

### Phase 0 — Foundations & corpus *(2 EW)*
Assemble a labeled corpus of ~20–30 diverse historical run directories (pass/fail/crash,
static/dynamic, vectored/vectorless, EM, single/distributed, analog/digital). Define the
Run Manifest schema. Normalize the mhelp catalog. Establish the evaluation harness.

> *Do not skip this.* Without a corpus, every later phase is built on one testcase and will
> overfit to it.

### Phase 1 — Deterministic Run Digest *(2–3 EW)*
L1 collectors + L3 analyzers + L5 HTML/MD/JSON. **No LLM.** Reproduces the factual content
of the reference report — KPIs, stage timing, roll-up warning counts, missing-input table,
IR/power summary — for any run directory.

**Exit criterion:** on the BRCM reference run, every number in the generated report matches
the hand-authored `mpower_result_summary.html` (including the 13 validated roll-up counts
from §2.4), and it runs cleanly on ≥20 corpus runs.
**Value delivered:** already shippable to AEs on its own — and, as §2.4 shows, already
catching findings that expert manual analysis misses.

### Phase 2 — LLM analyst layer *(2–3 EW)*
L4 narrative, cross-domain correlation, severity ranking, prioritized actions. Strict
grounding (no invented numbers) + citation enforcement. Model-agnostic interface so the
backend can be swapped for whatever is approved for customer data.

**Exit criterion:** blind review by 3 domain experts rates the generated analysis
"as good as or better than what I'd write" on ≥70% of corpus runs.

### Phase 3 — Interactive agent *(2–3 EW)*
Conversational follow-up ("why is this instance failing?", "show me the nets without
delay"), tool-use over the manifest and logs, session memory, multi-run comparison
("what changed vs last week's run?"). Multi-run diff is a heavily requested capability and
is nearly free once the Manifest exists.

### Phase 4 — EM/IR root-cause deep dive *(10–14 EW)*
The hardest and highest-value phase. Goes beyond "you have a violation" to "here's the
cause and the fix": hotspot spatial clustering, hotspot ↔ power-density ↔ grid-resistance
↔ via-density correlation, decap adequacy, blame attribution to block/cell/layer, and
concrete mitigation proposals. Requires reading larger result artifacts (streaming/sampling)
and genuine PI domain modeling. **Requires dedicated domain-expert co-development.**

### Phase 5 — Setup assistant & product integration *(6–10 EW)*
Pre-flight flow validation (catch setup errors *before* an 11-hour run — enormous value),
guided setup Q&A, and integration into the product surface: a Tcl command
(`analyze_run`), a GUI2 dashboard panel, and/or CI regression gating.

### Cross-cutting — Knowledge enrichment *(4–6 expert-weeks, runs parallel from Phase 0)*
Fill the 976 empty `<action>` fields for the top codes by real-world frequency; author the
top-100 playbooks. **Staff this with domain experts, not the core dev team.** It is the
long pole on *quality* and it is independent of all engineering work.

### Summary

| Phase | Deliverable | LOE (dev EW) | Cumulative |
|---|---|---:|---:|
| 0 | Corpus, manifest schema, eval harness | 2 | 2 |
| 1 | Deterministic digest + HTML report | 2–3 | 4–5 |
| 2 | LLM analyst narrative | 2–3 | 6–8 |
| 3 | Interactive agent, multi-run diff | 2–3 | 8–11 |
| 4 | EM/IR root-cause | 10–14 | 18–25 |
| 5 | Setup assistant + integration | 6–10 | 24–35 |
| — | Knowledge enrichment *(parallel)* | 4–6 **expert**-weeks | *not in dev total* |

**First shippable value: ~4–5 dev engineer-weeks** (Phases 0+1).
**Full vision: ~24–35 dev engineer-weeks**, i.e. roughly 2 engineers × 3–4 months, plus
**4–6 domain-expert weeks** of knowledge enrichment running in parallel.

---

## 6. Recommended sequencing

1. **Start Phase 0 and the knowledge enrichment simultaneously.** They have different
   staffing (dev vs domain expert) and enrichment is the long pole on quality.
2. **Ship Phase 1 standalone to a small AE pilot group.** Get real usage before adding LLM
   reasoning; the feedback will reshape the finding taxonomy.
3. **Only then add Phase 2.** The LLM layer is cheap to build *once the Manifest exists* and
   expensive to build without it.
4. **Decide Phase 4 vs Phase 5 by pilot feedback**, not upfront. If pilots say "the report
   is great but I wish it caught this before I burned 11 hours", Phase 5 pre-flight jumps
   the queue ahead of Phase 4.

---

## 7. Success metrics

| Metric | Target |
|---|---|
| Time-to-first-insight after a run | hours → < 5 minutes |
| Factual accuracy of reported numbers | 100% (deterministic; regression-tested) |
| Expert agreement with generated analysis | ≥ 70% "as good as mine" |
| Real issues surfaced that the user missed | ≥ 1 per run, median |
| Coverage of message codes with remediation | 44% → ≥ 90% for top-100 by frequency |
| Wasted long runs prevented (Phase 5) | track as $ / compute-hours saved |

---

## 8. Risks & mitigations

| Risk | Sev | Mitigation |
|---|---|---|
| **Customer data confidentiality.** Logs contain design names, hierarchy, cell libraries, IP names. Sending them to an external LLM may be contractually prohibited. | **High** | Treat as a hard architectural constraint from day 1: model-agnostic backend, support for on-prem/approved-endpoint inference, and a no-LLM mode (Phase 1 alone). Add optional scrubbing/anonymization of instance and library names. **Resolve the approved-endpoint question before Phase 2 starts.** |
| **LLM hallucinating numbers** destroys sign-off trust permanently. | **High** | Deterministic-first architecture; the model receives pre-computed findings and is never asked to compute. Automated post-generation check that every numeral in the narrative exists in the Manifest. |
| **Log format drift** across mPower releases breaks parsers. | Med | Version-aware parsers keyed on the banner version; parser regression tests in the mPower regression suite so drift is caught by the tool team, not the field. |
| **Overfitting to the BRCM testcase.** | Med | Phase 0 corpus is mandatory and diverse by construction. |
| **Domain-expert bandwidth** for playbooks and evaluation. | Med | Scope to top-100 codes by measured frequency; recruit named owners per `src/glue/maintainers.yml` team boundaries. |
| **Huge artifacts** (772 MB net lists, 84 GB DB) cause OOM/hangs. | Med | Hard size guards, streaming, sampling; never read the DB in Phases 1–3. |
| **Scope creep into a general EDA chatbot.** | Med | Every phase gated on a concrete, measurable user outcome. |
| **Adoption** — engineers ignore another report. | Med | Pilot early (end of Phase 1); optimize for the "Recommended Next Actions" list, which is the part users actually read. |

---

## 9. Open decisions (need owner input)

1. **Delivery form factor** — standalone CLI assumed for Phase 1. Confirm, or prioritize the
   Tcl command / GUI panel earlier.
2. **LLM endpoint & data policy** — which inference backend is approved for customer design
   data? *Blocks Phase 2.* Highest-priority open question.
3. **Target user** — internal AEs first, or external customers? Changes the bar for polish,
   packaging, and support.
4. **IR/EM spec thresholds** — where do per-design sign-off budgets come from (config file,
   user prompt, inferred)?
5. **Ownership** — which team owns this? It spans all six teams in `maintainers.yml`.
6. **Relationship to existing `fuse/` AI assistant integration** — extend it or build
   parallel? Should be checked before Phase 3.

---

## 10. Immediate next steps (first 2 weeks)

1. Confirm the open decisions in §9 — especially the LLM data-policy question.
2. Collect 20–30 historical run directories into a corpus; catalog by flow type and outcome.
3. Draft and review the Run Manifest schema. *(This is the key design artifact — get it
   reviewed widely before building on it.)*
4. Normalize the mhelp `<type>` field and produce a frequency-ranked list of message codes
   from the corpus, to target the `<action>` enrichment effort.
5. Identify per-team domain-expert owners for playbook authoring and evaluation.
