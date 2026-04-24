[![Actions Status](https://github.com/m-doughty/LLM-Classifiers-Emotions/actions/workflows/test.yml/badge.svg)](https://github.com/m-doughty/LLM-Classifiers-Emotions/actions)

NAME
====

LLM::Classifiers::Emotions - 28-way text emotion classifier using Cohee's quantized DistilBERT go-emotions model

SYNOPSIS
========

```raku
use LLM::Classifiers::Emotions;

my $clf = LLM::Classifiers::Emotions::Classifier.new;

say $clf.top("I'm so angry I could scream");
    # 'anger'

say $clf.top("I'm really curious how this works");
    # 'curiosity'

for $clf.classify("I love you more than anything")[^3] -> %r {
    say "{ %r<label> }: { %r<score>.fmt('%.3f') }";
}
    # love: 0.263
    # caring: 0.175
    # admiration: 0.116
```

STATUS
======

v0.1 — single-text inference only. Batch / multi-label modes are deferred to later releases. See LIMITATIONS below.

Output is fully deterministic under the CPU execution provider — bit-identical logits across runs for the same input on the same ONNX Runtime version. Tests assert this invariant.

INSTALLATION
============

```shell
zef install LLM::Classifiers::Emotions
```

At install time, `Build.rakumod` downloads three files (tokenizer.json, config.json, and a 68 MB quantized ONNX model) from the module's GitHub Release, verifies SHA256 against bundled checksums, and stages them under `$XDG_DATA_HOME/ LLM-Classifiers-Emotions/`. If the GitHub Release path fails (or if you set `LLM_EMOTIONS_FROM_HF=1`), it falls back to downloading directly from Cohee's HuggingFace repo at a pinned revision via `HuggingFace::API`.

No system-wide changes are made. `zef uninstall` leaves the staged model alone — remove it manually if you want the ~70 MB back.

Environment variables
---------------------

<table class="pod-table">
<tbody>
<tr> <td>LLM_EMOTIONS_BINARY_URL</td> <td>Override GitHub Release base URL</td> </tr> <tr> <td>LLM_EMOTIONS_BINARY_ONLY=1</td> <td>Refuse HF fallback; fail if the primary path doesn&#39;t complete</td> </tr> <tr> <td>LLM_EMOTIONS_FROM_HF=1</td> <td>Skip primary, go straight to HuggingFace (dev / debug)</td> </tr> <tr> <td>LLM_EMOTIONS_CACHE_DIR</td> <td>Override download cache dir (default $XDG_CACHE_HOME)</td> </tr> <tr> <td>LLM_EMOTIONS_DATA_DIR</td> <td>Override staged base dir (default $XDG_DATA_HOME)</td> </tr> <tr> <td>LLM_EMOTIONS_MODEL_DIR</td> <td>(runtime) load model from this dir instead of the staged dir</td> </tr>
</tbody>
</table>

API
===

LLM::Classifiers::Emotions::Classifier
--------------------------------------

### new(:@providers = [CPU], :$model-dir = IO::Path, :$log-id = 'llm-emotions')

Load the tokenizer, model, and label map. `@providers` is passed through to `ONNX::Native::Session`; defaults to CPU-only so output is reproducible. `$model-dir` overrides the XDG-staged lookup — useful for dev iteration or pointing at an alternate go-emotions variant.

Throws `X::LLM::Classifiers::Emotions::ModelMissing` if the staged dir isn't found and no override is supplied. Throws `X::LLM::Classifiers::Emotions::InvalidConfig` if `config.json` doesn't parse.

### classify(Str:D $text --> List[Hash])

Classify one text. Returns a list of 28 Hash entries, sorted descending by score, each with `label => Str ` and `score => Num `. Scores sum to 1.0 ± 1e-5 (softmax invariant).

Input longer than 512 tokens is truncated — DistilBERT's positional embeddings are fixed at that length, and Cohee's tokenizer doesn't truncate on its own.

### top(Str:D $text, Numeric :$min-score --> Str)

Argmax label — the most-likely single emotion. With `:$min-score`, returns `Str` (the type object / undefined) if the top score falls below the threshold.

### labels(--> List[Str])

The 28 go-emotions labels in id order: `admiration amusement anger annoyance approval caring confusion curiosity desire disappointment disapproval disgust embarrassment excitement fear gratitude grief joy love nervousness optimism pride realization relief remorse sadness surprise neutral`.

### raw-logits(Str:D $text --> List[Num])

Pre-softmax 28-element logit list, straight from the ONNX session. Exposed for tests that want to assert bit-identical output as a determinism check. Normal callers should use `classify`.

### dispose() / DESTROY

Release the underlying ONNX session. Idempotent.

MODEL
=====

  * **Source:** `Cohee/distilbert-base-uncased-go-emotions-onnx` on HuggingFace ([https://huggingface.co/Cohee/distilbert-base-uncased-go-emotions-onnx](https://huggingface.co/Cohee/distilbert-base-uncased-go-emotions-onnx)). The same model SillyTavern uses for its "expressions" feature.

  * **Architecture:** DistilBERT-base-uncased (6 transformer layers, 66 M parameters), fine-tuned for sequence classification on the GoEmotions dataset.

  * **Quantization:** int8 dynamic quantization. ~68 MB on disk vs. ~268 MB for the full-precision variant. Trivial accuracy difference for this task.

  * **Training data:** [GoEmotions](https://aclanthology.org/2020.acl-main.372/) (Demszky et al, 2020) — ~58k English Reddit comments labelled for 27 emotion categories + "neutral".

  * **Revision pin:** `d22488bc83be87678f12eee8a3f65a65de94ef85`. The `BINARY_TAG` file at the dist root and `resources/REVISION` together pin the exact model weights shipped by this release. Bump both together when updating.

DETERMINISM
===========

Under the default CPU provider, ONNX Runtime uses MLAS (Microsoft's matrix math library) for the inference hot path. MLAS is fully deterministic across runs for the same input + same ORT version: logits are bit-identical, softmax preserves that, and argmax is stable.

**CoreML divergence note:** if you pass `:providers<coreml cpu>` on macOS, CoreML may produce slightly different scores (at the ~1e-3 level) because some ops downcast to fp16 internally on the Neural Engine. The *argmax* label still agrees with CPU for go-emotions across all inputs we've tested. Tests pin `:cpu` so they pass regardless of provider.

Upgrading ONNX Runtime (the `ONNX::Native` BINARY_TAG) may shift logits at the LSB level if MLAS gets a new kernel. Re-compute test expectations after such an upgrade.

PLATFORM NOTES
==============

macOS
-----

Staged under `~/.local/share/LLM-Classifiers-Emotions/`. First load of the model is ~200ms on Apple Silicon; subsequent classify calls are ~30ms for typical chat-length input.

Linux
-----

Same staging path. Performance is comparable on contemporary x86_64 (MLAS's AVX2 kernels kick in), slightly slower on aarch64.

Windows
-------

Not supported in v0.1 — `ONNX::Native` itself doesn't ship Windows binaries yet. Once that lands, this module should work without change (the protocols and file formats are platform-neutral).

LIMITATIONS
===========

  * Single-text inference only — no batch mode. Classifying thousands of strings takes thousands of session runs. Batching is a worthwhile ~2x speedup and is queued for v0.2.

  * Single-label output only. The underlying GoEmotions dataset permits multi-label annotations; this module's argmax discards that information.

  * Input beyond 512 tokens is silently truncated. Callers that care about long-document sentiment should summarise or chunk upstream.

  * English-only. GoEmotions is entirely English Reddit; the model has no non-English training signal and scores garbage on non-English input.

  * Training-set biases — Reddit 2020 comments skew US-centric, young, male, and more casual than written text generally. Performance on formal text or non-Reddit dialects is degraded.

AUTHOR
======

  * Matt Doughty

MODEL CREDIT
============

  * Cohee — ONNX / quantized conversion of the original model

  * Demszky et al (Google Research, 2020) — GoEmotions dataset

  * HuggingFace — original DistilBERT base model

COPYRIGHT AND LICENSE
=====================

This Raku module is Copyright 2026 Matt Doughty, distributed under the Artistic License 2.0.

The underlying model weights ship separately (pulled at install time) and are subject to their own licenses: MIT per Cohee's repo, inheriting from google/distilbert-base-uncased (Apache-2.0) and the GoEmotions corpus (CC BY 4.0).

