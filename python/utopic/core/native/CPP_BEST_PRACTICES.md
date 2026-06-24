# C++ Best Practices — Utopic/native

This document defines the C++ conventions for the native runtime
(`main.cpp`, `diffusion_driver.{h,cpp}`, `tool_extract.h`). The north star is
**llama.cpp's** house style: simple, dependency-light, readable, fast. Where our
code diverges, this document says so and shows the fix.

The conventions below are extracted directly from the local llama.cpp checkout:

- `/Users/adavya/llama.cpp/CONTRIBUTING.md` — coding + naming guidelines
- `/Users/adavya/llama.cpp/AGENTS.md` — comment style, ASCII-only rule
- `/Users/adavya/llama.cpp/.clang-format` — the authoritative formatter config
- `/Users/adavya/llama.cpp/.editorconfig` — whitespace/encoding
- `/Users/adavya/llama.cpp/include/llama-cpp.h` — the canonical RAII wrappers
- `/Users/adavya/llama.cpp/include/llama.h` — public C API style
- `/Users/adavya/llama.cpp/SECURITY.md` — untrusted-input posture

When in doubt, run `clang-format` against llama.cpp's `.clang-format` and copy
the surrounding code's pattern. Our `.clang-format` should be a copy of theirs.

---

## 1. Overview & philosophy

llama.cpp's guidelines (CONTRIBUTING.md, lines 74-101) boil down to:

> Avoid adding third-party dependencies, extra files, extra headers, etc.
> Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid
> templates, keep it simple. Vertical alignment makes things more readable and
> easier to batch edit.

Our runtime already embodies the most important consequence of this: **the
driver and CLI link against the public `llama.h` only** — no `common/`, no
`httplib`, no server. `main.cpp` line 1 states this explicitly, and
`diffusion_driver.cpp` lines 3-9 re-implement a 3-line `LOG_*` shim rather than
pull in `common/log.h`. Keep it that way. Every new dependency is a portability
and security liability across our target backends (Metal, CUDA/GB10, AMD).

Principles, in priority order:

1. **Simple over clever.** A plain `for` loop beats a `std::ranges` pipeline.
   The reviewer (and the next backend porter) must understand it instantly.
2. **Dependency-light.** Public `llama.h` / `ggml.h` only. No new third-party
   headers without a written justification.
3. **Readable via vertical alignment.** Align declarations, assignments, and
   struct initializers in columns (this is why `.clang-format` sets
   `AlignConsecutiveAssignments` / `AlignConsecutiveDeclarations`).
4. **Fast, but measured.** This is single-request latency code. Don't allocate
   in the per-step hot loop; don't add abstractions the compiler can't see
   through. Optimizations land with measured evidence, not a hunch.
5. **ASCII only in code and comments.** No em-dash, no `->` arrow glyph, no
   `x`/`...` unicode (AGENTS.md lines 62). Use `-`, `->`, `x`, `...`.
   *(Our existing comments use the em-dash and unicode arrows liberally,
   e.g. `tool_extract.h:1`, `main.cpp:5`, `diffusion_driver.cpp:3`. New code
   must use ASCII; the cleanup should convert existing ones.)*

---

## 2. Formatting & naming

### Whitespace (`.editorconfig`, `.clang-format`)

- **4 spaces** for indentation, never tabs (`UseTab: Never`, `IndentWidth: 4`).
- **LF** line endings, **UTF-8**, final newline, **no trailing whitespace**.
- **Column limit 120** (`ColumnLimit: 120`).
- Braces **on the same line** (`BreakBeforeBraces: Attach`): `if (x) {`.
- One statement per line in non-trivial code. *(Our hot paths cram many
  statements per line, e.g. `main.cpp:234`, `:240`, `:419`, `:431`. These were
  written for density during research; production code should expand them — see
  Section 7.)*

### Pointers and references (CONTRIBUTING.md line 80; `PointerAlignment: Middle`)

Space on **both sides** of `*` and `&`:

```cpp
void * ptr;          // OK
int  & a;            // OK
llama_context * ctx; // OK
void* ptr;           // not OK
```

Our headers already do this correctly — see the function-pointer typedef in
`diffusion_driver.h:21` (`void * user_data`) and the API signatures
`diffusion_driver.h:151-156`.

### Vertical alignment (CONTRIBUTING.md line 79)

This is the single most visible llama.cpp trait. Align in columns:

```cpp
// diffusion_driver.h:28 — GOOD, matches llama.cpp exactly
struct diffusion_params {
    int32_t                   steps                   = 0;
    float                     temperature             = 0;
    llama_token               mask_token_id           = LLAMA_TOKEN_NULL;
    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
};
```

Function parameters are aligned one-per-line when they wrap
(`BinPackParameters: false`):

```cpp
// diffusion_driver.cpp:111 — GOOD
void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated);
```

### Naming (CONTRIBUTING.md lines 103-160)

- **`snake_case`** for functions, variables, and types.
- **Enum values UPPER_CASE, prefixed with the enum name:**

  ```cpp
  // diffusion_driver.h:7 — GOOD
  enum diffusion_algorithm {
      DIFFUSION_ALGORITHM_ORIGIN           = 0,
      DIFFUSION_ALGORITHM_CONFIDENCE_BASED = 4,
  };
  ```

- **`<class>_<method>` with `<method>` = `<action>_<noun>`** for public
  functions. Our public API follows this: `diffusion_generate`,
  `diffusion_generate_entropy_bound`. Internal statics follow it loosely
  (`calculate_confidence`, `get_num_transfer_tokens`, `add_gumbel_noise`).
- **Optimize names for longest common prefix.** Prefer `n_tokens_max` over
  `max_n_tokens` so related names sort/group together.
- **`struct foo {}`, never `typedef struct foo {} foo`.** In C++ omit the
  optional `struct`/`enum` keyword when not needed
  (`llama_context * ctx`, not `struct llama_context * ctx`).
- **Sized integer types in any API surface**: `int32_t`, `int64_t`, `size_t` for
  sizes/offsets. The driver header is consistent here. *(`main.cpp` mixes bare
  `int`/`long` for `canvas_length`, `n_input`, etc. — acceptable for a local CLI
  `main`, but anything that crosses a header boundary uses sized types.)*

### Naming deviation in `tool_extract.h`

`tool_extract.h` uses **PascalCase** types (`ToolCall`) and a `toolx`
namespace. This is the clearest deviation from llama.cpp in our tree
(llama.cpp would name it `tool_call` with no namespace, free functions prefixed
`tool_`). It is self-contained and harmless, but for consistency a cleanup
should rename to `struct tool_call` and `tool_extract` / `tool_to_openai_json`
free functions, dropping the namespace (llama.cpp avoids namespaces in this
kind of utility header).

---

## 3. Resource management & RAII

Every allocation must have **one clear owner** and a deterministic free. This is
the area where our `.cpp` files most need tightening.

### The llama.cpp resource types and their frees

| Resource | Acquire | Release | RAII wrapper (llama-cpp.h) |
|---|---|---|---|
| model | `llama_model_load_from_file` | `llama_model_free` | `llama_model_ptr` |
| context | `llama_init_from_model` | `llama_free` | `llama_context_ptr` |
| sampler | `llama_sampler_init_*` | `llama_sampler_free` | `llama_sampler_ptr` |
| LoRA adapter | `llama_adapter_lora_init` | `llama_adapter_lora_free` | `llama_adapter_lora_ptr` |
| batch | `llama_batch_init` | `llama_batch_free` | *(none — wrap it yourself)* |

llama.cpp ships ready-made RAII wrappers in
`/Users/adavya/llama.cpp/include/llama-cpp.h`:

```cpp
struct llama_context_deleter {
    void operator()(llama_context * context) { llama_free(context); }
};
typedef std::unique_ptr<llama_context, llama_context_deleter> llama_context_ptr;
```

**Rule: include `llama-cpp.h` and use these wrappers for model/context/sampler.**
There is no wrapper for `llama_batch`; write a small local one.

### Where our code is correct

The driver's *normal* exit path frees everything
(`diffusion_driver.cpp:790-792`, `:1055`), and `main.cpp:571-573` frees
ctx -> model -> backend in order on the success path. Good.

### The leaks/foot-guns to fix

1. **Early returns leak.** `main.cpp` has many `return 1;` / `return 0;`
   paths after `llama_model_load_from_file` (`:301`), after
   `llama_init_from_model` (`:372`), and inside every self-test block
   (`:400`, `:438`, `:460`) that do **not** free the model/context/batch. The
   self-tests `return 0;` mid-function leaking model + ctx + batches. With RAII
   wrappers these all become safe automatically.

   ```cpp
   // BEFORE (main.cpp:300-301, :371-372) — leaks on every early return below
   llama_model * model = llama_model_load_from_file(model_path, mp);
   if (!model) { fprintf(stderr, "...failed to load %s\n", model_path); return 1; }
   ...
   llama_context * ctx = llama_init_from_model(model, cp);
   if (!ctx) { fprintf(stderr, "...ctx init failed\n"); return 1; }

   // AFTER — owner is the unique_ptr; every return frees in reverse order
   #include "llama-cpp.h"
   llama_model_ptr model(llama_model_load_from_file(model_path, mp));
   if (!model) { fprintf(stderr, "...failed to load %s\n", model_path); return 1; }
   ...
   llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
   if (!ctx) { fprintf(stderr, "...ctx init failed\n"); return 1; }
   // pass model.get() / ctx.get() to the C API; no manual frees, no leaks
   ```

2. **`llama_batch` needs a scope guard.** The driver frees batches by hand at
   *each* error branch (`diffusion_driver.cpp:248-251`, `:885-886`) and on the
   normal path. That is error-prone: add a branch, forget the free, leak. Wrap
   it:

   ```cpp
   struct llama_batch_guard {
       llama_batch b;
       explicit llama_batch_guard(llama_batch b) : b(b) {}
       ~llama_batch_guard() { llama_batch_free(b); }
       llama_batch_guard(const llama_batch_guard &)             = delete;
       llama_batch_guard & operator=(const llama_batch_guard &) = delete;
   };
   ```

   Then `llama_batch_guard batch{ llama_batch_init(params.max_length, 0, 1) };`
   and use `batch.b`. Every `return` now frees it. Same for the samplers
   (`llama_sampler_ptr`).

3. **One owner, no raw owning pointers.** Never store a heap pointer you must
   remember to free. Use `std::vector` (we already do, e.g. `main.cpp:330`,
   `diffusion_driver.cpp` token buffers) or a `unique_ptr` deleter. Raw
   pointers in our APIs (`const llama_token * input_tokens`,
   `diffusion_driver.h:152`) are **non-owning views** — that is fine and
   idiomatic; the contract is "caller owns, callee borrows."

---

## 4. Error handling

llama.cpp's rules, applied to us:

- **No C++ exceptions across the C API boundary, none in hot paths.** The
  public `diffusion_*` functions take `int32_t & n_generated` and signal
  "nothing produced" by setting it to 0 and returning early
  (`diffusion_driver.cpp:117-120`) — this is the right pattern. Keep it.
- **Validate inputs at the top and return early**, exactly as
  `diffusion_driver.cpp:118`:

  ```cpp
  n_generated = 0;
  if (!ctx || !input_tokens || !output_tokens || n_input <= 0 ||
      params.max_length <= n_input) {
      return;
  }
  ```

- **Check every `llama_decode` return.** We do
  (`main.cpp:202`, `:235`, `:417`; `diffusion_driver.cpp:248`), logging to
  `stderr` and bailing. Keep checking it everywhere — a failed decode that is
  ignored reads stale/garbage logits.
- **`GGML_ASSERT` for programmer-error invariants, not for runtime/user input.**
  llama.cpp uses `GGML_ASSERT(cond)` / `GGML_ABORT("msg")`
  (`src/llama.cpp:46`, `:107`) for "this can only happen if the code is wrong."
  Untrusted/runtime conditions (a bad prompt, a missing flag, a decode failure)
  must be a graceful return + `stderr` message, never an assert/abort.
- **Distinguish exit codes.** `main` returns non-zero on failure
  (`main.cpp:294`, `:301`, `:372`). Keep that; callers rely on it.

---

## 5. Security & safety

Per llama.cpp's `SECURITY.md`, **prompts and model output are untrusted input**
and **env vars are untrusted configuration**. Our runtime ingests all three, so:

### 5.1 Bounds safety on every buffer

The tokenize/detokenize dance is the classic two-call pattern: call with a
buffer, and if it returns negative, the magnitude is the required size — resize
and retry. We do this correctly:

```cpp
// main.cpp:45-46, :331-332 — GOOD, the canonical safe pattern
int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, false);
if (n < 0) { t.resize(-n); n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, false); }
```

Detokenize bounds: `main.cpp:530-538` guards `t == eos`, `t < 0`,
`t >= n_vocab`, `t == mask` **before** indexing — this is correct and must be
preserved on every path that consumes `out[]`. Note the SSD path
(`main.cpp:455`) duplicates this guard; keep them in sync or factor a helper.

**Always pass the buffer size to the C API** (`llama_token_to_piece(..., piece,
sizeof(piece), ...)`, `main.cpp:536`). Never assume a fixed piece fits — check
the negative-return-means-too-small contract there too (`main.cpp:159` does;
`:536` truncates silently with a 256-byte buffer, which is acceptable for a
piece but should be commented as a deliberate cap).

### 5.2 Integer overflow on size computations

Size math that feeds an allocation or an index must not overflow. Multiply in a
wide type and cast once. Our code already casts to `size_t` before multiplying
in the bitmask math:

```cpp
// main.cpp:155, :170 — GOOD, the multiply is done in size_t
std::vector<uint64_t> allow((size_t) 3 * n_words, 0);
allow[(size_t) 0 * n_words + (v >> 6)] |= (1ull << (v & 63));
```

and in the SSD batch index `(size_t) nseq * L` (`main.cpp:201`, `:208`). Apply
the same `(size_t)` discipline anywhere `int * int` indexes or sizes a buffer,
especially with attacker-influenced `n_predict` / `--steps` / canvas sizes.
`llama_batch_init(nseq * L, ...)` (`main.cpp:185`) multiplies two `int`s — for
large self-test fan-outs this should be range-checked or computed in `size_t`.

### 5.3 Untrusted env and CLI parsing

`main.cpp` reads ~30 `DG_*` env vars via `getenv` + `atoi`/`atof`
(`:341`-`:511`). `atoi`/`atof` are tolerant (no error on garbage, UB on
overflow for `atoi`). For a local research CLI this is acceptable, but:

- **Bound every numeric knob** that sizes a buffer or a loop. A hostile/typo'd
  `DG_SLOT_LEN=999999999` or `DG_SSD_N` flows straight into `n_ctx`
  (`main.cpp:364`) and batch allocation. Clamp to sane maxima after parsing.
- The `arg()` flag scanner (`main.cpp:36-39`) reads `argv[i+1]`; it correctly
  loops `i < argc - 1` so it never reads past the end. Keep that guard.
- Treat **model metadata as untrusted** too: `canvas_length` comes from GGUF
  (`main.cpp:307-308`) via `strtol` into a `long` — validate it is positive and
  within a ceiling before using it to size the context.

### 5.4 Avoiding undefined behavior

- **No signed integer overflow**; do size math in `size_t`/`int64_t` (5.2).
- **No out-of-bounds index**; guard before subscript (5.1).
- **Initialize before use** — prefer `T x{};` and in-struct default member
  initializers (the driver structs do this, `diffusion_driver.h:28+`).
- **`std::regex` is a DoS surface.** `tool_extract.h` runs regexes over model
  output (`:45-47`). Catastrophic backtracking on adversarial output can hang
  the process. The current patterns are simple (`\w+`, bounded char classes) so
  they are safe, but never add a regex with nested quantifiers over untrusted
  text; prefer a hand-written scan (as `match_brace`, `tool_extract.h:32`,
  already does for brace balancing).
- **`match_brace` is truncation-safe** (returns `s.size()` on unterminated
  input, `tool_extract.h:38`) — that defensiveness is the standard to match.

### 5.5 Safe string/buffer handling

- Use `std::string` / `std::vector` for owning buffers (we do). Never `strcpy`
  / `sprintf` into fixed arrays; use `snprintf` with `sizeof`, or `std::string`.
- `json_escape` (`tool_extract.h:67`) correctly escapes `"`, `\\`, `\n` before
  emitting JSON — output must always be escaped, since it is built from model
  text. Extend it if you ever emit other control chars.

---

## 6. API design (C public header, C++ impl)

llama.cpp's model: a **C-compatible public header**, a **C++ implementation**,
no C++ types or exceptions across the boundary.

- **`diffusion_driver.h` is the public surface.** Today it `#include`s
  `llama.h` and uses `enum`, `struct`, function-pointer typedefs, and plain
  `int32_t`/`float`/pointer fields — all C-compatible **except** the
  `const std::string &` would be (it has none; good). Keep STL out of this
  header's function signatures: pass `const llama_token *` + length, not
  `std::vector`. We already do (`diffusion_driver.h:151`).
- **Opaque handles, sized types, prefixed enums** as in `llama.h`
  (`LLAMA_API`, `int32_t llama_token`, `LLAMA_VOCAB_TYPE_*`). Our enums match
  this (`DIFFUSION_TRANSFER_SCHEDULE_*`, `diffusion_driver.h:16`).
- **No RTTI, no exceptions across the API.** Don't `throw` out of
  `diffusion_generate`; return-and-set-output-param (Section 4).
- **Callbacks are C function pointers**, not `std::function`
  (`diffusion_step_callback_t`, `diffusion_driver.h:21`) — correct; a
  `std::function` would drag the STL into the ABI and cost an allocation.
- **Document non-obvious header fields with a one-line comment.** The
  `diffusion_params` field comments are exemplary in *content* but several are
  multi-paragraph essays (`diffusion_driver.h:55-78`). llama.cpp comments are
  terse (AGENTS.md lines 80-132: "keep comments concise; avoid redundant or
  excessive inline commentary"). Trim header comments to the invariant + a
  pointer to where the rationale lives.

---

## 7. Project-specific rules (with before/after from our files)

### 7.1 ASCII-only comments (AGENTS.md line 62)

```cpp
// BEFORE — tool_extract.h:1, main.cpp:5, diffusion_driver.cpp:3 (em-dash, unicode arrow)
// tool_extract.h — tolerant tool-call extractor ...
// ... key->value, hidden -> ~126k vocab ...

// AFTER — ASCII only
// tool_extract.h - tolerant tool-call extractor ...
// ... key->value, hidden -> ~126k vocab ...
```

### 7.2 One statement per line in production paths

The research hot-loops pack many statements onto one line for density. That is
fine in a throwaway self-test, but production code must be one statement per
line so it is debuggable and reviewable:

```cpp
// BEFORE — main.cpp:240 (six statements, one line)
int am=0; float mx=lg[0]; for (int v=1;v<n_vocab;v++) if(lg[v]>mx){mx=lg[v];am=v;}
double Z=0; for (int v=0;v<n_vocab;v++) Z+=exp((double)(lg[v]-mx));

// AFTER — argmax helper, expanded, reused everywhere we scan logits
static int32_t argmax(const float * lg, int32_t n_vocab) {
    int32_t am = 0;
    float   mx = lg[0];
    for (int32_t v = 1; v < n_vocab; v++) {
        if (lg[v] > mx) { mx = lg[v]; am = v; }
    }
    return am;
}
```

The same `argmax` scan is hand-inlined in at least five places
(`main.cpp:207`, `:240`, `:419`, `:431`; SSD path) — factor it once. This is
both a readability and a correctness win (one place to fix, no copy drift).

### 7.3 RAII the model/context/batch/sampler (Section 3)

The single highest-value cleanup. Replace the manual free chains in
`main.cpp` and the per-branch `llama_batch_free` / `llama_sampler_free` in
`diffusion_driver.cpp` (`:248-251`, `:790-792`, `:885-886`, `:1055`) with
`llama_model_ptr` / `llama_context_ptr` / `llama_sampler_ptr` (from
`llama-cpp.h`) and a `llama_batch_guard`. This removes every leak on the
self-test early-return paths and makes new error branches safe by construction.

### 7.4 Bound untrusted numeric knobs

```cpp
// BEFORE — main.cpp:344, :361 (unbounded; feeds n_ctx / batch alloc)
const int slot_len = getenv("DG_SLOT_LEN") ? atoi(getenv("DG_SLOT_LEN")) : 8;
const int ssd_n    = getenv("DG_SSD_N")    ? atoi(getenv("DG_SSD_N"))    : 4;

// AFTER — clamp to sane maxima before they size memory
static int env_int(const char * name, int def, int lo, int hi) {
    const char * v = getenv(name);
    int x = v ? atoi(v) : def;
    return x < lo ? lo : (x > hi ? hi : x);
}
const int slot_len = env_int("DG_SLOT_LEN", 8, 1, 256);
const int ssd_n    = env_int("DG_SSD_N",    4, 1, 64);
```

### 7.5 Keep the public surface dependency-free

`diffusion_driver.cpp` re-defines `LOG_*` as a 3-line `fprintf` shim
(`:3-9`) specifically to avoid `common/log.h`. **Do not "improve" this by
pulling in `common/`.** The whole point of the native runtime is to link
against public `libllama` only, so it ports cleanly to the CUDA/AMD backends.

### 7.6 Tool-extractor naming (Section 2)

Rename `toolx::ToolCall` -> `tool_call`, drop the namespace, and prefix the free
functions `tool_*` to match llama.cpp's no-namespace, snake_case utility-header
style. Low priority, but it is the most visible naming outlier in the tree.

---

## Summary — key conventions and the top deviations to fix

**Key conventions (match llama.cpp):**
- 4-space indent, LF, UTF-8, 120 cols, attach braces, `void * p` / `int & a`,
  vertical column alignment, ASCII only.
- `snake_case` everywhere; `ENUM_PREFIX_VALUE` for enum members;
  `struct foo {}` not `typedef`; sized int types (`int32_t`/`size_t`) on any
  header boundary; names optimized for common prefix.
- C-compatible public header (`diffusion_driver.h`), C++ impl, C function-pointer
  callbacks, no exceptions/RTTI across the boundary; errors via early-return +
  output param, `GGML_ASSERT` only for programmer-error invariants.
- Every resource has one owner: use `llama_model_ptr` / `llama_context_ptr` /
  `llama_sampler_ptr` (from `llama-cpp.h`) and a `llama_batch` guard.
- Treat prompts, model output, env vars, and GGUF metadata as untrusted: bounds-
  check before indexing, do size math in `size_t`, clamp numeric knobs, escape
  all emitted strings, avoid backtracking regex on model output.
- Dependency-light: public `llama.h` / `ggml.h` only, no `common/`.

**Top deviations in our code the cleanup should fix (in priority order):**
1. **Resource leaks on early returns** — `main.cpp`'s self-test blocks
   (`:400`, `:438`, `:460`) and error paths (`:301`, `:372`) `return` without
   freeing model/ctx/batch. Adopt the `llama-cpp.h` RAII wrappers + a
   `llama_batch_guard`; also factor the per-branch frees in
   `diffusion_driver.cpp` (`:248-251`, `:885-886`).
2. **Unbounded untrusted numeric knobs** — `DG_SLOT_LEN` / `DG_SSD_N` / canvas
   sizes flow unchecked into `n_ctx` and batch allocation (`main.cpp:344`,
   `:361`, `:364`); add a clamped `env_int` helper and validate GGUF
   `canvas_length`.
3. **Non-ASCII comments** — em-dashes / unicode arrows throughout
   (`tool_extract.h:1`, `main.cpp:5`, `diffusion_driver.cpp:3`); convert to
   ASCII per AGENTS.md.
4. **Density one-liners + duplicated `argmax`** — multi-statement lines in the
   hot paths (`main.cpp:234`, `:240`, `:419`, `:431`); expand to one statement
   per line and factor the repeated argmax/confidence scan into one helper.
5. **`tool_extract.h` naming** — PascalCase `ToolCall` + `toolx` namespace
   diverge from llama.cpp; rename to `tool_call` + snake_case free functions.
6. **Over-long header comments** — `diffusion_params` field essays
   (`diffusion_driver.h:55-78`) exceed llama.cpp's "concise comments" rule;
   trim to invariant + rationale pointer.
7. **Integer-multiply sizing** — `llama_batch_init(nseq * L, ...)`
   (`main.cpp:185`) and similar `int * int` index/size math should be done in
   `size_t` and/or range-checked.
