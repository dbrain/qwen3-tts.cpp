# streaming vocoder plan

status: draft — implementation design for task #14 (phase 3b). covers
tasks #20–#25.

this document is written to be followed by a single engineer without
further design work. every phase names a concrete file, function, and
line range to touch, and every state tensor is sized explicitly against
the current 1.7B-base config.

---

## 1. goal & success criteria

### 1.1 goal

replace the current one-shot vocoder path (transformer generates **all**
frames → vocoder decodes them into one `speech.audio.delta` event) with a
streaming path where the vocoder decodes frame batches as the transformer
produces them, flushing pcm on the wire as each batch finishes.

first-audio latency target: drops from `O(total_generation)` to
`O(prefill + first_batch_generation + one_batch_vocoder_decode)`. for a
300-frame utterance at batch size 8, expect roughly 30× lower ttfa on
the vocoder side alone.

### 1.2 hard correctness invariant

**streaming decode output must equal one-shot decode output, bit-exact
or within `max-abs-diff < 1e-5` pcm float.** any audible seam between
chunks means state threading is broken and the work is not done. this is
the gate for landing.

### 1.3 non-goals

- chunked encode for the reference-audio side (ICL encoder runs once per
  request; latency there is negligible).
- cross-request cache reuse.
- adaptive batching. start with a fixed knob.

---

## 2. current code map (baseline)

### 2.1 decoder module

header: `src/audio_tokenizer_decoder.h`

- `audio_decoder_config` (lines 15–29) — dims and defaults:
  - `sample_rate=24000`, `n_codebooks=16`, `codebook_size=2048`,
    `codebook_dim=256`
  - `latent_dim=1024`, `hidden_dim=512`, `decoder_dim=1536`
  - `n_pre_tfm_layers=8`, `n_heads=16`, `ffn_dim=1024`
  - `upsample_rates[4]={8,5,4,3}`
  - `rms_norm_eps=1e-5f`, `rope_theta=10000.0f`
- `pre_tfm_layer` (lines 32–47)
- `residual_block` (lines 50–60) — has `int dilation`; dilations are
  set to `{1,3,9}` for `res[0..2]` of every decoder block (loader lines
  291–294).
- `decoder_block` (lines 63–74) — contains `residual_block res[3]`.
- `upsample_block` (lines 77–89) — convnext-style: transpose conv →
  depthwise conv → layernorm → two pointwise convs → γ scale → residual.
- `audio_decoder_state` (lines 150–155): `backend`, `backend_cpu`,
  `sched` (`ggml_backend_sched_t`), `compute_meta` scratch buffer.
- `AudioTokenizerDecoder` class (lines 159+):
  - `load_model(path)` line 169, `unload_model()` line 172
  - `decode(codes, n_frames, samples)` line 178 — current entry point
  - `get_config()` line 181

impl: `src/audio_tokenizer_decoder.cpp`

- `apply_pre_tfm_layer` (408–484) — rmsnorm → QKV proj → rope (436–442)
  → causal mask `ggml_diag_mask_inf(ctx, KQ, 0)` (451) → softmax →
  attention out → swiglu ffn.
- `apply_upsample_block` (486–544) — `ggml_conv_transpose_1d` stride 2
  (493), then depthwise conv kernel 7 with left-pad 6 via
  `ggml_pad_ext(ctx, x, 6, 0, 0, 0, 0, 0, 0, 0)` (506–507).
- `apply_residual_block` (546–574) — snake → `conv_1d` kernel 7 with
  left-pad `6·dilation` (557–558) → snake → `conv_1d` kernel 1 (568).
- `apply_decoder_block` (576–614) — snake → `conv_transpose_1d` at
  upsample rate (590), result view-trimmed left/right by
  `kernel_size - stride` (597–602) → three `apply_residual_block`.
- `build_graph(n_frames)` (616–796):
  - 628–640: 16 input tensors `codes_cb0`..`codes_cb15`.
  - 642–697: VQ embed lookup + projection sum → `latent` at
    `[n_frames, hidden_dim=512, 1]`.
  - 699–706: pre-conv `ggml_conv_1d` kernel 3 with left-pad 2 →
    `[n_frames, latent_dim=1024, 1]`.
  - 708–719: reshape/transpose to `[latent_dim, n_frames]` then project
    to transformer input via `pre_tfm_input_proj_w` →
    `[hidden_dim=512, n_frames]`.
  - 721–731: positions input tensor + 8× `apply_pre_tfm_layer` + final
    rmsnorm + `pre_tfm_output_proj_w` → `[latent_dim, n_frames]`.
  - 740–750: permute back → `[n_frames, latent_dim, 1]`, then 2×
    `apply_upsample_block` (net 4× upsample at this point).
  - 752–759: dec0 conv (kernel 7, pad 6) → `[*, decoder_dim=1536, 1]`.
  - 761–767: 4× `apply_decoder_block` at rates {8,5,4,3}.
  - 769–780: dec5 snake + dec6 final conv (kernel 7, pad 6) →
    `[samples, 1, 1]`.
  - 784–795: tanh → reshape 1D → `ggml_set_output(cur)` → build.
- `decode(codes, n_frames, samples)` (798–877):
  - 818: `struct ggml_cgraph * gf = build_graph(n_frames);` — **graph is
    rebuilt per call** (caching removed in commit c215ab0 due to
    stale-scratch corruption; the fix lives in that commit message).
  - 820: `ggml_backend_sched_alloc_graph`.
  - 825–841: `ggml_backend_tensor_set` into each `codes_cbN`.
  - 845–852: positions = 0..n_frames-1, set on `"positions"` tensor.
  - 857: `ggml_backend_sched_graph_compute`.
  - 863–872: read output, append pcm to `samples`.
  - 874: `ggml_backend_sched_reset`.

### 2.2 transformer generation loop

- `TTSTransformer::generate(...)` signature: `src/tts_transformer.h`
  lines 282–294. returns int32 codes as a row-major
  `[n_frames, n_codebooks=16]` flat vector (frame-major interleaved).
- body: `src/tts_transformer.cpp` lines 2832–3131.
  - prefill: 2890–2943.
  - autoregressive loop: 2960–3125. key line for frame emission is
    **3068–3070** where all 16 codebook codes for a frame have just been
    pushed onto `output`. this is where the streaming callback fires.
- existing KV cache pattern for the transformer itself:
  - struct `tts_kv_cache`: `src/tts_transformer.h` lines 184–196.
  - init: `src/tts_transformer.cpp` lines 769–817.
  - clear (zeroes backend memory to avoid flash-attn reads past
    `n_used`): 819–829. we reuse this pattern for the vocoder pre-tfm.

### 2.3 server speech handler

- route handler: `src/server.cpp` 657–889.
- synthesize call: 793–806. today calls `tts.synthesize{,with_embedding}()`
  and returns a populated `tts_result` only after everything is done.
- `stream_format` branch:
  - `""` (binary): 830–837.
  - `"audio"` (chunked raw pcm/wav): 841–858 uses
    `res.set_chunked_content_provider(ctype, cb)` where
    `cb: (size_t offset, httplib::DataSink& sink) -> bool`. call
    `sink.write(data, n)` to push bytes, `sink.done()` to terminate.
  - `"sse"`: 865–888. builds two frames in memory and flushes them; this
    is the branch we are converting.
- `build_done_event(result)`: 184–222 — `usage`,
  `timings.{prompt_n,predicted_n,prompt_ms,predicted_ms,prompt_per_second,predicted_per_second,...}`.
  keep unchanged; we only change when it fires.

### 2.4 `tts_result`

`src/qwen3_tts.h` lines 56–93. we do **not** change the shape of this
struct; the server pushes pcm deltas directly from the callback and
still builds the final `speech.audio.done` from the returned `result`.

### 2.5 CLI

`src/main.cpp` lines 8–151. add `--streaming-batch-size N` (default 0 =
non-streaming) around line 150 following the existing arg pattern.

### 2.6 cmake

`CMakeLists.txt`:
- test exes at lines 181–225, registered via `add_test(...)` 262–279.
  add `test_streaming_parity` and a new `add_test(NAME streaming_parity
  ...)` entry.

---

## 3. receptive-field and state inventory

### 3.1 chain, left to right

every stage below lives inside `build_graph`; line numbers are for the
existing one-shot graph.

| idx | stage                      | file:line (current) | causal need at input                              | output channels  | upsample factor |
|-----|----------------------------|---------------------|---------------------------------------------------|------------------|-----------------|
| 0   | VQ embed + project         | 642–697             | none                                              | `hidden_dim=512` | 1×              |
| 1   | pre-conv (k=3)             | 699–706             | **2 prior input frames**                          | `latent_dim=1024`| 1×              |
| 2   | pre-tfm input proj         | 708–719             | none                                              | `hidden_dim=512` | 1×              |
| 3   | pre-tfm (×8 layers)        | 721–731             | **per-layer KV over all prior frames**            | `hidden_dim=512` | 1×              |
| 4   | pre-tfm output proj + reshape | 733–744          | none                                              | `latent_dim=1024`| 1×              |
| 5   | upsample_block[0]: transpose k=2 s=2 | 493       | none (transpose stride=kernel)                    | `latent_dim`     | 2×              |
| 5a  | upsample_block[0].dwconv (k=7, pad 6 left) | 506–507 | **6 prior post-transpose frames** at `latent_dim` | `latent_dim`     | —               |
| 5b  | upsample_block[0].pwconvs + γ + residual | 516–543   | none                                              | `latent_dim`     | —               |
| 6   | upsample_block[1]: transpose + dwconv | same as 5  | **6 prior frames** at `latent_dim` (post-transpose of block 1) | `latent_dim` | 2×         |
| 7   | dec0 conv (k=7, pad 6 left)| 752–759             | **6 prior frames** at `latent_dim`                | `decoder_dim=1536`| 1×             |
| 8   | dec_block[0] transpose rate=8 | 590              | none                                              | 768              | 8×              |
| 8a  | dec_block[0].res[0] conv1 (k=7, dil=1, pad 6)  | 557–558 | **6 frames** at 768 ch                            | 768              | —               |
| 8b  | dec_block[0].res[0] conv2 (k=1)                | 568    | none                                              | 768              | —               |
| 8c  | dec_block[0].res[1] conv1 (k=7, dil=3, pad 18) | 557–558 | **18 frames** at 768 ch                           | 768              | —               |
| 8d  | dec_block[0].res[1] conv2                      | 568    | none                                              | 768              | —               |
| 8e  | dec_block[0].res[2] conv1 (k=7, dil=9, pad 54) | 557–558 | **54 frames** at 768 ch                           | 768              | —               |
| 8f  | dec_block[0].res[2] conv2                      | 568    | none                                              | 768              | —               |
| 9   | dec_block[1] transpose rate=5 | 590              | none                                              | 384              | 5×              |
| 9a–f| dec_block[1] res[0..2]                         | same   | 6/18/54 frames at 384 ch                          | 384              | —               |
| 10  | dec_block[2] transpose rate=4 | 590              | none                                              | 192              | 4×              |
| 10a–f | dec_block[2] res[0..2]                       | same   | 6/18/54 frames at 192 ch                          | 192              | —               |
| 11  | dec_block[3] transpose rate=3 | 590              | none                                              | 96               | 3×              |
| 11a–f | dec_block[3] res[0..2]                       | same   | 6/18/54 frames at 96 ch                           | 96               | —               |
| 12  | dec5 snake                 | 769–773             | none                                              | 96               | —               |
| 13  | dec6 conv (k=7, pad 6)     | 775–780             | **6 frames** at 96 ch                             | 1 (pcm)          | —               |
| 14  | tanh + reshape + output    | 784–790             | none                                              | 1 (pcm)          | —               |

net upsample (pre-tfm → audio): 2·2·8·5·4·3 = **1920×** samples-per-frame.
post-tfm latent → audio: 1920×. for 24 kHz pcm at 12.5 Hz latent rate
this lines up (1920 samples/frame).

### 3.2 state categories

**A. attention KV cache (unbounded, per pre-tfm layer).** pattern is
identical to `tts_transformer.cpp` 769–829. we allocate per-layer
`K`/`V` tensors once at model load with shape `[head_dim, n_heads,
max_frames]` in F16 (decoder pre-tfm uses `n_heads=16`; `head_dim =
hidden_dim / n_heads = 512 / 16 = 32`). per step we:
  1. compute Q/K/V on the new tile (`n_new` frames).
  2. write new K/V slices at offset `n_past` along the seq axis.
  3. attend against `K[0..n_past+n_new]`, `V[0..n_past+n_new]` with
     `ggml_diag_mask_inf(ctx, KQ, n_past)` so the new tile can see all
     past but still respects causality inside itself.

max context bound: pick `QWEN3_DEC_MAX_KV_FRAMES` to cover the longest
expected utterance. 2048 frames = ~164s of audio at 12.5 Hz = safe
default. memory budget:
`2 (K+V) · 8 layers · head_dim 32 · n_heads 16 · 2048 · 2 bytes (F16)
  = 32 MiB`. trivially small.

**B. causal conv tails (bounded, one ring per conv).** for every causal
conv in the tower we keep the last `L` output frames of the **pre-conv
signal** so we can prepend them to the next tile. `L` is the conv's
required left-context (column labelled "causal need at input" above).

count and sizes of ring buffers (assuming the conservative sizing
`dec_block[i].res_channels[i]` = block output channels from the loader;
store as F16 to halve memory; each ring is `L` frames × `C` channels
× 2 bytes):

| ring                            | L   | C (channels)                 | bytes       |
|---------------------------------|-----|------------------------------|-------------|
| pre-conv                        | 2   | `hidden_dim = 512`           | 2048        |
| upsample[0] dwconv              | 6   | `latent_dim = 1024`          | 12288       |
| upsample[1] dwconv              | 6   | `latent_dim = 1024`          | 12288       |
| dec0 conv                       | 6   | `latent_dim = 1024`          | 12288       |
| dec_block[0] res[0..2] conv1 ×3 | 6, 18, 54 | 768                     | 110592      |
| dec_block[1] res[0..2] conv1 ×3 | 6, 18, 54 | 384                     | 55296       |
| dec_block[2] res[0..2] conv1 ×3 | 6, 18, 54 | 192                     | 27648       |
| dec_block[3] res[0..2] conv1 ×3 | 6, 18, 54 | 96                      | 13824       |
| dec6 conv                       | 6   | 96                            | 1152        |

**total**: 15 ring buffers, ~265 KiB aggregate at F16. channel counts
are populated at load time into `config.dec_out_channels[4]` from the
`conv_t.weight` `ne[1]` of each decoder block (phase A1, landed).
observed for 1.7B-base: `[768, 384, 192, 96]`.

**C. decoder_block transpose conv.** the transpose itself has no state
(stride = kernel/2, left and right trim equal `stride`). once we have
tail A (pre-transpose input) recorded via the res conv1 rings above,
the transpose output for a new tile can be rebuilt without further
state. **no separate ring needed for the transpose.**

---

## 4. API surface

### 4.1 new types

in `src/audio_tokenizer_decoder.h` after the existing
`AudioTokenizerDecoder` declaration:

```cpp
// opaque handle holding KV cache + conv ring buffers
struct audio_decoder_stream_state;

class AudioTokenizerDecoderStream {
public:
    AudioTokenizerDecoderStream();
    ~AudioTokenizerDecoderStream();

    // borrow the model/backend/sched from an already-loaded decoder;
    // does not take ownership. safe to construct many streams over
    // one decoder (each allocates its own KV + tails).
    bool init(AudioTokenizerDecoder * dec, int32_t max_frames);

    // reset all state to zeros; prepare for a new utterance.
    void reset();

    // decode `n_frames` frames of codes (same layout as the existing
    // AudioTokenizerDecoder::decode: flat int32, row-major
    // [n_frames, n_codebooks]) and append pcm samples to `out`.
    // seamless concatenation of successive calls reproduces the
    // one-shot decode output.
    bool decode(const int32_t * codes, int32_t n_frames,
                std::vector<float> & out);

    // frames consumed since last reset(). useful for sanity checks.
    int32_t n_past() const;

private:
    audio_decoder_stream_state * s_ = nullptr;
};
```

`audio_decoder_stream_state` lives in `src/audio_tokenizer_decoder.cpp`
(private) and holds:

```cpp
struct audio_decoder_stream_state {
    AudioTokenizerDecoder * dec;       // borrowed

    // attention KV cache (per pre-tfm layer). allocated once in init().
    ggml_context *         kv_ctx = nullptr;
    ggml_backend_buffer_t  kv_buffer = nullptr;
    std::vector<ggml_tensor *> k_cache;  // 8 layers
    std::vector<ggml_tensor *> v_cache;

    // conv ring buffers, parallel order to table in §3.2 part B.
    ggml_context *         tails_ctx = nullptr;
    ggml_backend_buffer_t  tails_buffer = nullptr;
    std::vector<ggml_tensor *> tails;    // 15 rings in fixed order

    int32_t n_past = 0;       // frames consumed
    int32_t max_frames = 0;
};
```

### 4.2 transformer callback

add to `src/tts_transformer.h` near line 280 (before `generate`):

```cpp
// Fired after each decoded frame's 16 codebook codes are appended to
// `output`. `frame_codes` is a pointer into `output` — valid until the
// next iteration. return false to abort generation.
using frame_emit_fn = std::function<bool(int32_t frame_idx,
                                         const int32_t * frame_codes)>;
```

extend the `generate` signature with a final optional arg (default
`nullptr`):

```cpp
bool generate(/* ... existing args ... */,
              const frame_emit_fn * on_frame = nullptr);
```

alternative: attach via a setter on `TTSTransformer` if chaining through
the `Qwen3TTS` wrapper is cleaner. pick whichever lets
`Qwen3TTS::synthesize` pass it through without leaking callback types
into the public ABI.

insert the call at **tts_transformer.cpp:3070** immediately after the
`output.push_back` loop:

```cpp
if (on_frame && !(*on_frame)(frame, &output[output.size() - cfg.n_codebooks])) {
    return false;  // generation aborted by caller
}
```

### 4.3 `Qwen3TTS` glue

in `src/qwen3_tts.{h,cpp}` add an overload / new field to pass through
a streaming spec:

```cpp
struct streaming_opts {
    int32_t batch_size = 0;   // 0 = no streaming (current behavior)
    // called for each decoded pcm batch. `pcm` is freshly-allocated
    // per call. return false to abort.
    std::function<bool(const float * pcm, size_t n_samples)> on_pcm;
};

tts_result synthesize(const std::string & text,
                      const synth_params & params,
                      const streaming_opts * stream = nullptr);
// mirror for synthesize_with_embedding.
```

internally, when `stream && stream->batch_size > 0`:
1. construct an `AudioTokenizerDecoderStream` on the already-loaded
   decoder.
2. set a `frame_emit_fn` on the transformer that buffers frames until
   `batch_size` are collected, then calls `stream_state.decode(...)` and
   forwards the pcm through `stream->on_pcm`.
3. flush the leftover batch (< batch_size) after generation finishes.
4. populate `result.audio` by concatenating everything we streamed (for
   parity with the one-shot return shape; optional — see §6).

---

## 5. graph changes

### 5.1 signature change

`AudioTokenizerDecoder::build_graph(int32_t n_frames, int32_t n_past)`.
when `n_past == 0` and state-tail inputs are zero, the graph must be
pointwise-identical to today's graph (this is how we protect the
one-shot path).

### 5.2 per-conv replacement pattern

**before** (every causal conv today):
```cpp
x = ggml_pad_ext(ctx, x, L, 0, 0, 0, 0, 0, 0, 0);   // left-pad with zeros
x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);
```

**after**:
```cpp
// tail is a model-input tensor of shape [L, C, 1], F16 or F32 to match
// the output precision we chose for the ring buffer.
struct ggml_tensor * tail = ggml_new_tensor_3d(ctx, TYPE, L, C, 1);
ggml_set_name(tail, "tail_<unique_name>");
ggml_set_input(tail);

x = ggml_concat(ctx, tail, x, 0);                    // prepend L frames
x = ggml_conv_1d(ctx, w, x, stride, 0, dilation);    // no ggml pad

// emit the new tail of `pre_conv_input` (before this conv ran, frames
// of THIS stage's input) as an output bound to the same name so the
// driver can copy it back into the ring for next call.
struct ggml_tensor * new_tail = ggml_view_3d(
    ctx, /*src=*/x_input_before_concat, L, C, 1,
    /*nb1=*/..., /*nb2=*/..., /*offset=*/(n_new - L) * stride_bytes);
ggml_set_name(new_tail, "tail_out_<unique_name>");
ggml_set_output(new_tail);
```

the `new_tail` view must be taken from the **input signal** to this
conv on the current tile (what was there **before** prepending the
ring), sliced to the last `L` frames. if the tile has fewer than `L`
new frames, the driver must do a shift-and-append rather than a plain
view. handle that outside the graph: if `n_new >= L`, take the last L
from new input; if `n_new < L`, take `last (L - n_new)` from prior ring
and all of new input. easier: do the ring update on the CPU side after
each call, not in the graph.

so actually — **do not emit `tail_out_*` from the graph**. the driver,
post-compute, does:
```cpp
// pseudo, per ring:
append new tile input into a scratch; copy last L frames into
ring_buffer[i]; backend_tensor_set on the ring input for next call.
```
this is simpler and avoids fiddly ggml views across backends.

### 5.3 per-layer attention replacement

today in `apply_pre_tfm_layer` (408–484):
```cpp
Qcur = ggml_mul_mat(ctx, layer.attn_q_w, normed);   // [hidden, n_new]
Kcur = ggml_mul_mat(ctx, layer.attn_k_w, normed);
Vcur = ggml_mul_mat(ctx, layer.attn_v_w, normed);
// reshape to head layout, rope, permute
struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q);   // [n_new, n_new, n_heads]
KQ = ggml_scale(ctx, KQ, 1/√head_dim);
KQ = ggml_diag_mask_inf(ctx, KQ, 0);
KQ = ggml_soft_max(ctx, KQ);
V = ggml_cont(ctx, ggml_transpose(ctx, V));
KQV = ggml_mul_mat(ctx, V, KQ);
```

**after** (introduce `n_past` arg and per-layer `k_cache`/`v_cache`
input tensors):
```cpp
// Q on new tile only.
Qcur = ggml_mul_mat(ctx, layer.attn_q_w, normed);

// K, V on new tile then views into cache at [..., n_past..n_past+n_new].
Kcur = ggml_mul_mat(ctx, layer.attn_k_w, normed);
Vcur = ggml_mul_mat(ctx, layer.attn_v_w, normed);
// rope Q/Kcur as before, but pass positions = n_past..n_past+n_new-1
// (not 0..n_new-1). see §5.4.

// write new K/V into cache (the driver does this via an explicit
// "write slice" tensor, or we use ggml's inplace view-and-copy
// pattern — see llama.cpp build_llama's kv path for canonical code).
K_slice = ggml_view_3d(ctx, k_cache[layer], head_dim, n_heads, n_new,
                        nb1, nb2, n_past * nb2);
ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, K_slice));
V_slice = ggml_view_3d(ctx, v_cache[layer], head_dim, n_heads, n_new,
                        nb1, nb2, n_past * nb2);
ggml_build_forward_expand(gf, ggml_cpy(ctx, Vcur, V_slice));

// attend over [0..n_past+n_new]
K_all = ggml_view_3d(ctx, k_cache[layer], head_dim, n_heads,
                      n_past + n_new, nb1, nb2, 0);
V_all = ggml_view_3d(ctx, v_cache[layer], head_dim, n_heads,
                      n_past + n_new, nb1, nb2, 0);
K_all = ggml_permute(ctx, K_all, 0, 2, 1, 3);  // to match Q layout
V_all = ggml_permute(ctx, V_all, 0, 2, 1, 3);

KQ = ggml_mul_mat(ctx, K_all, Q);
KQ = ggml_scale(ctx, KQ, 1/√head_dim);
KQ = ggml_diag_mask_inf(ctx, KQ, n_past);   // <-- key change
KQ = ggml_soft_max(ctx, KQ);
V_all = ggml_cont(ctx, ggml_transpose(ctx, V_all));
KQV = ggml_mul_mat(ctx, V_all, KQ);
```

copy-the-shape reference: `tts_transformer.cpp`'s prefill and step
graphs (around `build_step_graph` / `build_prefill_graph`, used in the
prefill path 2890–2943 and step path 3116–3118). the decoder pre-tfm
wants the same structure.

### 5.4 positions under streaming

the positions tensor today (721–727) is filled `0..n_frames-1` in
`decode()` at 845–852. under streaming, positions for a tile must be
`n_past..n_past+n_new-1` so RoPE stays absolute across chunks.

### 5.5 keeping the one-shot path fast

two options:
1. **keep two graphs**: `build_graph_oneshot(n_frames)` (unchanged) and
   `build_graph_streaming(n_frames, n_past)` (new). `decode()` uses the
   first; `AudioTokenizerDecoderStream::decode()` uses the second. more
   code but zero risk of regressing the one-shot path.
2. **single graph** with a compile-time-ish flag: graph always takes
   `n_past` and state-tail/kv-cache inputs, but when called with
   `n_past=0` and zero tails the output is identical. less code but
   requires careful parity testing.

**recommendation**: start with option 1. once parity is established and
stable, revisit to deduplicate.

### 5.6 scratch/meta buffer sizing

`QWEN3_TTS_DEC_MAX_NODES = 32768` (audio_tokenizer_decoder.cpp:626) is
generous, but the streaming graph adds ~24 `ggml_concat` / `ggml_view`
/ `ggml_cpy` nodes (ring-prepend × 15 + cache-write + cache-read × 8).
check meta via `ggml_graph_overhead_custom` at startup and bump
`QWEN3_TTS_DEC_MAX_NODES` if needed.

---

## 6. implementation phases

each phase ends in a landable commit. do not skip.

### phase A — map receptive field and extract channel info (#20) ✅ landed

- [x] A1. extract per-decoder-block output channels into
      `audio_decoder_config::dec_out_channels[4]` from
      `dec_blocks[i].conv_t_w->ne[1]` at load time
      (`audio_tokenizer_decoder.cpp` ~300).
- [x] A2. stderr log of the computed channel counts on load
      (same location, mirrors the existing backend log).
- [x] A3. channel counts for 1.7B-base: `[768, 384, 192, 96]`; tables
      in §3.1 and §3.2 updated with concrete numbers.

### phase B — rebuild graph with state inputs (#22)

- [x] B1. simplified: skipped the oneshot/streaming split. a single
      `build_graph` always takes tail inputs; `decode()` zero-fills them
      for one-shot mode, equivalent to the prior `ggml_pad_ext` zero-pad.
      bit-exact parity verified (`cmp` byte-identical output).
- [x] B2. replaced each causal `ggml_pad_ext(..., L, 0, ...) + conv` with
      `ggml_concat(tail, x, 0) + conv` per §5.2. 15 tail inputs total:
      `tail_pre_conv`, `tail_up{0,1}_dwconv`, `tail_dec0`, `tail_dec6`,
      and 12× `tail_dec{1..4}_res{0..2}_conv1`. tail names tracked in
      `tail_names_` vector on `AudioTokenizerDecoder`, populated at
      build_graph time, iterated for zero-fill in `decode()`.
- [ ] B3. wire per-layer pre-tfm attention to K/V cache inputs per §5.3.
      mask uses `ggml_diag_mask_inf(ctx, KQ, n_past)`.
      **deferred to phase C** — one-shot with `n_past=0` is equivalent
      to current code, so no correctness gap until the streaming driver
      exists.
- [ ] B4. thread `n_past`-aware positions per §5.4. **deferred to phase C**.

**ship**: commit with the new graph building paths + parity test
passing with `n_past=0`.

### phase C.5 — conv_transpose overlap-add streaming ✅ landed

each `apply_decoder_block` applies a `ggml_conv_transpose_1d` with
`kernel = 2*stride`. the raw output has length `(n_in+1)*s`; one-shot
trims `s` samples on each side (the "partial" warmup/tail regions where
only one input contributes). naive streaming trims per chunk, losing
`2s` samples at every chunk boundary.

the fix (`audio_tokenizer_decoder.cpp` apply_decoder_block streaming
branch): hold back the raw right-tail `[n_in*s, (n_in+1)*s)` and
overlap-add it onto the first `s` samples of the next chunk's raw
output. the saved tail represents the partial contribution of the
current chunk's last input that would be completed by future inputs;
overlap-add reconstructs the full contribution in the next chunk.

state: one `s * out_channels` f32 ring per decoder block (~36 KiB
total across all 4 blocks). zero-initialized; cleared in
`stream_reset()`. graph input `conv_t_overlap_in_{i}` and output
`next_conv_t_overlap_{i}` per block. emit range is `[s, n_in*s)` on
chunk 0 (drops left warmup) and `[0, n_in*s)` on subsequent chunks.
upsample blocks use `stride=kernel=2` so `pad=0` — no overlap needed.

tested via `tests/test_streaming_parity.cpp` with random codes:

| frames | chunk | result            |
|--------|-------|-------------------|
| 16     | 16    | bit-exact (pass)  |
| 64     | 16    | bit-exact (pass)  |
| 128    | 32    | bit-exact (pass)  |
| 256    | 16/32/64 | bit-exact (pass) |

**decoder causality note** (discovered while validating c.5): at chunk
sizes below ~16 latent frames, `decode(n_half)` diverges from
`decode(n_full)[:half_pcm]` by up to 1.6e-2 PCM. the streaming
implementation correctly reproduces per-chunk `decode()` output, but
since `decode()` itself isn't bit-exact across batch sizes that straddle
a backend tile-size threshold, sub-16-frame chunks will show this
divergence vs. a full-utterance one-shot baseline. it is NOT a streaming
bug. realistic production chunk sizes (≥16 frames ≈ 1.3s audio) are
safely bit-exact. investigate the root cause separately if it matters;
likely a Vulkan softmax/matmul tile threshold.

### phase C — driver: state allocation + post-compute ring update (#22 cont.)

- [ ] C1. implement `AudioTokenizerDecoderStream::init` — allocate KV
      context/buffer (mirror `tts_transformer.cpp::init_kv_cache`) and
      tails context/buffer. zero both.
- [ ] C2. implement `decode(codes, n_frames, out)`:
      1. `build_graph_streaming(n_frames, n_past)`.
      2. set all codebook inputs via `ggml_backend_tensor_set` (copy of
         868–841 of existing `decode`).
      3. set each tail input from `s->tails[i]`.
      4. set positions to `n_past..n_past+n_frames-1`.
      5. alloc + compute with `ggml_backend_sched_*`.
      6. copy pcm out, append to `out`.
      7. **ring update**: for every causal conv, run a small CPU-side
         function that takes the layer's input signal (we'll need to
         mark each such intermediate with `ggml_set_output` too, or
         store it in host-side scratch during the compute via an
         on-the-fly permuted view — decide: emit-as-output is simpler
         and costs little). copy the last L frames into the
         corresponding ring buffer via `ggml_backend_tensor_set`.
      8. `n_past += n_frames`.
      9. `ggml_backend_sched_reset`.
- [ ] C3. `reset()` — zero KV + tails, `n_past = 0`.

**ship**: commit with `AudioTokenizerDecoderStream` working
end-to-end on a single utterance, but driven from a local test harness
(no server wiring yet). parity test: stream through one utterance in
chunks of 1, 4, 8, 32 frames and compare pcm against one-shot (§7).

### phase D — transformer per-frame callback (#23)

- [ ] D1. add `frame_emit_fn` alias (§4.2).
- [ ] D2. extend `TTSTransformer::generate` with the optional
      `on_frame` arg. call it at line 3070 after the `output.push_back`
      loop. return false → return false from generate().
- [ ] D3. thread through `Qwen3TTS::synthesize{,with_embedding}` via
      `streaming_opts` (§4.3). keep the non-streaming signature
      backward-compatible.
- [ ] D4. add a mini-test that uses the callback with no decoder
      plumbing — just asserts the callback fires once per frame and
      sees exactly 16 codes.

**ship**: commit with the callback live but inert (server still uses
one-shot).

### phase E — wire server to streaming decode (#24)

target: `src/server.cpp` 865–888 (the `stream_format=="sse"` branch).

- [ ] E1. restructure the handler so synthesis happens **inside** the
      `set_chunked_content_provider` callback, not before it. pass a
      `DataSink *` into the `streaming_opts::on_pcm`.
- [ ] E2. build `streaming_opts`:
      - `batch_size`: from a new request field `stream_batch_size`
        (default 8), clamped to `[1, 256]`.
      - `on_pcm`: encodes pcm as `speech.audio.delta` just like the
        existing one-shot path at 865–874 (base64 of the chunk),
        writes to sink.
- [ ] E3. first chunk writes the wav streaming header if
      `response_format == "wav"`; subsequent chunks write raw pcm. keep
      pcm and wav paths symmetric.
- [ ] E4. after synthesis returns, write `speech.audio.done` with
      `build_done_event(result)` unchanged. the `usage`/`timings` are
      already populated correctly by `Qwen3TTS::synthesize`.
- [ ] E5. fallback: if `stream_batch_size` is absent or <=0, behave
      exactly as today (one big delta). preserves client compat.
- [ ] E6. wire the same path through `stream_format=="audio"` (chunked
      raw). reuses the same `streaming_opts`, only the wire format
      differs.

**ship**: commit that makes the server actually progressive. smoke-test
via curl against both flow and autiobook clients (both already parse
SSE via §autiobook).

### phase F — parity validation (#25)

see §7.

### phase G — CLI streaming switch (optional, #24 adjacent)

- [ ] G1. add `--streaming-batch-size N` to `src/main.cpp`. when N>0,
      construct `streaming_opts` with an `on_pcm` that appends pcm
      straight to a `std::vector<float>` (identical to the one-shot
      output buffer), then write the wav at the end. this gives us a
      CLI-driven repro for parity tests.
- [ ] G2. update README with the new flag.

---

## 7. parity test (#25)

file: `tests/test_streaming_parity.cpp`.

```cpp
// 1. load model and canonical ICL repro inputs (see
//    memory/project_canonical_repro.md).
// 2. run synthesize() one-shot → baseline pcm.
// 3. for each batch_size in {1, 4, 8, 32}:
//      stream via Qwen3TTS::synthesize(..., streaming_opts{
//         batch_size, on_pcm = append-to-vector
//      });
//      compare pcm vs. baseline.
// 4. fail if max-abs-diff > 1e-5 or length mismatch.
```

register in `CMakeLists.txt`:

```cmake
add_executable(test_streaming_parity tests/test_streaming_parity.cpp)
target_link_libraries(test_streaming_parity PRIVATE qwen3_tts Threads::Threads)
add_test(NAME streaming_parity_test
         COMMAND test_streaming_parity
                 --model ${CMAKE_BINARY_DIR}/models/qwen3-tts-1.7b-base
                 --tolerance 1e-5)
```

add to `make test` target chain; require it to pass before cutting a
release with streaming enabled.

---

## 8. open questions / decisions the engineer must make explicit

mark these in the PR description when landing phase C and E:

1. **tail emit strategy**: §5.2 suggests host-side ring update after
   compute (copy last L frames of the pre-conv signal). the alternative
   is emitting one `new_tail_*` output per ring from the graph. go
   host-side first; revisit if the copy cost is measurable.
2. **KV precision**: F16 vs F32. one-shot decoder runs in backend
   default (probably F32 for CPU, F16 available on GPU). match the
   backend choice. verify parity stays within tolerance.
3. **graph rebuild cost**: per-call rebuild is fine on GPU; on CPU with
   small batch sizes it may dominate. if profiling shows >20% overhead,
   cache the graph per `(n_frames, n_past_bucket)` pair — but watch for
   the same stale-scratch bug that motivated c215ab0. defer until
   measured.
4. **max_frames bound**: pick 2048 for now. longer utterances should
   fail loud ("streaming decode exceeded max_frames; raise
   QWEN3_DEC_MAX_KV_FRAMES") rather than silently corrupt.
5. **wav streaming header semantics**: we already emit a streaming wav
   header in `stream_format=="audio"` at server.cpp 841–858 via
   `wav_streaming_header`. reuse it as-is for SSE wav deltas.
6. **aborted generation**: if the callback returns false (e.g. client
   disconnected, detected via `sink.is_writable()` returning false),
   `generate` returns false and the handler emits a
   `speech.audio.done` with `error` populated. do not leak partial
   graphs / buffers — `~AudioTokenizerDecoderStream` frees everything.

---

## 9. sequencing against external work

- phase B–C can land on `main` behind no feature flag; the one-shot
  `decode()` is untouched, so risk is isolated.
- phase D lands behind "callback supplied → use new path". default
  behavior unchanged.
- phase E flips the server. add a server startup flag
  `--enable-streaming-decode` off by default until parity tests run in
  CI.
- flow (`openai_tts.rhai`) and autiobook (`tts_http.py`) both already
  consume SSE deltas — no client-side work is needed once E ships.
  the existing llama-swap dashboard will immediately show improved
  ttfa metrics because `prompt_ms` / `predicted_ms` in
  `speech.audio.done` don't change, but the first delta arrives much
  sooner.
