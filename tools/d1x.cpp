// d1 noul with early exit. the whole prompt (text + image) goes in one embedding batch, so one graph run
// can stop after layer L when a linear probe on the last token is confident.
//
// usage: d1x <model.gguf> <mmproj.gguf|-> [probe.bin]
// stdin, one request per line: <image path or -> TAB <prompt, \n escaped, image as <__media__>> TAB <mode>
//   mode: full | exit | dump:<file>   (dump appends (n_layer+1) x n_embd floats: l_out-* then result_norm)
// stdout, one json line per request.
// probe.bin: int32 layer, float bias, float lo, float hi, n_embd floats w. p = sigmoid(w . rmsnorm(h) + bias),
//   applied to l_out-<layer> of the last token. exit if p >= hi or p <= lo.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) { return std::chrono::duration<double, std::milli>(clk::now() - t).count(); }

struct state {
    int n_layer = 0, n_embd = 0;
    // what the callback does
    enum { NONE, EMBED, DUMP, EXIT } mode = NONE;
    std::vector<float> rows;      // DUMP: (n_layer + 1) x n_embd, EMBED: n_tokens x n_embd
    int probe_layer = -1; float probe_b = 0, probe_lo = 0, probe_hi = 1; std::vector<float> probe_w;
    struct probe { int layer; float b, lo, hi; std::vector<float> w; };
    std::vector<probe> probes; // several exit points: "a.bin,b.bin" on the command line
    int exit_layer = -1;
    float p_probe = NAN; bool exited = false;
    int n_heads = 0; std::vector<float> heads_b, heads_w, heads_p; // multi-head probe (D1X_HEADS=1), always exits
};

static std::string slot_name(int il) { return "l_out-" + std::to_string(il); }

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = (state *) ud;
    switch (s->mode) {
        case state::NONE: return ask ? false : true;
        case state::EMBED: {
            if (strcmp(t->name, "model.embed_tokens") != 0) return ask ? false : true;
            if (ask) return true;
            s->rows.resize(ggml_nelements(t));
            ggml_backend_tensor_get(t, s->rows.data(), 0, ggml_nbytes(t));
            return false; // got what we need, stop the graph
        }
        case state::DUMP: {
            int slot = -1;
            if (strncmp(t->name, "l_out-", 6) == 0) slot = atoi(t->name + 6);
            else if (strcmp(t->name, "result_norm") == 0) slot = s->n_layer;
            if (ask) return slot >= 0;
            if (slot < 0) return true;
            const int64_t n = t->ne[0];
            ggml_backend_tensor_get(t, s->rows.data() + (size_t) slot * n, (t->ne[1] - 1) * t->nb[1], n * sizeof(float));
            return true;
        }
        case state::EXIT: {
            if (!s->probes.empty() && s->n_heads == 0) {
                const state::probe * pr = nullptr;
                for (const auto & q : s->probes) if (slot_name(q.layer) == t->name) pr = &q;
                if (!pr) return ask ? false : true;
                if (ask) return true;
                const int64_t n = t->ne[0];
                std::vector<float> h(n);
                ggml_backend_tensor_get(t, h.data(), (t->ne[1] - 1) * t->nb[1], n * sizeof(float));
                double ss = 0; for (float v : h) ss += (double) v * v;
                const double inv = 1.0 / std::sqrt(ss / n + 1e-5);
                double z = pr->b; for (int64_t i = 0; i < n; i++) z += pr->w[i] * h[i] * inv;
                s->p_probe = 1.0f / (1.0f + std::exp(-z));
                s->exited = s->p_probe >= pr->hi || s->p_probe <= pr->lo;
                if (s->exited) s->exit_layer = pr->layer + 1;
                return !s->exited;
            }
            if (s->probe_layer < 0 || slot_name(s->probe_layer) != t->name) return ask ? false : true;
            if (ask) return true;
            const int64_t n = t->ne[0];
            std::vector<float> h(n);
            ggml_backend_tensor_get(t, h.data(), (t->ne[1] - 1) * t->nb[1], n * sizeof(float));
            double ss = 0; for (float v : h) ss += (double) v * v;
            const double inv = 1.0 / std::sqrt(ss / n + 1e-5);
            if (s->n_heads > 0) {
                s->heads_p.resize(s->n_heads);
                for (int k = 0; k < s->n_heads; k++) {
                    double zk = s->heads_b[k];
                    const float * w = s->heads_w.data() + (size_t) k * n;
                    for (int64_t i = 0; i < n; i++) zk += w[i] * h[i] * inv;
                    s->heads_p[k] = 1.0f / (1.0f + std::exp(-zk));
                }
                s->exited = true;
                return false;
            }
            double z = s->probe_b; for (int64_t i = 0; i < n; i++) z += s->probe_w[i] * h[i] * inv;
            s->p_probe = 1.0f / (1.0f + std::exp(-z));
            s->exited = s->p_probe >= s->probe_hi || s->p_probe <= s->probe_lo;
            return !s->exited;
        }
    }
    return true;
}

static std::string unescape(const std::string & in) {
    std::string o;
    for (size_t i = 0; i < in.size(); i++) {
        if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == 'n') { o += '\n'; i++; }
        else if (in[i] == '\\' && i + 1 < in.size() && in[i + 1] == '\\') { o += '\\'; i++; }
        else o += in[i];
    }
    return o;
}

static std::vector<llama_token> single_tokens(const llama_vocab * v, std::vector<std::string> forms) {
    std::vector<llama_token> out;
    for (auto & f : forms) {
        llama_token t[8];
        if (llama_tokenize(v, f.c_str(), f.size(), t, 8, false, false) == 1) out.push_back(t[0]);
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model mmproj|- [probe.bin]\n", argv[0]); return 1; }
    llama_backend_init();
    llama_log_set([](ggml_log_level l, const char * s, void *) { if (l >= GGML_LOG_LEVEL_ERROR) fputs(s, stderr); }, nullptr);

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    state st;
    st.n_layer = llama_model_n_layer(model);
    st.n_embd  = llama_model_n_embd(model);

    if (argc > 3 && getenv("D1X_HEADS")) {
        // int32 layer, int32 K, K biases, K x n_embd weights
        FILE * f = fopen(argv[3], "rb");
        int32_t L, K; fread(&L, 4, 1, f); fread(&K, 4, 1, f);
        st.probe_layer = L; st.n_heads = K;
        st.heads_b.resize(K); fread(st.heads_b.data(), 4, K, f);
        st.heads_w.resize((size_t) K * st.n_embd); fread(st.heads_w.data(), 4, st.heads_w.size(), f); fclose(f);
        fprintf(stderr, "heads: %d at layer %d\n", K, L);
    } else if (argc > 3 && strchr(argv[3], ',')) {
        std::string list = argv[3];
        for (size_t a = 0; a <= list.size();) {
            size_t z = list.find(',', a); if (z == std::string::npos) z = list.size();
            FILE * f = fopen(list.substr(a, z - a).c_str(), "rb");
            state::probe q; int32_t L;
            fread(&L, 4, 1, f); q.layer = L; fread(&q.b, 4, 1, f); fread(&q.lo, 4, 1, f); fread(&q.hi, 4, 1, f);
            q.w.resize(st.n_embd); fread(q.w.data(), 4, st.n_embd, f); fclose(f);
            st.probes.push_back(q);
            fprintf(stderr, "exit point at layer %d\n", L + 1);
            a = z + 1;
        }
    } else if (argc > 3) {
        FILE * f = fopen(argv[3], "rb");
        int32_t L; fread(&L, 4, 1, f); st.probe_layer = L;
        fread(&st.probe_b, 4, 1, f); fread(&st.probe_lo, 4, 1, f); fread(&st.probe_hi, 4, 1, f);
        st.probe_w.resize(st.n_embd); fread(st.probe_w.data(), 4, st.n_embd, f); fclose(f);
        fprintf(stderr, "probe: layer %d, exit if p<=%.3f or p>=%.3f\n", L, st.probe_lo, st.probe_hi);
    }

    auto cp = llama_context_default_params();
    cp.n_ctx = 4096; cp.n_batch = 2048; cp.n_ubatch = 2048;
    cp.n_seq_max = 8; cp.kv_unified = true; // multi mode: one branch per question
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = cb; cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);

    mtmd_context * mctx = nullptr;
    if (strcmp(argv[2], "-") != 0) {
        auto mcp = mtmd_context_params_default();
        mcp.use_gpu = true; mcp.print_timings = false;
        if (getenv("D1X_IMG_MIN")) mcp.image_min_tokens = atoi(getenv("D1X_IMG_MIN"));
        if (getenv("D1X_IMG_MAX")) mcp.image_max_tokens = atoi(getenv("D1X_IMG_MAX"));
        mctx = mtmd_init_from_file(argv[2], model, mcp);
        if (!mctx) return 1;
    }

    const auto yes = single_tokens(vocab, {"yes", "Yes", "YES"});
    const auto no  = single_tokens(vocab, {"no", "No", "NO"});

    // embeddings of text token runs, cached: the text around the image is the same every frame
    std::map<std::vector<llama_token>, std::vector<float>> embd_cache;
    auto embed_text = [&](const std::vector<llama_token> & toks) -> const std::vector<float> & {
        auto it = embd_cache.find(toks);
        if (it != embd_cache.end()) return it->second;
        st.mode = state::EMBED;
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(toks.data()), toks.size()));
        st.mode = state::NONE;
        return embd_cache[toks] = st.rows;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
        const std::string img = line.substr(0, t1), full = unescape(line.substr(t1 + 1, t2 - t1 - 1)), mode = line.substr(t2 + 1);
        // multi mode: "<prefix><<BRANCH>><question 1 tail><<BRANCH>><question 2 tail>..." -> every branch continues the prefix
        std::vector<std::string> parts;
        for (size_t a = 0;;) {
            const size_t z = full.find("<<BRANCH>>", a);
            parts.push_back(full.substr(a, z == std::string::npos ? std::string::npos : z - a));
            if (z == std::string::npos) break;
            a = z + 10;
        }
        const std::string prompt = parts[0];
        const int n_br = (int) parts.size() - 1;

        // 1. build the whole prompt as embeddings
        const auto t_enc = clk::now();
        std::vector<float> embd;
        if (img == "-") {
            std::vector<llama_token> toks(prompt.size() + 16);
            toks.resize(llama_tokenize(vocab, prompt.c_str(), prompt.size(), toks.data(), toks.size(), false, true));
            embd = embed_text(toks);
        } else {
            mtmd_helper_init_opt opt = {};
            auto bw = mtmd_helper_bitmap_init_from_file(mctx, img.c_str(), false, opt);
            mtmd_input_text txt = {prompt.c_str(), prompt.size(), false, true};
            mtmd_input_chunks * chunks = mtmd_input_chunks_init();
            const mtmd_bitmap * bmps[1] = {bw.bitmap};
            if (mtmd_tokenize(mctx, chunks, &txt, bmps, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
                const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                    size_t n; const llama_token * tk = mtmd_input_chunk_get_tokens_text(c, &n);
                    const auto & e = embed_text(std::vector<llama_token>(tk, tk + n));
                    embd.insert(embd.end(), e.begin(), e.end());
                } else {
                    if (mtmd_encode_chunk(mctx, c) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
                    const float * e = mtmd_get_output_embd(mctx);
                    embd.insert(embd.end(), e, e + mtmd_input_chunk_get_n_tokens(c) * st.n_embd);
                }
            }
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bw.bitmap);
        }
        const double ms_enc = ms_since(t_enc);
        const int n_tok = embd.size() / st.n_embd;

        // embd:<file> -> append int32 n_tok + n_tok x n_embd fp16 (the exact model input), no decode
        if (mode.rfind("embd:", 0) == 0) {
            FILE * f = fopen(mode.substr(5).c_str(), "ab");
            int32_t n = n_tok; fwrite(&n, 4, 1, f);
            std::vector<ggml_fp16_t> h(embd.size());
            ggml_fp32_to_fp16_row(embd.data(), h.data(), embd.size());
            fwrite(h.data(), sizeof(ggml_fp16_t), h.size(), f);
            fclose(f);
            printf("{\"n_tok\": %d}\n", n_tok); fflush(stdout);
            continue;
        }

        // 2. one decode over everything. in multi mode the prefix belongs to every branch's sequence,
        //    and each branch's tokens only to its own, so the mask keeps branches apart
        std::vector<int> br_len, br_last;
        int n_all = n_tok;
        for (int k = 1; k <= n_br; k++) {
            std::vector<llama_token> tk(parts[k].size() + 16);
            tk.resize(llama_tokenize(vocab, parts[k].c_str(), parts[k].size(), tk.data(), tk.size(), false, true));
            const auto & e = embed_text(tk);
            embd.insert(embd.end(), e.begin(), e.end());
            br_len.push_back(tk.size());
            n_all += tk.size();
            br_last.push_back(n_all - 1);
        }
        const int n_seq = std::max(1, n_br);
        llama_batch b = llama_batch_init(n_all, st.n_embd, n_seq);
        memcpy(b.embd, embd.data(), embd.size() * sizeof(float));
        for (int i = 0; i < n_tok; i++) {
            b.pos[i] = i; b.logits[i] = n_br == 0 && i == n_tok - 1;
            b.n_seq_id[i] = n_seq;
            for (int k = 0; k < n_seq; k++) b.seq_id[i][k] = k;
        }
        for (int k = 0, i = n_tok; k < n_br; k++) {
            for (int j = 0; j < br_len[k]; j++, i++) {
                b.pos[i] = n_tok + j; b.n_seq_id[i] = 1; b.seq_id[i][0] = k; b.logits[i] = i == br_last[k];
            }
        }
        b.n_tokens = n_all;

        if (mode.rfind("dump:", 0) == 0) { st.mode = state::DUMP; st.rows.assign((size_t) (st.n_layer + 1) * st.n_embd, 0.0f); }
        else if (mode == "exit") st.mode = state::EXIT;
        else st.mode = state::NONE;
        st.exited = false; st.p_probe = NAN; st.heads_p.clear(); st.exit_layer = -1;

        llama_memory_clear(llama_get_memory(ctx), true);
        const auto t_lm = clk::now();
        const int rc = llama_decode(ctx, b);
        llama_synchronize(ctx);
        if (rc != 0 && st.mode != state::EXIT) fprintf(stderr, "decode rc %d\n", rc);
        const double ms_lm = ms_since(t_lm);
        llama_batch_free(b);

        auto read_yes = [&](int idx) {
            const float * lg = llama_get_logits_ith(ctx, idx);
            float my = -INFINITY, mn = -INFINITY;
            for (auto t : yes) my = std::max(my, lg[t]);
            for (auto t : no)  mn = std::max(mn, lg[t]);
            return 1.0 / (1.0 + std::exp((double) mn - my));
        };
        double p_yes = NAN;
        std::string ms_out = "[";
        if (!st.exited && n_br == 0) p_yes = read_yes(-1);
        for (int k = 0; !st.exited && k < n_br; k++) ms_out += (k ? ", " : "") + std::to_string(read_yes(br_last[k]));
        ms_out += "]";
        if (st.mode == state::DUMP) {
            FILE * f = fopen(mode.substr(5).c_str(), "ab");
            fwrite(st.rows.data(), sizeof(float), st.rows.size(), f);
            fclose(f);
        }
        st.mode = state::NONE;
        std::string hs = "[";
        for (size_t k = 0; st.exited && k < st.heads_p.size(); k++) hs += (k ? ", " : "") + std::to_string(st.heads_p[k]);
        hs += "]";
        printf("{\"exit_layer\": %d, \"multi\": %s, \"heads\": %s, \"p_yes\": %s, \"p_probe\": %s, \"exited\": %s, \"n_tok\": %d, \"ms_enc\": %.2f, \"ms_lm\": %.2f}\n",
               st.exit_layer, ms_out.c_str(), hs.c_str(),
               std::isnan(p_yes) ? "null" : std::to_string(p_yes).c_str(),
               std::isnan(st.p_probe) ? "null" : std::to_string(st.p_probe).c_str(),
               st.exited ? "true" : "false", n_all, ms_enc, ms_lm);
        fflush(stdout);
    }
    return 0;
}
