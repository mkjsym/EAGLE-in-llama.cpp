#include "sampling.h"

#include "common.h"

#include <cmath>
#include <unordered_map>

// the ring buffer works similarly to std::deque, but with a fixed capacity
// TODO: deduplicate with llama-impl.h
template<typename T>
struct ring_buffer {
    ring_buffer(size_t cap) : capacity(cap), data(cap) {}

    T & front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    const T & front() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[first];
    }

    T & back() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    const T & back() const {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        return data[pos];
    }

    void push_back(const T & value) {
        if (sz == capacity) {
            // advance the start when buffer is full
            first = (first + 1) % capacity;
        } else {
            sz++;
        }
        data[pos] = value;
        pos = (pos + 1) % capacity;
    }

    T pop_front() {
        if (sz == 0) {
            throw std::runtime_error("ring buffer is empty");
        }
        T value = data[first];
        first = (first + 1) % capacity;
        sz--;
        return value;
    }

    const T & rat(size_t i) const {
        if (i >= sz) {
            throw std::runtime_error("ring buffer: index out of bounds");
        }
        return data[(first + sz - i - 1) % capacity];
    }

    std::vector<T> to_vector() const {
        std::vector<T> result;
        result.reserve(sz);
        for (size_t i = 0; i < sz; i++) {
            result.push_back(data[(first + i) % capacity]);
        }
        return result;
    }

    void clear() {
        // here only reset the status of the buffer
        sz = 0;
        first = 0;
        pos = 0;
    }

    bool empty() const {
        return sz == 0;
    }

    size_t size() const {
        return sz;
    }

    size_t capacity = 0;
    size_t sz = 0;
    size_t first = 0;
    size_t pos = 0;
    std::vector<T> data;
};

struct common_sampler {
    common_params_sampling params;

    struct llama_sampler * grmr;
    struct llama_sampler * chain;

    ring_buffer<llama_token> prev;

    std::vector<llama_token_data> cur;

    llama_token_data_array cur_p;

    void set_logits(struct llama_context * ctx, int idx) {
        const auto * logits = llama_get_logits_ith(ctx, idx);

        const llama_model * model = llama_get_model(ctx);
        const llama_vocab * vocab = llama_model_get_vocab(model);

        const int n_vocab = llama_vocab_n_tokens(vocab);

        cur.resize(n_vocab);

        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur[token_id] = llama_token_data{token_id, logits[token_id], 0.0f};
        }

        cur_p = { cur.data(), cur.size(), -1, false };
    }
};

std::string common_params_sampling::print() const {
    char result[1024];

    snprintf(result, sizeof(result),
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, top_n_sigma = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f",
            penalty_last_n, penalty_repeat, penalty_freq, penalty_present,
            dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
            top_k, top_p, min_p, xtc_probability, xtc_threshold, typ_p, top_n_sigma, temp,
            mirostat, mirostat_eta, mirostat_tau);

    return std::string(result);
}

struct common_sampler * common_sampler_init(const struct llama_model * model, const struct common_params_sampling & params) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();

    lparams.no_perf = params.no_perf;

    struct llama_sampler * grmr;
    if (params.grammar.compare(0, 11, "%llguidance") == 0) {
#ifdef LLAMA_USE_LLGUIDANCE
        grmr = llama_sampler_init_llg(vocab, "lark", params.grammar.c_str());
#else
        GGML_ABORT("llguidance (cmake -DLLAMA_LLGUIDANCE=ON) is not enabled");
#endif // LLAMA_USE_LLGUIDANCE
    } else {
        std::vector<const char *> trigger_words;
        trigger_words.reserve(params.grammar_trigger_words.size());
        for (const auto & str : params.grammar_trigger_words) {
            trigger_words.push_back(str.word.c_str());
        }

        grmr = params.grammar_lazy
             ? llama_sampler_init_grammar_lazy(vocab, params.grammar.c_str(), "root",
                                               trigger_words.data(), trigger_words.size(),
                                               params.grammar_trigger_tokens.data(), params.grammar_trigger_tokens.size())
             :      llama_sampler_init_grammar(vocab, params.grammar.c_str(), "root");
    }

    auto * result = new common_sampler {
        /* .params = */ params,
        /* .grmr   = */ grmr,
        /* .chain  = */ llama_sampler_chain_init(lparams),
        /* .prev   = */ ring_buffer<llama_token>(std::max(32, params.n_prev)),
        /* .cur    = */ {},
        /* .cur_p  = */ {},
    };

    llama_sampler_chain_add(result->chain,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                params.logit_bias.size(),
                params.logit_bias.data()));

    if (params.mirostat == 0) {
        if (params.top_n_sigma >= 0) {
            llama_sampler_chain_add(result->chain, llama_sampler_init_top_k        (params.top_k));
            llama_sampler_chain_add(result->chain, llama_sampler_init_temp         (params.temp));
            llama_sampler_chain_add(result->chain, llama_sampler_init_top_n_sigma  (params.top_n_sigma));
        } else {
            for (const auto & cnstr : params.samplers) {
                switch (cnstr) {
                    case COMMON_SAMPLER_TYPE_DRY:
                        {
                            std::vector<const char *> c_breakers;
                            c_breakers.reserve(params.dry_sequence_breakers.size());
                            for (const auto & str : params.dry_sequence_breakers) {
                                c_breakers.push_back(str.c_str());
                            }

                            llama_sampler_chain_add(result->chain, llama_sampler_init_dry      (vocab, llama_model_n_ctx_train(model), params.dry_multiplier, params.dry_base, params.dry_allowed_length, params.dry_penalty_last_n, c_breakers.data(), c_breakers.size()));
                        }
                        break;
                    case COMMON_SAMPLER_TYPE_TOP_K:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_top_k    (params.top_k));
                        break;
                    case COMMON_SAMPLER_TYPE_TOP_P:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_top_p    (params.top_p, params.min_keep));
                        break;
                    case COMMON_SAMPLER_TYPE_MIN_P:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_min_p    (params.min_p, params.min_keep));
                        break;
                    case COMMON_SAMPLER_TYPE_XTC:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_xtc      (params.xtc_probability, params.xtc_threshold, params.min_keep, params.seed));
                        break;
                    case COMMON_SAMPLER_TYPE_TYPICAL_P:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_typical  (params.typ_p, params.min_keep));
                        break;
                    case COMMON_SAMPLER_TYPE_TEMPERATURE:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_temp_ext (params.temp, params.dynatemp_range, params.dynatemp_exponent));
                        break;
                    case COMMON_SAMPLER_TYPE_INFILL:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_infill   (vocab));
                        break;
                    case COMMON_SAMPLER_TYPE_PENALTIES:
                        llama_sampler_chain_add(result->chain, llama_sampler_init_penalties(params.penalty_last_n, params.penalty_repeat, params.penalty_freq, params.penalty_present));
                        break;
                    default:
                        GGML_ASSERT(false && "unknown sampler type");
                }
            }
        }
        llama_sampler_chain_add(result->chain, llama_sampler_init_dist(params.seed));
    } else if (params.mirostat == 1) {
        llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), params.seed, params.mirostat_tau, params.mirostat_eta, 100));
    } else if (params.mirostat == 2) {
        llama_sampler_chain_add(result->chain, llama_sampler_init_temp(params.temp));
        llama_sampler_chain_add(result->chain, llama_sampler_init_mirostat_v2(params.seed, params.mirostat_tau, params.mirostat_eta));
    } else {
        GGML_ASSERT(false && "unknown mirostat version");
    }

    return result;
}

void common_sampler_free(struct common_sampler * gsmpl) {
    if (gsmpl) {
        llama_sampler_free(gsmpl->grmr);

        llama_sampler_free(gsmpl->chain);

        delete gsmpl;
    }
}

void common_sampler_accept(struct common_sampler * gsmpl, llama_token token, bool accept_grammar) {
    if (accept_grammar) {
        llama_sampler_accept(gsmpl->grmr, token);
    }

    llama_sampler_accept(gsmpl->chain, token);

    gsmpl->prev.push_back(token);
}

void common_sampler_reset(struct common_sampler * gsmpl) {
    llama_sampler_reset(gsmpl->grmr);

    llama_sampler_reset(gsmpl->chain);
}

struct common_sampler * common_sampler_clone(common_sampler * gsmpl) {
    return new common_sampler {
        /* .params = */ gsmpl->params,
        /* .grmr   = */ llama_sampler_clone(gsmpl->grmr),
        /* .chain  = */ llama_sampler_clone(gsmpl->chain),
        /* .prev   = */ gsmpl->prev,
        /* .cur    = */ gsmpl->cur,
        /* .cur_p  = */ gsmpl->cur_p,
    };
}

void common_perf_print(const struct llama_context * ctx, const struct common_sampler * gsmpl) {
    // TODO: measure grammar performance

    if (gsmpl) {
        llama_perf_sampler_print(gsmpl->chain);
    }
    if (ctx) {
        llama_perf_context_print(ctx);
    }
}

// llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
//     gsmpl->set_logits(ctx, idx);

//     auto & grmr  = gsmpl->grmr;
//     auto & chain = gsmpl->chain;
//     auto & cur_p = gsmpl->cur_p; // initialized by set_logits

//     if (grammar_first) {
//         llama_sampler_apply(grmr, &cur_p);
//     }

//     llama_sampler_apply(chain, &cur_p);

//     GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

//     const llama_token id = cur_p.data[cur_p.selected].id;

//     if (grammar_first) {
//         return id;
//     }

//     // check if it the sampled token fits the grammar
//     {
//         llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
//         llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

//         llama_sampler_apply(grmr, &single_token_data_array);

//         const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
//         if (is_valid) {
//             return id;
//         }
//     }

//     // resampling:
//     // if the token is not valid, sample again, but first apply the grammar sampler and then the sampling chain
//     gsmpl->set_logits(ctx, idx);

//     llama_sampler_apply(grmr,  &cur_p);
//     llama_sampler_apply(chain, &cur_p);

//     GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

//     return cur_p.data[cur_p.selected].id;
// }

// 샘플러 객체(gsmpl), 컨텍스트(ctx), 로짓 인덱스(idx), 문법 우선 적용 여부(grammar_first)를 받아
// 다음 토큰 ID를 샘플링하여 반환하는 함수
llama_token common_sampler_sample(struct common_sampler * gsmpl, struct llama_context * ctx, int idx, bool grammar_first) {
    // 1. 초기 설정: 로짓 가져오기
    // gsmpl 내부 함수를 호출하여 ctx에서 idx 위치의 로짓을 가져와
    // 샘플러 내부의 후보 토큰 배열(cur_p)을 초기화함.
    gsmpl->set_logits(ctx, idx);

    // 2. 샘플러 객체 내 멤버 참조 설정
    auto & grmr  = gsmpl->grmr;  // 문법(grammar) 샘플러 참조
    auto & chain = gsmpl->chain; // 일반 샘플링 방법들의 체인(chain) 참조 (예: top-k, top-p, 온도 등)
    auto & cur_p = gsmpl->cur_p; // 현재 후보 토큰 및 확률 배열 참조 (set_logits에서 초기화됨)

    // 3. 첫 번째 샘플링 시도
    if (grammar_first) {
        // 3.1. 문법 우선 적용 (grammar_first = true)
        // 문법 샘플러(grmr)를 먼저 적용하여 문법에 맞지 않는 토큰들의 확률을 0 또는 음의 무한대로 만듦.
        llama_sampler_apply(grmr, &cur_p);
    }

    // 3.2. 일반 샘플링 체인 적용
    // 온도, top-k, top-p, Mirostat 등 설정된 샘플링 방법들을 순차적으로 적용.
    // 이 과정에서 최종적으로 하나의 토큰이 선택됨 (cur_p.selected에 인덱스 저장).
    llama_sampler_apply(chain, &cur_p);

    // 샘플링 후 토큰이 선택되었는지 확인 (설정 오류 등으로 선택되지 않을 수 있음)
    GGML_ASSERT(cur_p.selected != -1 && "no selected token during sampling - check your sampling configuration");

    // 선택된 토큰의 ID 가져오기
    const llama_token id = cur_p.data[cur_p.selected].id;

    // 4. 결과 반환 또는 문법 재검사/재샘플링
    if (grammar_first) {
        // 4.1. 문법이 먼저 적용된 경우:
        // 이미 문법 검사를 통과한 상태에서 샘플링되었으므로, 선택된 id는 항상 문법에 유효함.
        // 따라서 바로 반환.
        return id;
    }

    // 4.2. 문법이 나중에 적용되는 경우 (grammar_first = false):
    //    - 일반 샘플링 체인에서 선택된 'id'가 문법에 맞는지 확인해야 함.
    {
        // 선택된 토큰 'id' 하나만 포함하는 임시 데이터 배열 생성
        llama_token_data       single_token_data       = { id, 1.0f, 0.0f };
        llama_token_data_array single_token_data_array = { &single_token_data, 1, -1, false };

        // 문법 샘플러(grmr)를 이 단일 토큰 배열에 적용
        llama_sampler_apply(grmr, &single_token_data_array);

        // 문법 샘플러 적용 후 로짓(logit) 값이 -INFINITY가 아니면 유효한 토큰임
        const bool is_valid = single_token_data_array.data[0].logit != -INFINITY;
        if (is_valid) {
            // 유효하다면, 원래 샘플링된 id 반환
            return id;
        }
    }

    // 4.3. 재샘플링 (Resampling):
    //    - 일반 샘플링으로 선택된 토큰 'id'가 문법 검사를 통과하지 못한 경우.
    //    - 문법을 먼저 적용한 후 다시 샘플링을 수행해야 함.

    // 중요: 재샘플링을 위해 로짓을 다시 원본 상태로 설정해야 함!
    // 이전 llama_sampler_apply(chain, ...) 호출이 cur_p를 변경했기 때문.
    gsmpl->set_logits(ctx, idx);

    // 이번에는 문법 샘플러(grmr)를 *먼저* 적용하여 유효한 토큰들만 남김.
    llama_sampler_apply(grmr,  &cur_p);
    // 그 다음, 문법을 통과한 토큰들 중에서 일반 샘플링 체인(chain)을 적용하여 최종 토큰 선택.
    llama_sampler_apply(chain, &cur_p);

    // 재샘플링 후에도 토큰이 선택되었는지 확인
    GGML_ASSERT(cur_p.selected != -1 && "no selected token during re-sampling - check your sampling configuration");

    // 재샘플링을 통해 선택된 (문법에 맞는) 토큰 ID 반환
    return cur_p.data[cur_p.selected].id;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const std::vector<int> & idxs, const llama_tokens & draft, bool grammar_first) {
    GGML_ASSERT(idxs.size() == draft.size() + 1 && "idxs.size() must be draft.size() + 1");

    std::vector<llama_token> result;
    result.reserve(idxs.size());

    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);

        if (draft[i] != id) {
            break;
        }
    }

    if (i == draft.size()) {
        const llama_token id = common_sampler_sample(gsmpl, ctx, idxs[i], grammar_first);

        common_sampler_accept(gsmpl, id, true);

        result.push_back(id);
    }

    return result;
}

std::vector<llama_token> common_sampler_sample_and_accept_n(struct common_sampler * gsmpl, struct llama_context * ctx, const llama_tokens & draft, bool grammar_first) {
    std::vector<int> idxs(draft.size() + 1);
    for (size_t i = 0; i < idxs.size(); ++i) {
        idxs[i] = i;
    }

    return common_sampler_sample_and_accept_n(gsmpl, ctx, idxs, draft, grammar_first);
}

uint32_t common_sampler_get_seed(const struct common_sampler * gsmpl) {
    return llama_sampler_get_seed(gsmpl->chain);
}

// helpers

llama_token_data_array * common_sampler_get_candidates(struct common_sampler * gsmpl) {
    return &gsmpl->cur_p;
}

llama_token common_sampler_last(const struct common_sampler * gsmpl) {
    return gsmpl->prev.rat(0);
}

std::string common_sampler_print(const struct common_sampler * gsmpl) {
    std::string result = "logits ";

    for (int i = 0; i < llama_sampler_chain_n(gsmpl->chain); i++) {
        const auto * smpl = llama_sampler_chain_get(gsmpl->chain, i);
        result += std::string("-> ") + llama_sampler_name(smpl) + " ";
    }

    return result;
}

std::string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx_main, int n) {
    n = std::min(n, (int) gsmpl->prev.size());

    if (n <= 0) {
        return "";
    }

    std::string result;
    result.reserve(8*n); // 8 is the average length of a token [citation needed], TODO: compute this from the vocab

    for (int i = n - 1; i >= 0; i--) {
        const llama_token id = gsmpl->prev.rat(i);

        GGML_ASSERT(id != LLAMA_TOKEN_NULL && "null token in the sampling history - should not happen");

        result += common_token_to_piece(ctx_main, id);
    }

    return result;
}

char common_sampler_type_to_chr(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return 'd';
        case COMMON_SAMPLER_TYPE_TOP_K:       return 'k';
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return 'y';
        case COMMON_SAMPLER_TYPE_TOP_P:       return 'p';
        case COMMON_SAMPLER_TYPE_MIN_P:       return 'm';
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return 't';
        case COMMON_SAMPLER_TYPE_XTC:         return 'x';
        case COMMON_SAMPLER_TYPE_INFILL:      return 'i';
        case COMMON_SAMPLER_TYPE_PENALTIES:   return 'e';
        default : return '?';
    }
}

std::string common_sampler_type_to_str(enum common_sampler_type cnstr) {
    switch (cnstr) {
        case COMMON_SAMPLER_TYPE_DRY:         return "dry";
        case COMMON_SAMPLER_TYPE_TOP_K:       return "top_k";
        case COMMON_SAMPLER_TYPE_TYPICAL_P:   return "typ_p";
        case COMMON_SAMPLER_TYPE_TOP_P:       return "top_p";
        case COMMON_SAMPLER_TYPE_MIN_P:       return "min_p";
        case COMMON_SAMPLER_TYPE_TEMPERATURE: return "temperature";
        case COMMON_SAMPLER_TYPE_XTC:         return "xtc";
        case COMMON_SAMPLER_TYPE_INFILL:      return "infill";
        case COMMON_SAMPLER_TYPE_PENALTIES:   return "penalties";
        default : return "";
    }
}

std::vector<common_sampler_type> common_sampler_types_from_names(const std::vector<std::string> & names, bool allow_alt_names) {
    std::unordered_map<std::string, common_sampler_type> sampler_canonical_name_map {
        { "dry",         COMMON_SAMPLER_TYPE_DRY },
        { "top_k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top_p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "typ_p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min_p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "temperature", COMMON_SAMPLER_TYPE_TEMPERATURE },
        { "xtc",         COMMON_SAMPLER_TYPE_XTC },
        { "infill",      COMMON_SAMPLER_TYPE_INFILL },
        { "penalties",   COMMON_SAMPLER_TYPE_PENALTIES },
    };

    // since samplers names are written multiple ways
    // make it ready for both system names and input names
    std::unordered_map<std::string, common_sampler_type> sampler_alt_name_map {
        { "top-k",       COMMON_SAMPLER_TYPE_TOP_K },
        { "top-p",       COMMON_SAMPLER_TYPE_TOP_P },
        { "nucleus",     COMMON_SAMPLER_TYPE_TOP_P },
        { "typical-p",   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typical",     COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ-p",       COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "typ",         COMMON_SAMPLER_TYPE_TYPICAL_P },
        { "min-p",       COMMON_SAMPLER_TYPE_MIN_P },
        { "temp",        COMMON_SAMPLER_TYPE_TEMPERATURE },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(names.size());

    for (const auto & name : names) {
        auto sampler = sampler_canonical_name_map.find(name);
        if (sampler != sampler_canonical_name_map.end()) {
            samplers.push_back(sampler->second);
        } else {
            if (allow_alt_names) {
                sampler = sampler_alt_name_map.find(name);
                if (sampler != sampler_alt_name_map.end()) {
                    samplers.push_back(sampler->second);
                }
            }
        }
    }

    return samplers;
}

std::vector<common_sampler_type> common_sampler_types_from_chars(const std::string & chars) {
    std::unordered_map<char, common_sampler_type> sampler_name_map = {
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_DRY),         COMMON_SAMPLER_TYPE_DRY },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_K),       COMMON_SAMPLER_TYPE_TOP_K },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TYPICAL_P),   COMMON_SAMPLER_TYPE_TYPICAL_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TOP_P),       COMMON_SAMPLER_TYPE_TOP_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_MIN_P),       COMMON_SAMPLER_TYPE_MIN_P },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_TEMPERATURE), COMMON_SAMPLER_TYPE_TEMPERATURE },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_XTC),         COMMON_SAMPLER_TYPE_XTC },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_INFILL),      COMMON_SAMPLER_TYPE_INFILL },
        { common_sampler_type_to_chr(COMMON_SAMPLER_TYPE_PENALTIES),   COMMON_SAMPLER_TYPE_PENALTIES },
    };

    std::vector<common_sampler_type> samplers;
    samplers.reserve(chars.size());

    for (const auto & c : chars) {
        const auto sampler = sampler_name_map.find(c);
        if (sampler != sampler_name_map.end()) {
            samplers.push_back(sampler->second);
        }
    }

    return samplers;
}
