#include "speculative.h"

#include "log.h"
#include "common.h"
#include "sampling.h"

#include <cstring>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct common_speculative {
    struct llama_context * ctx;
    struct common_sampler * smpl;

    llama_batch batch;
    llama_tokens prompt;
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_dft) {
    auto * result = new common_speculative {
        /* .ctx    = */ ctx_dft,
        /* .smpl   = */ nullptr,
        /* .batch  = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .prompt = */ {},
    };

    // TODO: optimize or pass from outside?
#if 0
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 40;
        params.top_p = 0.9;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#else
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 10;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#endif

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    common_sampler_free(spec->smpl);

    llama_batch_free(spec->batch);

    delete spec;
}

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but "
                     "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)) {
        LOG_ERR("%s: draft vocab special tokens must match target vocab to use speculation\n", __func__);
        LOG_ERR("%s: tgt: bos = %d (%d), eos = %d (%d)\n", __func__, llama_vocab_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_tgt), llama_vocab_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_tgt));
        LOG_ERR("%s: dft: bos = %d (%d), eos = %d (%d)\n", __func__, llama_vocab_bos(vocab_dft), llama_vocab_get_add_bos(vocab_dft), llama_vocab_eos(vocab_dft), llama_vocab_get_add_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);

        const int vocab_diff = std::abs(n_vocab_tgt - n_vocab_dft);

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but "
                         "target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    __func__, n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft vocab vocab must match target vocab to use speculation but "
                             "token %d content differs - target '%s', draft '%s'\n", __func__, i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// llama_tokens는 std::vector<llama_token>의 별칭(alias)일 가능성이 높음
llama_tokens common_speculative_gen_draft(
    struct common_speculative * spec, // Draft 모델의 상태 및 헬퍼 객체 포인터 (주석 정확)
    struct common_speculative_params params, // Speculative decoding 파라미터 (주석 수정 필요: Target 모델 구조체가 아님)
    const llama_tokens & prompt_tgt, // Target 모델이 처리한 토큰 히스토리 (주석 수정 필요: 단순 입력 프롬프트 아님)
    llama_token id_last,
    struct llama_context * ctx_tgt,
    const std::vector<float>& initial_hidden_state) { // Target 모델이 마지막으로 '수락(accept)'한 토큰 ID

// 1. Speculator 객체(spec) 내부 멤버들에 대한 참조(reference) 설정
auto & batch  = spec->batch; // Draft 모델용 배치(batch) 객체
auto & ctx    = spec->ctx;   // Draft 모델의 llama_context
auto & smpl   = spec->smpl;  // Draft 모델용 샘플러(sampler)
auto & prompt = spec->prompt;// Draft 모델이 자체적으로 추적하는 내부 토큰 히스토리

// 2. KV 캐시 재사용(Reuse) 로직: 이전 스텝과의 연관성 찾기
int reuse_i = 0; // Draft 모델 내부 히스토리(prompt)에서 재사용 시작 위치
int reuse_n = 0; // 재사용할 토큰의 개수 (가장 긴 공통 접두사 길이)

// Draft 컨텍스트에서 draft 토큰을 생성할 공간을 제외한 최대 히스토리 길이 계산
const int n_ctx = llama_n_ctx(ctx) - params.n_draft;
// Target 모델 히스토리(prompt_tgt)에서 비교를 시작할 위치 계산
const int i_start = std::max<int>(1, (int) prompt_tgt.size() - n_ctx);
//i_start += 1;

int ids_size = initial_hidden_state.size() / 4096 - 1;
//printf("draft phase ids_size: %d\n", ids_size);

// Draft 모델의 이전 내부 히스토리(prompt)와 Target 모델의 최신 히스토리(prompt_tgt)를 비교하여
// 재사용 가능한 가장 긴 공통 시퀀스(prefix)를 찾는다. 이는 Draft 모델의 KV 캐시를 최대한 활용하기 위함.
for (int i = 0; i < (int) prompt.size(); ++i) { // Draft 내부 히스토리 순회
    int cur = 0; // 현재 일치하는 길이
    // Target 히스토리의 끝부분과 Draft 히스토리의 부분을 비교
    while (i_start + cur < (int) prompt_tgt.size() &&
           i       + cur < (int) prompt.size() &&
           prompt_tgt[i_start + cur] == prompt[i + cur]) {
        cur++; // 일치하면 길이 증가
    }

    cur = (cur - (ids_size)) > 0 ? (cur - (ids_size)) : cur; // 초기 히든 스테이트 길이만큼 감소
    // 더 긴 공통 시퀀스를 찾았고, 최소 재사용 길이(params.n_reuse)를 만족하거나
    // Target 히스토리 전체가 Draft 컨텍스트에 맞는 경우, 재사용 정보 업데이트
    if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
        //printf("cur: %d\n", cur+ids_size);
        reuse_i = i;     // 재사용 시작 인덱스 (draft 히스토리 기준)
        reuse_n = cur; // 재사용 길이
    }
}

LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt.size());
//printf("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt.size());

llama_tokens result; // 생성된 draft 토큰들을 저장할 벡터
result.reserve(params.n_draft); // 미리 메모리 할당

// 3. 재사용 결과 처리 및 Draft 모델 상태 조정
if (reuse_n == 0) { // 재사용할 부분이 전혀 없을 경우
    llama_kv_cache_clear(ctx); // Draft 모델의 KV 캐시 전체 삭제
    prompt.clear();           // Draft 모델의 내부 히스토리 초기화
} else { // 재사용할 부분이 있을 경우
    // 특별 케이스: 이전 draft가 (너무 짧아서) 버려졌지만, Target 모델이 id_last까지는 동의한 경우.
    //            Draft 모델은 이미 id_last 이후 토큰들을 계산했을 수 있음.
    if (reuse_i + reuse_n < (int) prompt.size() && prompt[reuse_i + reuse_n] == id_last) {
        // 이 경우, Draft 모델을 다시 실행할 필요 없이 이전에 계산했던 결과(prompt 벡터의 뒷부분)를 재사용.
        for (int i = reuse_i + reuse_n + 1; i < (int) prompt.size(); ++i) {
            result.push_back(prompt[i]); // 이전 결과 토큰 추가
            if (params.n_draft <= (int) result.size()) break; // 목표 개수만큼 채우면 종료
        }
        return result; // 미리 계산된 결과 반환
    }

    // 일반적인 KV 캐시 재사용 처리: 불필요한 부분 제거
    if (reuse_i > 0) { // 재사용 부분이 Draft 히스토리 시작 부분이 아니라면
        // KV 캐시에서 재사용 시작 이전 부분(0 ~ reuse_i) 제거 및 인덱스 조정
        llama_kv_cache_seq_rm (ctx, 0, 0, reuse_i);
        llama_kv_cache_seq_add(ctx, 0, reuse_i, -1, -reuse_i); // 시퀀스 위치 조정
        // Draft 내부 히스토리 벡터에서도 해당 부분 제거
        prompt.erase(prompt.begin(), prompt.begin() + reuse_i);
    }

    if (reuse_n < (int) prompt.size()) { // 재사용 부분이 Draft 히스토리 전체가 아니라면
        // KV 캐시에서 재사용 부분 이후(reuse_n ~ end) 제거
        int prompt_size = prompt.size();
        llama_kv_cache_seq_rm (ctx, 0, reuse_n, -1);
        // Draft 내부 히스토리 벡터에서도 해당 부분 제거
        prompt.erase(prompt.begin() + reuse_n, prompt.end());
        //printf("prompt erased %d~%d\n", prompt.size() + 1, prompt_size);
    }
    // 결과적으로 Draft 모델의 KV 캐시와 내부 히스토리(prompt)는 정확히 reuse_n개의 토큰 상태만 가짐.
}

// 4. 새로운 토큰 처리 (Target 히스토리 중 재사용되지 않은 부분)
common_batch_clear(batch); // Draft 배치 초기화

// Target 히스토리(prompt_tgt)에서 재사용된 부분(i_start + reuse_n) 이후의 토큰들을 처리
//printf("%d - %d - %d", i_start, reuse_n, prompt_tgt.size());
if (prompt.size() == 0) {
    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        // 이 토큰들을 Draft 배치에 추가 (위치는 상대적 인덱스 사용)
        //printf("draft 배치에 추가\n");
        //printf("\n\nreused token index: %d\n\n", i);
        common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false); // 로짓 필요 없음 (false)
        // const std::string token_str = common_token_to_piece(ctx_tgt, prompt_tgt[i]);
        // // 컬러 출력 처리 (옵션)
        // LOG("\n\nReuse Part:  \u001b[%dm%s\u001b[37m\n\n", (36 - 0 % 6), token_str.c_str());
        // fflush(stdout); // 즉시 출력되도록 버퍼 비우기
        // Draft 내부 히스토리에도 추가
        prompt.push_back(prompt_tgt[i]);
    }
}
else {
    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        // 이 토큰들을 Draft 배치에 추가 (위치는 상대적 인덱스 사용)
        //printf("draft 배치에 추가\n");
        //printf("\n\nreused token index: %d\n\n", i);
        common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false); // 로짓 필요 없음 (false)
        // const std::string token_str = common_token_to_piece(ctx_tgt, prompt_tgt[i]);
        // // 컬러 출력 처리 (옵션)
        // LOG("\n\nReuse Part:  \u001b[%dm%s\u001b[37m\n\n", (36 - 0 % 6), token_str.c_str());
        // fflush(stdout); // 즉시 출력되도록 버퍼 비우기
        // Draft 내부 히스토리에도 추가
        prompt.push_back(prompt_tgt[i]);
    }   
}

// 만약 처리할 새로운 토큰들이 있었다면 (일반적으로 드문 경우)
if (batch.n_tokens > 0) {
    //printf("llama_decode_eagle 실행됨, batch.n_tokens: %d, initial_hidden_state.size(): %d\n", batch.n_tokens, initial_hidden_state.size());
    // Draft 모델(ctx)을 실행하여 이 토큰들에 대한 KV 캐시 업데이트
    //printf("\nllama_decode_eagle, batch.n_tokens: %d\n", batch.n_tokens);
    llama_decode_eagle(ctx, batch, ctx_tgt, initial_hidden_state.data(), initial_hidden_state.size());
}

// 5. 마지막 수락 토큰(id_last) 처리
const llama_pos n_past = prompt.size(); // 현재 Draft 히스토리 길이 (KV 캐시 위치)
LOG_DBG("%s: n_past = %d\n", __func__, n_past);

common_batch_clear(batch); // 배치 초기화
// Target에서 온 마지막 토큰(id_last)을 Draft 배치에 추가 (이 위치의 로짓이 필요함, true)
common_batch_add (batch, id_last, n_past, { 0 }, true);
// const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
// // 컬러 출력 처리 (옵션)
// LOG("\n\nid_last in draft phase:  \u001b[%dm%s\u001b[37m\n\n", (36 - 0 % 6), token_str.c_str());
// fflush(stdout); // 즉시 출력되도록 버퍼 비우기
// Draft 내부 히스토리에도 추가
prompt.push_back(id_last);
//printf("여긴 진입하냐..\n");
// Draft 모델(ctx)을 실행하여 id_last를 처리하고 다음 토큰 예측을 위한 로짓(logits) 계산
//printf("draft1\n");
llama_decode_draft(ctx, batch, ctx_tgt);
//printf("여긴 진입하냐..2\n");
// 6. Draft 토큰 생성 (샘플링 루프)
common_sampler_reset(smpl); // Draft 샘플러 상태 초기화

// 목표 개수(params.n_draft)만큼 Draft 토큰 생성 시도
for (int i = 0; i < params.n_draft; ++i) {
    common_batch_clear(batch); // 다음 디코딩을 위해 배치 초기화
    // Draft 샘플러(smpl)를 사용하여 Draft 컨텍스트(ctx)의 마지막 로짓에서 다음 토큰 샘플링
    common_sampler_sample(smpl, ctx, -1, true); // grammar first (true)

    // 샘플링된 후보 토큰 및 확률 가져오기
    const auto * cur_p = common_sampler_get_candidates(smpl);

    // 디버깅: 상위 후보 토큰 정보 출력
    for (int k = 0; k < std::min(1, (int) cur_p->size); ++k) {
        LOG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                  k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx, cur_p->data[k].id).c_str());
    }

    // 가장 확률 높은 토큰 선택
    const llama_token id = cur_p->data[0].id;

    // 샘플러에게 선택된 토큰 알림 (내부 상태 업데이트용)
    common_sampler_accept(smpl, id, true);

    // 결과 벡터에 생성된 draft 토큰 추가
    result.push_back(id);

    // 목표 개수 도달 시 종료
    if (params.n_draft <= (int) result.size()) {
        break;
    }

    // 생성된 토큰의 확률이 너무 낮으면(params.p_min 미만) 더 이상 생성 중단 (신뢰도 부족)
    if (cur_p->data[0].p < params.p_min) {
        //printf("\n\n%d Terminate Draft Sequence\n\n", i);
        //break;
    }

    // 다음 토큰 생성을 위해 방금 생성한 토큰(id)을 다시 Draft 배치에 추가
    common_batch_add(batch, id, n_past + i + 1, { 0 }, true); // 로짓 필요 (true)

    //printf("draft2\n");
    // Draft 모델을 다시 실행하여 방금 추가된 토큰을 처리하고 다음 위치의 로짓 계산
    llama_decode_draft(ctx, batch, ctx_tgt);

    // Draft 내부 히스토리에도 생성된 토큰 추가
    prompt.push_back(id);
}

// 7. 생성된 Draft 토큰 시퀀스 반환
return result;
}

// llama_tokens는 std::vector<llama_token>의 별칭(alias)일 가능성이 높음
llama_tokens target_model_initialize(
    struct common_speculative * spec, // Draft 모델의 상태 및 헬퍼 객체 포인터 (주석 정확)
    struct common_speculative_params params, // Speculative decoding 파라미터 (주석 수정 필요: Target 모델 구조체가 아님)
    const llama_tokens & prompt_tgt, // Target 모델이 처리한 토큰 히스토리 (주석 수정 필요: 단순 입력 프롬프트 아님)
    llama_token id_last) { // Target 모델이 마지막으로 '수락(accept)'한 토큰 ID

// 1. Speculator 객체(spec) 내부 멤버들에 대한 참조(reference) 설정
auto & batch  = spec->batch; // Draft 모델용 배치(batch) 객체
auto & ctx    = spec->ctx;   // Draft 모델의 llama_context
auto & smpl   = spec->smpl;  // Draft 모델용 샘플러(sampler)
auto & prompt = spec->prompt;// Draft 모델이 자체적으로 추적하는 내부 토큰 히스토리

// 2. KV 캐시 재사용(Reuse) 로직: 이전 스텝과의 연관성 찾기
int reuse_i = 0; // Draft 모델 내부 히스토리(prompt)에서 재사용 시작 위치
int reuse_n = 0; // 재사용할 토큰의 개수 (가장 긴 공통 접두사 길이)

// Draft 컨텍스트에서 draft 토큰을 생성할 공간을 제외한 최대 히스토리 길이 계산
const int n_ctx = llama_n_ctx(ctx) - params.n_draft;
// Target 모델 히스토리(prompt_tgt)에서 비교를 시작할 위치 계산
const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

// Draft 모델의 이전 내부 히스토리(prompt)와 Target 모델의 최신 히스토리(prompt_tgt)를 비교하여
// 재사용 가능한 가장 긴 공통 시퀀스(prefix)를 찾는다. 이는 Draft 모델의 KV 캐시를 최대한 활용하기 위함.
for (int i = 0; i < (int) prompt.size(); ++i) { // Draft 내부 히스토리 순회
    int cur = 0; // 현재 일치하는 길이
    // Target 히스토리의 끝부분과 Draft 히스토리의 부분을 비교
    while (i_start + cur < (int) prompt_tgt.size() &&
           i       + cur < (int) prompt.size() &&
           prompt_tgt[i_start + cur] == prompt[i + cur]) {
        cur++; // 일치하면 길이 증가
    }

    // 더 긴 공통 시퀀스를 찾았고, 최소 재사용 길이(params.n_reuse)를 만족하거나
    // Target 히스토리 전체가 Draft 컨텍스트에 맞는 경우, 재사용 정보 업데이트
    if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
        reuse_i = i;     // 재사용 시작 인덱스 (draft 히스토리 기준)
        reuse_n = cur; // 재사용 길이
    }
}

LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt.size());

llama_tokens result; // 생성된 draft 토큰들을 저장할 벡터
result.reserve(params.n_draft); // 미리 메모리 할당

// 3. 재사용 결과 처리 및 Draft 모델 상태 조정
if (reuse_n == 0) { // 재사용할 부분이 전혀 없을 경우
    llama_kv_cache_clear(ctx); // Draft 모델의 KV 캐시 전체 삭제
    prompt.clear();           // Draft 모델의 내부 히스토리 초기화
} else { // 재사용할 부분이 있을 경우
    // 특별 케이스: 이전 draft가 (너무 짧아서) 버려졌지만, Target 모델이 id_last까지는 동의한 경우.
    //            Draft 모델은 이미 id_last 이후 토큰들을 계산했을 수 있음.
    if (reuse_i + reuse_n < (int) prompt.size() && prompt[reuse_i + reuse_n] == id_last) {
        // 이 경우, Draft 모델을 다시 실행할 필요 없이 이전에 계산했던 결과(prompt 벡터의 뒷부분)를 재사용.
        for (int i = reuse_i + reuse_n + 1; i < (int) prompt.size(); ++i) {
            result.push_back(prompt[i]); // 이전 결과 토큰 추가
            if (params.n_draft <= (int) result.size()) break; // 목표 개수만큼 채우면 종료
        }
        return result; // 미리 계산된 결과 반환
    }

    // 일반적인 KV 캐시 재사용 처리: 불필요한 부분 제거
    if (reuse_i > 0) { // 재사용 부분이 Draft 히스토리 시작 부분이 아니라면
        // KV 캐시에서 재사용 시작 이전 부분(0 ~ reuse_i) 제거 및 인덱스 조정
        llama_kv_cache_seq_rm (ctx, 0, 0, reuse_i);
        llama_kv_cache_seq_add(ctx, 0, reuse_i, -1, -reuse_i); // 시퀀스 위치 조정
        // Draft 내부 히스토리 벡터에서도 해당 부분 제거
        prompt.erase(prompt.begin(), prompt.begin() + reuse_i);
    }

    if (reuse_n < (int) prompt.size()) { // 재사용 부분이 Draft 히스토리 전체가 아니라면
        // KV 캐시에서 재사용 부분 이후(reuse_n ~ end) 제거
        llama_kv_cache_seq_rm (ctx, 0, reuse_n, -1);
        // Draft 내부 히스토리 벡터에서도 해당 부분 제거
        prompt.erase(prompt.begin() + reuse_n, prompt.end());
    }
    // 결과적으로 Draft 모델의 KV 캐시와 내부 히스토리(prompt)는 정확히 reuse_n개의 토큰 상태만 가짐.
}

// 4. 새로운 토큰 처리 (Target 히스토리 중 재사용되지 않은 부분)
common_batch_clear(batch); // Draft 배치 초기화

// Target 히스토리(prompt_tgt)에서 재사용된 부분(i_start + reuse_n) 이후의 토큰들을 처리
for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
    // 이 토큰들을 Draft 배치에 추가 (위치는 상대적 인덱스 사용)
    common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false); // 로짓 필요 없음 (false)
    // Draft 내부 히스토리에도 추가
    prompt.push_back(prompt_tgt[i]);
}

// 만약 처리할 새로운 토큰들이 있었다면 (일반적으로 드문 경우)
if (batch.n_tokens > 0) {
    // Draft 모델(ctx)을 실행하여 이 토큰들에 대한 KV 캐시 업데이트
    llama_decode(ctx, batch);
}

// 5. 마지막 수락 토큰(id_last) 처리
const llama_pos n_past = prompt.size(); // 현재 Draft 히스토리 길이 (KV 캐시 위치)
LOG_DBG("%s: n_past = %d\n", __func__, n_past);

common_batch_clear(batch); // 배치 초기화
// Target에서 온 마지막 토큰(id_last)을 Draft 배치에 추가 (이 위치의 로짓이 필요함, true)
common_batch_add (batch, id_last, n_past, { 0 }, true);
// Draft 내부 히스토리에도 추가
prompt.push_back(id_last);

// Draft 모델(ctx)을 실행하여 id_last를 처리하고 다음 토큰 예측을 위한 로짓(logits) 계산
llama_decode(ctx, batch);

// 6. Draft 토큰 생성 (샘플링 루프)
common_sampler_reset(smpl); // Draft 샘플러 상태 초기화

// 목표 개수(params.n_draft)만큼 Draft 토큰 생성 시도
for (int i = 0; i < params.n_draft; ++i) {
    common_batch_clear(batch); // 다음 디코딩을 위해 배치 초기화

    // Draft 샘플러(smpl)를 사용하여 Draft 컨텍스트(ctx)의 마지막 로짓에서 다음 토큰 샘플링
    common_sampler_sample(smpl, ctx, 0, true); // grammar first (true)

    // 샘플링된 후보 토큰 및 확률 가져오기
    const auto * cur_p = common_sampler_get_candidates(smpl);

    // 디버깅: 상위 후보 토큰 정보 출력
    for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
        LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                  k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx, cur_p->data[k].id).c_str());
    }

    // 가장 확률 높은 토큰 선택
    const llama_token id = cur_p->data[0].id;

    // 샘플러에게 선택된 토큰 알림 (내부 상태 업데이트용)
    common_sampler_accept(smpl, id, true);

    // 결과 벡터에 생성된 draft 토큰 추가
    result.push_back(id);

    // 목표 개수 도달 시 종료
    if (params.n_draft <= (int) result.size()) {
        break;
    }

    // 생성된 토큰의 확률이 너무 낮으면(params.p_min 미만) 더 이상 생성 중단 (신뢰도 부족)
    if (cur_p->data[0].p < params.p_min) {
        break;
    }

    // 다음 토큰 생성을 위해 방금 생성한 토큰(id)을 다시 Draft 배치에 추가
    common_batch_add(batch, id, n_past + i + 1, { 0 }, true); // 로짓 필요 (true)

    // Draft 모델을 다시 실행하여 방금 추가된 토큰을 처리하고 다음 위치의 로짓 계산
    llama_decode(ctx, batch);

    // Draft 내부 히스토리에도 생성된 토큰 추가
    prompt.push_back(id);
}

// 7. 생성된 Draft 토큰 시퀀스 반환
return result;
}
