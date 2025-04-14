#include "arg.h" // 프로그램 인자 관련 헤더
#include "common.h" // 공통 유틸리티 및 함수 헤더
#include "sampling.h" // 토큰 샘플링 관련 헤더
#include "log.h" // 로깅 관련 헤더
#include "llama.h" // llama.cpp 핵심 라이브러리 헤더

#include <algorithm> // 표준 알고리즘 라이브러리 (정렬 등)
#include <cstdio> // 표준 입출력 라이브러리 (C 스타일)
#include <cstring> // 문자열 처리 라이브러리 (C 스타일)
#include <random> // 난수 생성 라이브러리
#include <set> // 표준 집합 라이브러리
#include <string> // 표준 문자열 라이브러리
#include <vector> // 표준 동적 배열 라이브러리

// 추측성 디코딩 시 허용되는 최대 어휘 크기 차이
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
// 어휘 일치 검사를 시작할 토큰 ID (특수 토큰 이후부터 검사)
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// 드래프트 시퀀스(가지)의 상태를 저장하는 구조체
struct seq_draft {
    bool active   = false; // 현재 시퀀스가 활성 상태인지 여부
    bool drafting = false; // 현재 시퀀스가 드래프팅(토큰 생성) 중인지 여부
    bool skip     = false; // 현재 시퀀스를 이번 스텝에서 건너뛸지 여부 (트리 분기 시 사용)

    int i_batch_dft = 0; // 드래프트 모델 배치에서의 마지막 토큰 인덱스
    std::vector<int> i_batch_tgt; // 타겟 모델 배치에서의 토큰 인덱스 목록

    std::vector<llama_token> tokens; // 이 시퀀스에서 드래프트된 토큰 목록
    std::vector<std::vector<llama_token_data>> dists; // 각 드래프트된 토큰 위치에서의 확률 분포 목록

    struct common_sampler * smpl = nullptr; // 이 시퀀스에 사용될 샘플러 상태
};

int main(int argc, char ** argv) {
    common_params params; // 공통 파라미터 구조체

    // 온도가 0 이하일 때도 후보 토큰 확률을 얻기 위해 필요함
    params.sampling.n_probs = 128;

    // 명령줄 인자 파싱, 실패 시 종료
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    // 생성할 토큰 수(-1은 무한)가 유효한지 확인
    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // 공통 초기화 수행
    common_init();

    // 드래프트 모델 경로가 지정되지 않았으면 오류 발생 및 종료
    if (params.speculative.model.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // 최대 병렬 드래프팅 시퀀스 수 (트리 브랜치 수)
    const int n_seq_dft = params.n_parallel;

    // 드래프트 브랜치를 분할할 확률 임계값 (n_seq_dft > 1 일 때만 사용)
    const float p_draft_split = params.speculative.p_split;

    // 난수 생성기 초기화 (시드가 지정되지 않으면 random_device 사용)
    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist; // 0.0 ~ 1.0 사이 균등 분포 난수 생성기

    // llama.cpp 백엔드 및 NUMA 초기화
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL; // 타겟 모델 포인터
    llama_model * model_dft = NULL; // 드래프트 모델 포인터

    llama_context * ctx_tgt = NULL; // 타겟 모델 컨텍스트 포인터
    llama_context * ctx_dft = NULL; // 드래프트 모델 컨텍스트 포인터

    // 타겟 모델 로드
    common_init_result llama_init_tgt = common_init_from_params(params); // common 함수 사용하여 로드

    model_tgt = llama_init_tgt.model.get(); // 모델 포인터 가져오기
    ctx_tgt   = llama_init_tgt.context.get(); // 컨텍스트 포인터 가져오기

    // 드래프트 모델 로드를 위해 파라미터 설정 변경
    params.devices = params.speculative.devices; // 드래프트 모델용 장치 설정
    params.model = params.speculative.model; // 드래프트 모델 경로 설정
    params.n_gpu_layers = params.speculative.n_gpu_layers; // 드래프트 모델용 GPU 레이어 수 설정
    if (params.speculative.cpuparams.n_threads > 0) { // 드래프트 모델용 CPU 스레드 수 설정 (지정된 경우)
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads; // 드래프트 모델용 배치 처리 스레드 수 설정
    // 드래프트 모델 로드
    common_init_result llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft.model.get(); // 드래프트 모델 포인터 가져오기
    ctx_dft   = llama_init_dft.context.get(); // 드래프트 컨텍스트 포인터 가져오기

    // 타겟 모델과 드래프트 모델의 어휘(vocabulary) 가져오기
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    // 어휘 타입 확인 및 비교 (SPM, BPE 등)
    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    // 어휘 타입이 다르면 추측성 디코딩 사용 불가, 오류 발생 및 종료
    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    // 특수 토큰 (BOS, EOS) 및 관련 설정이 일치하는지 확인
    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) || // BOS 토큰 자동 추가 설정 비교
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) || // EOS 토큰 자동 추가 설정 비교
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||                 // BOS 토큰 ID 비교
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)                    // EOS 토큰 ID 비교
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    // 어휘 크기 및 내용 비교
    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt); // 타겟 모델 어휘 크기
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft); // 드래프트 모델 어휘 크기
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft        // 어휘 크기 차이 계산
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        // 어휘 크기 차이가 허용 범위를 넘으면 오류 발생 및 종료
        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        // 어휘의 각 토큰 문자열이 일치하는지 검사 (특수 토큰 이후부터)
        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i); // 타겟 모델의 i번째 토큰 문자열
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i); // 드래프트 모델의 i번째 토큰 문자열
            // 문자열이 다르면 오류 발생 및 종료
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(), // common 함수 이용해 사람이 읽기 좋은 형태로 변환
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }


    // 프롬프트를 토큰화
    std::vector<llama_token> inp; // 입력 토큰들을 저장할 벡터
    inp = common_tokenize(ctx_tgt, params.prompt, true, true); // common 함수 사용, BOS 토큰 추가

    const int max_context_size      = llama_n_ctx(ctx_tgt); // 타겟 모델의 컨텍스트 크기
    const int max_tokens_list_size  = max_context_size - 4; // 약간의 여유 공간 확보

    // 프롬프트 길이가 너무 길면 오류 발생 및 종료
    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n"); // 로그 가독성을 위한 빈 줄 출력

    // 토큰화된 프롬프트를 문자열로 변환하여 출력
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size(); // 입력 토큰 수

    const auto t_enc_start = ggml_time_us(); // 인코딩 시작 시간 측정

    // 프롬프트를 두 모델 모두에서 평가(처리)하여 초기 상태 동기화
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1)); // 타겟 모델: 마지막 토큰 제외하고 처리
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1)); // 타겟 모델: 마지막 토큰만 따로 처리 (다음 토큰 예측 준비)
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));     // 드래프트 모델: 전체 프롬프트 처리

    const auto t_enc_end = ggml_time_us(); // 인코딩 종료 시간 측정

    // 두 모델의 어휘 크기가 같은지 확인 (위에서 이미 검사했지만, 추가 확인)
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // 한 번에 드래프트할 토큰 수 설정
    int n_draft = params.speculative.n_max;

    int n_predict = 0; // 현재까지 생성된(예측된) 총 토큰 수
    int n_drafted = 0; // 현재까지 드래프트 모델이 생성한 총 토큰 수
    int n_accept  = 0; // 현재까지 타겟 모델에 의해 수락된 드래프트 토큰 수

    int n_past_tgt = inp.size(); // 타겟 모델의 KV 캐시에 저장된 토큰 수 (초기값: 프롬프트 길이)
    int n_past_dft = inp.size(); // 드래프트 모델의 KV 캐시에 저장된 토큰 수 (초기값: 프롬프트 길이)

    // 생성 종료 조건 확인용 플래그 (EOS 토큰 생성 여부)
    bool has_eos = false;

    // 타겟 모델 샘플링 컨텍스트 초기화 (llama_context 내부 샘플링 인스턴스 재사용)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // 드래프트 시퀀스 데이터 구조체 벡터 초기화 (병렬 처리 수만큼)
    std::vector<seq_draft> drafts(n_seq_dft);

    // 각 드래프트 시퀀스마다 샘플러 초기화
    for (int s = 0; s < n_seq_dft; ++s) {
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    // 드래프트 및 타겟 모델용 배치 초기화
    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1); // 드래프트 배치는 시퀀스 1개 처리 가능
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft); // 타겟 배치는 여러 시퀀스 처리 가능

    const auto t_dec_start = ggml_time_us(); // 디코딩 시작 시간 측정

    // 프롬프트의 마지막 토큰으로부터 샘플링 시작 준비
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0; // 첫 번째 시퀀스, 첫 번째 토큰

    // 메인 생성 루프
    while (true) {
        std::set<int> active_seqs = {}; // 현재 활성화된 드래프트 시퀀스 인덱스 저장용 집합

        // 디버깅: 현재 활성 드래프트 시퀀스들 출력
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) { // 비활성 시퀀스는 건너뜀
                continue;
            }

            active_seqs.insert(s); // 활성 시퀀스 집합에 추가
            const auto & tokens = drafts[s].tokens; // 해당 시퀀스의 드래프트된 토큰들

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str()); // 토큰들을 문자열로 변환하여 로그 출력
        }

        int i_dft  = 0; // 현재 검증 중인 드래프트 토큰의 인덱스 (각 시퀀스 내)
        int s_keep = 0; // 현재까지 유지되고 있는 (가장 유력한) 드래프트 시퀀스 인덱스

        llama_token token_id; // 최종적으로 선택(수락 또는 샘플링)된 토큰 ID
        std::string token_str; // 선택된 토큰의 문자열 표현

        // --- 검증(Verification) 단계 ---
        // 드래프트된 토큰이 수락되지 않거나, 드래프트된 토큰을 모두 소진할 때까지 반복
        while (true) {

            // 타겟 토큰이 드래프트된 토큰과 일치하는지 확인
            // 확률적 샘플링 (temp > 0) 또는 탐욕적(greedy) 샘플링 (temp <= 0) 방식 사용
            {
                bool accept = false; // 현재 단계에서 드래프트 토큰이 수락되었는지 여부
                if (params.sampling.temp > 0) {
                    // 확률적 검증 (Stochastic Verification)
                    // 1. 타겟 모델 샘플러로 현재 위치(i_dft)에서의 확률 분포(dist_tgt) 계산
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true); // true: 확률 분포만 계산

                    auto & dist_tgt = *common_sampler_get_candidates(smpl); // 계산된 타겟 확률 분포 가져오기

                    float p_tgt = 0.0f; // 특정 토큰에 대한 타겟 모델 확률
                    float p_dft = 0.0f; // 특정 토큰에 대한 드래프트 모델 확률

                    // 2. 활성 드래프트 시퀀스 중 하나를 무작위로 선택하여 검증
                    while (active_seqs.size() > 0) {
                        // 활성 시퀀스 중 하나를 무작위 선택
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng)); // 선택된 시퀀스 인덱스
                        // 현재 검증 위치(i_dft)에 해당하는 드래프트 토큰이 없으면 해당 시퀀스 비활성화
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        // 이미 다른 시퀀스에서 토큰이 수락되었다면, 다른 토큰을 가진 시퀀스는 비활성화
                        if (accept) {
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue; // 다음 시퀀스 검증으로 넘어감
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng); // 0.0 ~ 1.0 사이 난수 생성
                        // 드래프트 모델의 확률 분포 가져오기
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        // GGML_ASSERT(dist_tgt.size <= dist_dft.size); // 어휘 크기가 비슷해야 함

                        // 선택된 드래프트 토큰에 대한 타겟(p_tgt) 및 드래프트(p_dft) 모델의 확률 찾기
                        p_tgt = 0.0f; // 확률 초기화
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        p_dft = 0.0f; // 확률 초기화
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);

                        // 3. 확률 기반 수락/거절 결정 (Speculative Sampling 핵심 로직)
                        // r <= p_target / p_draft 인 경우 수락
                        if (r <= p_tgt / p_dft) {
                            s_keep = s; // 이 시퀀스를 유지할 시퀀스로 설정
                            accept = true; // 수락 플래그 설정
                            token_id = drafts[s].tokens[i_dft]; // 수락된 토큰 ID 저장
                            token_str = common_token_to_piece(ctx_tgt, token_id); // 토큰 문자열 변환
                            common_sampler_accept(smpl, token_id, true); // 타겟 샘플러 상태 업데이트

                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break; // 수락했으므로 다른 시퀀스 검증 중단
                        } else {
                            // 거절된 경우
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false; // 해당 시퀀스 비활성화

                            // 잔차 샘플링(Residual Sampling)을 위한 확률 분포 조정 (거절된 토큰의 확률 제거 효과)
                            GGML_ASSERT(dist_tgt.sorted); // 정렬되어 있다고 가정
                            GGML_ASSERT(dist_dft.sorted);

                            // ID 기준으로 확률 분포 정렬 (효율적인 비교를 위해)
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f; // 조정된 확률의 합

                            // p_target = max(0, p_target - p_draft) 계산
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }
                                sum_probs += dist_tgt.data[i].p;
                            }

                            // 확률 정규화
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // 다시 확률(p) 기준으로 내림차순 정렬
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        // 현재 검증한 시퀀스를 활성 목록에서 제거
                        active_seqs.erase(s);
                        // 같은 토큰을 가진 다른 시퀀스들의 상태 동기화
                        for(int i = 0; i < n_seq_dft; i++) {
                            if (i == s) continue;
                            if (drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                drafts[i].active = drafts[i].active && accept; // 수락되었으면 활성 유지, 아니면 비활성
                                if (!drafts[i].active) {
                                    active_seqs.erase(i); // 비활성화되면 활성 목록에서도 제거
                                }
                            }
                        }
                    } // end while (active_seqs.size() > 0)

                    // 4. 모든 드래프트가 거절된 경우: 조정된 타겟 분포(잔차 분포)에서 직접 샘플링
                    if (!accept) {
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        // std::discrete_distribution을 사용하여 잔차 분포에서 샘플링
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }
                        std::discrete_distribution<> dist(probs.begin(), probs.end());
                        const int idx = dist(rng); // 샘플링된 인덱스

                        token_id = dist_tgt.data[idx].id; // 최종 토큰 ID
                        common_sampler_accept(smpl, token_id, true); // 타겟 샘플러 상태 업데이트
                        token_str = common_token_to_piece(ctx_tgt, token_id); // 토큰 문자열 변환
                    }
                } else {
                    // 탐욕적 검증 (Greedy Verification) - temp == 0 인 경우
                    // 1. 타겟 모델에서 가장 확률 높은 토큰(token_id) 샘플링
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]); // 샘플링 (temp=0이므로 가장 확률 높은 것)

                    common_sampler_accept(smpl, token_id, true); // 샘플러 상태 업데이트

                    token_str = common_token_to_piece(ctx_tgt, token_id); // 토큰 문자열 변환

                    // 2. 샘플링된 타겟 토큰(token_id)과 현재 위치(i_dft)의 드래프트 토큰 비교
                    accept = false; // 일단 수락 안됨으로 초기화
                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) { // 비활성 시퀀스 건너뜀
                            continue;
                        }

                        // 드래프트 토큰이 존재하고 타겟 토큰과 일치하면 수락
                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s; // 해당 시퀀스를 유지할 시퀀스로 설정
                            accept = true; // 수락
                        } else {
                            drafts[s].active = false; // 일치하지 않으면 해당 시퀀스 비활성화
                        }
                    }
                } // end if (params.sampling.temp > 0) ... else ...

                // 5. 생성 종료 조건 확인 (EOS 토큰)
                if (llama_vocab_is_eog(vocab_tgt, token_id)) { // End Of Generation 토큰인지 확인
                    has_eos = true;
                }
                ++n_predict; // 예측된 토큰 수 증가

                // 6. 검증 결과 처리
                if (accept) {
                    // 드래프트 토큰이 수락된 경우
                    ++n_accept; // 수락된 토큰 수 증가
                    ++n_past_tgt; // 타겟 KV 캐시 길이 증가
                    ++n_past_dft; // 드래프트 KV 캐시 길이 증가
                    ++i_dft; // 다음 드래프트 토큰 검증 위치로 이동
                    // 수락된 토큰 출력 (색상 옵션 사용 시 시퀀스별 색상 적용)
                    if (params.use_color) {
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue; // 검증 루프 계속 (다음 드래프트 토큰 검증)
                } else {
                    // 드래프트 토큰이 거절된 경우 (또는 드래프트 소진)
                    // 최종 선택된 토큰(타겟 모델에서 샘플링된 것) 출력
                    LOG("%s", token_str.c_str());
                    break; // 검증 루프 종료 -> 상태 업데이트 및 드래프팅 단계로 이동
                }
            } // end verification logic block
        } // end inner verification while loop

        // --- 상태 업데이트 및 드래프팅(Drafting) 단계 ---
        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // KV 캐시 정리: 유지하기로 결정된 시퀀스(s_keep)의 상태를 기준으로 동기화
            // TODO: 단순화 필요
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                // 드래프트 모델 KV 캐시 정리: s_keep 시퀀스를 0번으로 복사하고 나머지는 삭제
                llama_kv_cache_seq_keep(ctx_dft, s_keep);
                llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_dft, 0);

                // 타겟 모델 KV 캐시 정리: s_keep 시퀀스에서 거절된 부분(n_past_tgt 이후) 제거 후 0번으로 복사
                llama_kv_cache_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                llama_kv_cache_seq_keep(ctx_tgt, s_keep);
                llama_kv_cache_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
            }

            // 모든 드래프트 시퀀스 상태 초기화
            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // 방금 최종 확정된 토큰(token_id)을 다음 드래프팅의 시작점으로 설정 (0번 시퀀스에 저장)
            // (주의: 이 토큰은 아래 드래프팅 단계 후 검증 단계 시작 시 첫 토큰으로 사용되고 바로 삭제됨)
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>()); // 빈 확률 분포 추가
            drafts[0].i_batch_tgt.push_back(0); // 배치 인덱스 추가

            // 드래프트 모델에 이 확정된 토큰을 입력하여 다음 드래프팅 준비
            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true); // 시퀀스 0에 추가

            llama_kv_cache_seq_rm(ctx_dft, 0, n_past_dft, -1); // 드래프트 KV 캐시에서 이전 상태 제거
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            llama_decode(ctx_dft, batch_dft); // 드래프트 모델 실행

            ++n_past_dft; // 드래프트 KV 캐시 길이 증가
        }

        // 메인 루프 종료 조건 확인 (예측 토큰 수 도달 또는 EOS 생성)
        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }

        // 타겟 샘플러 상태를 복제하여 다음 드래프팅의 시작 상태로 사용 (0번 시퀀스)
        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl); // 이전 샘플러 해제
        }
        drafts[0].smpl = common_sampler_clone(smpl); // 타겟 샘플러 상태 복제

        // 드래프팅 단계 초기화
        int n_seq_cur  = 1; // 현재 활성 드래프팅 시퀀스 수 (초기값 1)
        int n_past_cur = n_past_dft; // 현재 드래프팅 시작 시점의 KV 캐시 길이

        // 모든 드래프트 시퀀스 상태 초기화 (0번만 활성화)
        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active       = true; // 0번 시퀀스 활성화
        drafts[0].drafting     = true; // 0번 시퀀스 드래프팅 시작
        drafts[0].i_batch_dft  = 0; // 0번 시퀀스의 드래프트 배치 인덱스 초기화

        // 타겟 모델 검증용 배치 초기화 및 시작 토큰 추가
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // --- 드래프팅 루프 (n_draft 만큼 토큰 생성 시도) ---
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0; // 드래프트 배치 초기화

            // 스킵 플래그 초기화
            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            // 활성 드래프팅 시퀀스 순회
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) { // 드래프팅 중이 아니거나 스킵 플래그가 있으면 건너뜀
                    continue;
                }

                // 현재 시퀀스(s)의 샘플러로 다음 토큰 후보들(cur_p) 생성
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true); // true: 확률 분포만 계산

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl); // 후보 토큰 및 확률 가져오기

                // 디버깅: 상위 후보 토큰들 로그 출력
                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s); // 현재 처리할 시퀀스 목록 (초기값: 현재 시퀀스 s)

                // --- 트리 분기 (Tree-based Sampling) 로직 ---
                // 확률이 높은 다른 후보 토큰이 있고, 여유 시퀀스가 있으면 브랜치 분할 시도
                for (int f = 1; f < 8; ++f) { // 상위 몇 개(여기선 7개) 후보까지 확인
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) { // 여유 시퀀스가 있고 분할 확률 조건 만족 시
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur); // 분기 로그

                        // 새로운 시퀀스(n_seq_cur)에 현재 시퀀스(s)의 KV 캐시 복사
                        llama_kv_cache_seq_rm(ctx_dft,     n_seq_cur, -1, -1); // 기존 내용 삭제 (혹시 모르니)
                        llama_kv_cache_seq_cp(ctx_dft, s, n_seq_cur, -1, -1); // 복사

                        // 타겟 배치에도 새로운 시퀀스 반영 (기존 s 시퀀스가 포함된 모든 토큰에 n_seq_cur 추가)
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // 새로운 시퀀스 상태 복사 및 설정
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true; // 이번 스텝에서는 샘플링 건너뜀 (다음 스텝부터 진행)

                        drafts[n_seq_cur].tokens       = drafts[s].tokens; // 현재까지 드래프트된 토큰 복사
                        drafts[n_seq_cur].dists        = drafts[s].dists; // 확률 분포 복사
                        drafts[n_seq_cur].i_batch_dft  = drafts[s].i_batch_dft; // 배치 인덱스 복사
                        drafts[n_seq_cur].i_batch_tgt  = drafts[s].i_batch_tgt; // 배치 인덱스 복사

                        // 샘플러 상태 복제
                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur); // 처리할 시퀀스 목록에 새 시퀀스 추가

                        n_seq_cur++; // 활성 시퀀스 수 증가
                    } else {
                        break; // 확률이 낮거나 여유 시퀀스 없으면 분기 중단
                    }
                } // end branch splitting loop

                // 선택된 후보 토큰(들)을 각 해당 시퀀스에 추가
                for (int is = 0; is < (int) sa.size(); ++is) { // sa: [원본 시퀀스 s, 새로 분기된 시퀀스들...]
                    const llama_token id = cur_p->data[is].id; // is=0이면 가장 확률 높은 토큰, is=1이면 두 번째 등등

                    const int s = sa[is]; // 처리할 시퀀스 인덱스

                    // 해당 시퀀스 샘플러 상태 업데이트
                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id); // 드래프트 토큰 목록에 추가
                    // 현재 단계의 확률 분포 저장 (나중에 검증 단계에서 사용)
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // 타겟 모델 검증용 배치에 토큰 추가
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens); // 배치 인덱스 저장
                    // common_batch_add는 내부적으로 같은 토큰이 이미 배치에 있으면 시퀀스 ID만 추가함
                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true); // logits=true 중요 (검증 시 필요)

                    // 다음 드래프팅 단계를 위한 드래프트 모델 배치에 토큰 추가
                    drafts[s].i_batch_dft = batch_dft.n_tokens; // 배치 인덱스 저장
                    common_batch_add(batch_dft, id, n_past_cur, { s }, true); // logits=true 중요 (샘플링 시 필요)

                    // 타겟 배치의 토큰 수가 드래프트 목표치를 넘으면 해당 시퀀스 드래프팅 중지
                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                } // end loop for adding tokens to sequences
            } // end loop for active drafting sequences

            // 이번 스텝에서 드래프트 배치에 추가된 토큰이 없으면 드래프팅 종료
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // 드래프트 모델을 실행하여 배치에 있는 토큰들의 다음 확률 분포 계산
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur; // 드래프트 KV 캐시 길이 증가 (이번 배치 길이만큼은 아님, 1 증가)
            ++n_drafted; // 드래프트된 총 토큰 수 증가

            // 타겟 배치의 토큰 수가 목표치를 넘었으면 드래프팅 종료
            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
        } // end drafting loop (for i < n_draft)

        // --- 타겟 모델 평가 ---
        // 드래프팅 단계에서 생성된 모든 후보 토큰들을 타겟 모델에서 한 번에 평가
        {
            // 타겟 KV 캐시 상태 복제 (0번 시퀀스 기준으로 나머지 시퀀스 생성)
            llama_kv_cache_seq_keep(ctx_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt); // 타겟 모델 실행 (배치 처리)
            ++n_past_tgt; // 타겟 KV 캐시 길이 증가 (이번 배치 길이만큼은 아님, 1 증가)
        }

        // 검증 단계 시작 전에, 이전 검증/샘플링 단계에서 추가했던 첫 토큰 제거
        // (이 토큰은 이미 처리/출력되었고, 다음 검증은 새로 드래프트된 토큰부터 시작해야 함)
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }
            // 시퀀스가 비어있지 않은 경우에만 제거 시도
            if (!drafts[s].tokens.empty()) {
                drafts[s].tokens.erase(drafts[s].tokens.begin());
                drafts[s].dists.erase(drafts[s].dists.begin());
            }
        }
    } // end main generation loop (while true)

    auto t_dec_end = ggml_time_us(); // 디코딩 종료 시간 측정

    LOG("\n\n"); // 로그 가독성 위한 빈 줄

    // 최종 성능 통계 출력
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);   // 드래프트 목표 토큰 수
    LOG_INF("n_predict = %d\n", n_predict); // 최종 생성된 토큰 수
    LOG_INF("n_drafted = %d\n", n_drafted); // 드래프트 모델이 생성한 총 토큰 수
    LOG_INF("n_accept  = %d\n", n_accept);  // 수락된 드래프트 토큰 수
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted); // 수락률

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: 모든 드래프트의 샘플링/그래머 타이밍 출력
    llama_perf_context_print(ctx_dft); // 드래프트 모델 성능 카운터 출력

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl); // 타겟 모델 성능 카운터 및 샘플러 타이밍 출력

    // 자원 해제
    common_sampler_free(smpl); // 타겟 샘플러 해제
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl); // 드래프트 샘플러들 해제
    }

    llama_batch_free(batch_dft); // 드래프트 배치 해제
    // llama_batch_free(batch_tgt); // 타겟 배치는 llama_context 해제 시 같이 해제될 것으로 예상 (별도 해제 함수 없음)

    llama_backend_free(); // llama.cpp 백엔드 해제

    LOG("\n\n"); // 로그 가독성 위한 빈 줄

    return 0; // 프로그램 정상 종료
}
