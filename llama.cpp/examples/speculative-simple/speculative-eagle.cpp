#include "arg.h"         // Command-line argument parsing helpers
#include "common.h"      // Common helper functions for llama.cpp examples
#include "sampling.h"    // Token sampling strategies (temperature, top-k, top-p, etc.)
#include "speculative.h" // Helper functions specifically for speculative decoding
#include "log.h"         // Logging utilities (LOG_INF, LOG_ERR, etc.)
#include "llama.h"       // Core llama.cpp library header

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

int main(int argc, char ** argv) {
    // 1. 파라미터 초기화 및 파싱
    common_params params; // 프로그램 실행에 필요한 파라미터들을 담을 구조체

    // common_params_parse: 커맨드 라인 인자(argc, argv)를 파싱하여 params 구조체를 채움
    // LLAMA_EXAMPLE_SPECULATIVE: 이 예제가 speculative decoding용임을 명시
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1; // 파싱 실패 시 종료
    }

    // 생성할 토큰 수(--n-predict) 유효성 검사
    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // 2. 기본 초기화
    common_init(); // 주석: llama 로그 파일 생성 (및 기타 공통 초기화, 예: 랜덤 시드 설정)

    // Speculative decoding에는 draft 모델 경로가 필수 (--model-draft)
    if (params.speculative.model.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // llama.cpp 백엔드 초기화 (CPU, CUDA, Metal 등)
    llama_backend_init();
    // NUMA (Non-Uniform Memory Access) 최적화 초기화 (설정된 경우)
    llama_numa_init(params.numa);

    // 3. 모델 및 컨텍스트 포인터 선언
    llama_model * model_tgt = NULL; // Target 모델 포인터
    //llama_model * model_dft = NULL; // Draft 모델 포인터 (여기서는 직접 사용 안 함)
    llama_context * ctx_tgt = NULL; // Target 모델 컨텍스트 포인터
    llama_context * ctx_dft = NULL; // Draft 모델 컨텍스트 포인터

    // 4. Target 모델 로딩
    // common_init_from_params: 파라미터(params)를 사용하여 모델 파일 로드 및 컨텍스트 생성
    // 주석: target model, context 초기화 - 정확함. 모델 파일 경로, GPU 레이어 수, 컨텍스트 크기 등 설정 적용
    common_init_result llama_init_tgt = common_init_from_params(params);

    // common_init_result 객체에서 실제 모델 및 컨텍스트 포인터 가져오기
    // 주석: llama_init_tgt 객체로부터 model/context 호출 - 정확함. (.get()은 스마트 포인터에서 원시 포인터를 얻음)
    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // 모델로부터 어휘 사전(vocabulary) 가져오기
    // 주석: model 객체로부터 vocab 호출 - 정확함.
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // 5. Draft 모델 로딩
    // Draft 모델에 적용할 파라미터들을 speculative 설정에서 가져와 params 구조체에 덮어쓰기
    // (예: --model-draft-gpu-layers 등)
    params.devices        = params.speculative.devices;
    params.model          = params.speculative.model; // Draft 모델 경로로 변경
    params.n_ctx          = params.speculative.n_ctx;
    params.n_batch        = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    params.n_gpu_layers   = params.speculative.n_gpu_layers;
    // Draft 모델용 CPU 스레드 수 설정 (별도 지정 시)
    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }
    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;

    // 수정된 파라미터(주로 draft 모델 경로)를 사용하여 draft 모델 로드 및 컨텍스트 생성
    // 주석: draft model, context 초기화 - 정확함.
    common_init_result llama_init_dft = common_init_from_params(params);

    // Draft 모델 컨텍스트 포인터 가져오기
    //model_dft = llama_init_dft.model.get(); // 모델 포인터는 직접 사용 안 함
    // 주석: llama_init_dft 객체로부터 context 호출 - 정확함.
    ctx_dft   = llama_init_dft.context.get();

    // 6. 모델 호환성 검사
    // Target 모델과 Draft 모델이 speculative decoding에 사용 가능하도록 호환되는지 확인
    // (예: 어휘 사전, 임베딩 차원 등)
    // 주석: target model과 draft model이 호환되는지 검사 - 정확함.
    if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        return 1;
    }

    // 7. 프롬프트 토큰화
    std::vector<llama_token> inp; // 토큰 ID들을 저장할 벡터
    // common_tokenize: 입력 프롬프트(params.prompt)를 토큰 ID 시퀀스로 변환
    // 내부적으로 target 컨텍스트(ctx_tgt)의 어휘 사전을 사용함.
    // true, true: 각각 BOS(문장 시작) 토큰 추가 여부, 특수 토큰 처리 여부일 가능성 높음.
    // 주석: target model과 호환되는 vocab을 내부에서 호출한 뒤 tokenize - 정확함.
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    // 프롬프트 길이가 Target 모델의 컨텍스트 크기 또는 배치 크기를 초과하는지 검사
    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));
        return 1;
    }
    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));
        return 1;
    }

    LOG("\n\n"); // 로그 가독성을 위한 개행

    // 토큰화된 프롬프트를 다시 텍스트로 변환하여 출력 (디버깅 목적)
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // 8. Speculative Decoding 파라미터 및 카운터 초기화
    int n_draft       = params.speculative.n_max;      // 매 스텝 생성할 최대 draft 토큰 수
    int n_draft_min   = params.speculative.n_min;      // 유효한 draft로 간주할 최소 토큰 수
    float p_min       = params.speculative.p_min;      // 관련 파라미터 (예: 최소 확률 등, common_speculative_params에서 사용)
    int n_predict = 0; // 총 생성된 토큰 수 (Target 모델 기준)
    int n_drafted = 0; // 총 생성 시도된 draft 토큰 수
    int n_accept  = 0; // 총 수락된 draft 토큰 수
    bool has_eos = false; // End-of-Sentence 토큰 생성 여부 플래그

    // ================================================
    // 여기까지는 표준 초기화 과정
    // Speculative decoding 관련 핵심 로직 시작
    // ================================================

    const auto t_enc_start = ggml_time_us(); // 프롬프트 처리 시간 측정 시작

    // 9. Target 모델용 샘플러 초기화
    // common_sampler_init: Target 모델(model_tgt)과 샘플링 파라미터(params.sampling)를 사용하여 샘플러 객체 생성
    // 샘플러는 생성된 로짓(logits)에서 다음 토큰을 선택하는 방식을 정의 (온도, top-k, top-p 등)
    // 주석: target model과 호환되는 sampler 객체를 생성해 반환 - 정확함.
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    std::cout <<"Model Initialized" << std::endl; // 모델 초기화 완료 메시지

    // 10. 초기 프롬프트 처리 (KV 캐시 워밍업)
    // llama_decode_init (또는 llama_decode): 프롬프트 토큰들(마지막 토큰 제외)을 Target 모델에 입력하여
    // 내부 상태(특히 KV 캐시)를 초기화/워밍업함.
    // llama_batch_get_one: 단일 시퀀스를 처리하기 위한 임시 배치 생성.
    // inp.size() - 1: 마지막 토큰은 루프 시작 시 처리하므로 제외.
    // 주석: 이 부분은 뭐하는 건지 정확히 모르겠다.. -> 초기 프롬프트를 Target 모델에 입력하여 KV 캐시를 채우는 과정입니다.
    llama_decode_init(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1)); //이 부분은 뭐하는 건지 정확히 모르겠다.. 다시한번 체크

    // 마지막 프롬프트 토큰은 따로 저장하여 루프의 첫 입력으로 사용
    llama_token id_last = inp.back();

    // 처리된 프롬프트 토큰들(마지막 제외)을 저장하는 벡터 (주로 draft 생성 시 참조 가능성)
    // 주석: 토큰화된 프롬프트에 target model의 컨텍스트 정보를 저장? -> 처리된 프롬프트 기록(마지막 제외)을 저장하는 벡터입니다.
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt)); // 메모리 예비 할당

    // 현재까지 처리된 토큰 수 (= KV 캐시 내 위치)
    int n_past = inp.size();

    // 11. Speculator 초기화
    // Speculative decoding 파라미터 설정
    struct common_speculative_params params_spec; // 주석: speculator 구조체 생성, 변수 초기화 - params_spec는 파라미터 구조체
    params_spec.n_draft = n_draft;
    params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft; // KV 캐시 재사용 관련 파라미터일 수 있음
    params_spec.p_min   = p_min;

    // common_speculative_init: Draft 모델 컨텍스트(ctx_dft)를 사용하여 speculator 헬퍼 객체 생성
    // 이 객체는 draft 토큰 생성 로직을 담당하며, 필요한 내부 상태나 버퍼를 가질 수 있음.
    // 주석: draft model의 context를 사용해서 speculative 구조체를 생성... - 정확함. (spec은 객체 포인터)
    struct common_speculative * spec = common_speculative_init(ctx_dft);

    // 12. Target 모델용 배치(Batch) 초기화
    // llama_batch_init: Target 모델에 토큰들을 전달하기 위한 재사용 가능한 배치 구조체 생성
    // llama_n_batch(ctx_tgt): 배치 크기, 0: 임베딩 입력 없음, 1: 시퀀스 ID 개수
    // 주석: target model과 관련된 연산을 위한 메모리 할당(연산을 llama_batch라는 단위로 수행?) - 정확함.
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_enc_end = ggml_time_us(); // 프롬프트 처리 시간 측정 종료
    const auto t_dec_start = ggml_time_us(); // 토큰 생성 시간 측정 시작

    // (주석 처리된 부분) Hidden State 추출 및 설정 시도 (현재는 비활성화됨)
    // 이 부분은 llama_get_embeddings_ith가 입력 임베딩을 반환하고, ctx_dft->inp_embd 직접 설정이
    // 비표준적이므로 주석 처리된 것으로 보임.

    // --- Extract hidden state after first target model run ---
    // 1. Decode the prompt in the target model
    printf("Target 모델이 Initial Token을 생성합니다.\n\n");
    common_batch_clear(batch_tgt); // 배치 초기화
    // Target에서 온 마지막 토큰(id_last)을 Draft 배치에 추가 (이 위치의 로짓이 필요함, true)
    common_batch_add (batch_tgt, id_last, n_past++, { 0 }, true);
    llama_decode_initial(ctx_tgt, batch_tgt, ctx_dft);
    
    llama_token new_token_id = common_sampler_sample(smpl, ctx_tgt, -1);
    id_last = new_token_id;
    prompt_tgt.push_back(id_last);
    //n_past++;
    n_predict++;

    // 샘플링된 토큰이 문장 끝(End of Generation) 토큰인지 확인
    if (llama_vocab_is_eog(vocab, id_last)) {
        printf(" [EOS]"); // EOS 토큰이면 표시하고 루프 종료
        return 0;
    }
    
    // 샘플링된 토큰 ID를 텍스트로 변환하여 출력
    char buf[128];
    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    if (n < 0) { // 변환 실패 시 오류
        fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
        return 1;
    }
    std::string s(buf, n);
    printf("{%s} First Token by Target Model\n\n", s.c_str()); // 출력
    fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함

    printf("여기부터 배치 처리가 시작됩니다.\n\n");
    fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함
    //배치 처리하는 부분 수정해야됨!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // batch_tgt = llama_batch_get_one(&new_token_id, 1);
    //common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);
    // common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);

    // 13.7. KV 캐시 정리 (Rollback)
    // Target 모델의 KV 캐시에서 거절된 draft 토큰들에 해당하는 항목들을 제거/무효화.
    // 현재 유효한 마지막 위치인 n_past 이후의 캐시 내용을 제거하여 다음 스텝과 일관성 유지.
    {
        LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);
        llama_kv_cache_seq_rm(ctx_tgt, 0, n_past, -1); // 시퀀스 0의 n_past 위치 이후 캐시 제거
    }

    // 2. Get the hidden state from the last layer
    // const int n_layers = llama_model_n_layer(model_tgt);
    // const int n_embd = llama_model_n_embd(model_tgt);
    // float * hidden_state = llama_get_embeddings_ith(ctx_tgt, inp.size()-1); // 마지막 토큰의 hidden state 추출

    // if (hidden_state == nullptr) {
    //     LOG_ERR("%s: Failed to get hidden state from target model\n", __func__);
    //     return 1;
    // }
    
    // // 3. Set the hidden state as input embedding for the draft model
    // //    - Create a tensor for the embedding
    // struct ggml_tensor * embd_tensor = ggml_new_tensor_2d(ctx_dft->ctx, GGML_TYPE_F32, n_embd, 1);
    // //    - Copy the hidden state to the tensor
    // memcpy(embd_tensor->data, hidden_state, n_embd * ggml_element_size(embd_tensor));
    // //    - Set the tensor as input embedding
    // ctx_dft->inp_embd = embd_tensor;
    // ggml_set_input(ctx_dft->inp_embd);
    // // --- End of hidden state extraction and setting ---

    //여기에 바로 Target Model의 Sampler를 사용해서 Initial Token을 생성해야하나? 아니면 common_speculative_gen_draft() 함수처럼
    //새로운 함수를 하나 구현해야하나..?
    // llama_tokens initial_draft = target_model_initialize(spec_target, params_spec, prompt_tgt, id_last);

    printf("Draft Generation Phase에 진입합니다.1\n");
    fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함
    // 13. 메인 생성 루프
    while (true) {
        // 13.1. Draft 토큰 생성
        // common_speculative_gen_draft: speculator 객체(spec)와 파라미터(params_spec),
        // 이전 토큰(id_last), 그리고 필요시 이전 기록(prompt_tgt)을 사용하여 draft 모델(ctx_dft)을 실행하고
        // 후보 토큰 시퀀스(draft)를 생성함.
        // 주석 수정: params_spec는 Target 모델 구조체가 아니라, Speculation 프로세스 파라미터임.
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);

        //printf("Draft Generation Phase에 진입합니다.2\n");
        //fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함

        // 13.2. Target 모델 입력 배치 준비
        common_batch_clear(batch_tgt); // 이전 배치 내용 초기화
        //printf("Draft Generation Phase에 진입합니다.3\n");
        //fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함
        // 마지막으로 수락된 토큰(id_last)을 현재 KV 캐시 위치(n_past)에 추가. n_past는 사용 후 증가됨 (++).
        common_batch_add(batch_tgt, id_last, n_past++, { 0 }, true);


        //printf("Draft Generation Phase에 진입합니다.4\n");
        //fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함
        // 생성된 draft 토큰들을 Target 배치에 추가 (최소 길이 조건 충족 시)
        if (draft.size() >= (size_t) n_draft_min) {
            for (size_t i = 0; i < draft.size(); ++i) {
                // id_last 다음 위치부터 draft 토큰들을 순서대로 배치에 추가
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }
        } else {
            draft.clear(); // 너무 짧은 draft는 무시
        }

        //printf("Draft Generation Phase에 진입합니다.5\n");
        //fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함

        // 13.3. Target 모델 디코딩 (Forward Pass)
        // 준비된 배치(id_last + 유효 draft 토큰들)를 Target 모델(ctx_tgt)에 입력하여
        // 각 토큰 위치에 대한 로짓(logits)을 계산함.
        // 주석: 여기가 타겟 모델 forward 시작점인듯 - 정확함.
        llama_decode(ctx_tgt, batch_tgt);

        //printf("Draft Generation Phase에 진입합니다.\n\n");
        //fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함

        // 13.4. 샘플링 및 Draft 토큰 수락 검증
        // common_sampler_sample_and_accept_n: Target 모델의 로짓과 샘플러(smpl), 그리고 원본 draft 토큰들을 비교.
        // Target 모델의 예측과 draft 토큰이 일치하는지 순차적으로 확인하여,
        // 일치하는(수락된) 토큰 시퀀스(ids)를 반환함. 최소 1개(id_last 다음 토큰)는 항상 포함됨.
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);

        GGML_ASSERT(ids.size() > 0); // 최소 1개 토큰은 항상 수락됨을 단언

        // 13.5. 카운터 업데이트
        // n_past: 수락된 토큰들만큼 KV 캐시 위치 업데이트. id_last 위치는 이미 증가했으므로 추가 수락분(ids.size() - 1)만큼 더함.
        n_past    += ids.size() - 1;
        if (!draft.empty()) { // 유효한 draft가 있었던 경우에만 카운트
            n_drafted += draft.size();      // 시도된 draft 토큰 수 누적
            n_accept  += ids.size() - 1;      // 수락된 draft 토큰 수 누적 (첫 토큰은 draft가 아님)
        }
        n_predict += ids.size();          // 총 생성된 토큰 수 누적 (Target 모델 기준)

        // 13.6. 수락된 토큰 처리 및 출력
        for (size_t i = 0; i < ids.size(); ++i) {
            // 이전 id_last를 prompt_tgt 기록에 추가 (필요시 사용)
            if (i > 0) { // ids의 첫번째 요소는 id_last에 해당하므로 제외
                prompt_tgt.push_back(id_last);
            }
            // 현재 수락된 토큰으로 id_last 업데이트 (다음 루프 입력용)
            id_last = ids[i];

            // EOS 토큰 검사
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break; // EOS면 내부 루프 탈출
            }

            // 수락된 토큰을 텍스트로 변환하여 출력
            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
            // 컬러 출력 처리 (옵션)
            if (params.use_color && i + 1 < ids.size()) { // 수락된 draft 토큰에 색 적용
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                //LOG("{ %s }", token_str.c_str());
                LOG("%s", token_str.c_str());
            }
             fflush(stdout); // 즉시 출력되도록 버퍼 비우기
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // 13.7. KV 캐시 정리 (Rollback)
        // Target 모델의 KV 캐시에서 거절된 draft 토큰들에 해당하는 항목들을 제거/무효화.
        // 현재 유효한 마지막 위치인 n_past 이후의 캐시 내용을 제거하여 다음 스텝과 일관성 유지.
        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);
            llama_kv_cache_seq_rm(ctx_tgt, 0, n_past, -1); // 시퀀스 0의 n_past 위치 이후 캐시 제거
        }

        // 13.8. 루프 종료 조건 검사
        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break; // 최대 생성 길이에 도달했거나 EOS가 생성되면 외부 루프 탈출
        }
    } // end of while(true)

    auto t_dec_end = ggml_time_us(); // 토큰 생성 시간 측정 종료

    const int n_input = inp.size(); // 입력 프롬프트 토큰 수

    // 14. 결과 로깅 및 성능 출력
    LOG("\n\n");
    // 프롬프트 처리 속도 및 생성 속도 출력
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    // Speculative decoding 관련 통계 출력
    LOG_INF("n_draft   = %d\n", n_draft); // 설정된 최대 draft 수
    LOG_INF("n_predict = %d\n", n_predict); // 최종 생성된 토큰 수
    LOG_INF("n_drafted = %d\n", n_drafted); // 시도된 총 draft 토큰 수
    LOG_INF("n_accept  = %d\n", n_accept);  // 수락된 총 draft 토큰 수
    if (n_drafted > 0) { // 0으로 나누기 방지
        LOG_INF("accept rate = %.3f%%\n", 100.0f * n_accept / n_drafted); // 수락률
    } else {
         LOG_INF("accept rate = N/A (no drafts attempted)\n");
    }


    LOG_INF("\n");
    LOG_INF("draft:\n\n"); // Draft 모델 성능 상세 정보 출력
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n"); // Target 모델 성능 상세 정보 출력
    common_perf_print(ctx_tgt, smpl); // common_perf_print는 ctx_tgt 성능 정보 출력

    // 15. 자원 해제
    common_sampler_free(smpl); // 샘플러 객체 메모리 해제
    common_speculative_free(spec); // Speculator 객체 메모리 해제
    // 모델 및 컨텍스트는 common_init_result의 소멸자(destructor)가 자동으로 처리 (스마트 포인터 사용 시)
    llama_backend_free(); // llama.cpp 백엔드 자원 해제

    LOG("\n\n");

    return 0; // 정상 종료
}
