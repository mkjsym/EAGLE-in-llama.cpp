// #include "llama.h"
// #include <cstdio>
// #include <cstring>
// #include <string>
// #include <vector>

// static void print_usage(int, char ** argv) {
//     printf("\nexample usage:\n");
//     printf("\n    %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
//     printf("\n");
// }

// int main(int argc, char ** argv) {
//     // path to the model gguf file
//     std::string model_path;
//     // prompt to generate text from
//     std::string prompt = "Hello my name is";
//     // number of layers to offload to the GPU
//     int ngl = 99;
//     // number of tokens to predict
//     int n_predict = 32;

//     // parse command line arguments

//     {
//         int i = 1;
//         for (; i < argc; i++) {
//             if (strcmp(argv[i], "-m") == 0) {
//                 if (i + 1 < argc) {
//                     model_path = argv[++i];
//                 } else {
//                     print_usage(argc, argv);
//                     return 1;
//                 }
//             } else if (strcmp(argv[i], "-n") == 0) {
//                 if (i + 1 < argc) {
//                     try {
//                         n_predict = std::stoi(argv[++i]);
//                     } catch (...) {
//                         print_usage(argc, argv);
//                         return 1;
//                     }
//                 } else {
//                     print_usage(argc, argv);
//                     return 1;
//                 }
//             } else if (strcmp(argv[i], "-ngl") == 0) {
//                 if (i + 1 < argc) {
//                     try {
//                         ngl = std::stoi(argv[++i]);
//                     } catch (...) {
//                         print_usage(argc, argv);
//                         return 1;
//                     }
//                 } else {
//                     print_usage(argc, argv);
//                     return 1;
//                 }
//             } else {
//                 // prompt starts here
//                 break;
//             }
//         }
//         if (model_path.empty()) {
//             print_usage(argc, argv);
//             return 1;
//         }
//         if (i < argc) {
//             prompt = argv[i++];
//             for (; i < argc; i++) {
//                 prompt += " ";
//                 prompt += argv[i];
//             }
//         }
//     }

//     // load dynamic backends

//     ggml_backend_load_all();

//     // initialize the model

//     llama_model_params model_params = llama_model_default_params();
//     model_params.n_gpu_layers = ngl;

//     llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
//     const llama_vocab * vocab = llama_model_get_vocab(model);

//     if (model == NULL) {
//         fprintf(stderr , "%s: error: unable to load model\n" , __func__);
//         return 1;
//     }

//     // tokenize the prompt

//     // find the number of tokens in the prompt
//     const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

//     // allocate space for the tokens and tokenize the prompt
//     std::vector<llama_token> prompt_tokens(n_prompt);
//     if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
//         fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
//         return 1;
//     }

//     // initialize the context

//     llama_context_params ctx_params = llama_context_default_params();
//     // n_ctx is the context size
//     ctx_params.n_ctx = n_prompt + n_predict - 1;
//     // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
//     ctx_params.n_batch = n_prompt;
//     // enable performance counters
//     ctx_params.no_perf = false;

//     llama_context * ctx = llama_init_from_model(model, ctx_params);

//     if (ctx == NULL) {
//         fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
//         return 1;
//     }

//     // initialize the sampler

//     auto sparams = llama_sampler_chain_default_params();
//     sparams.no_perf = false;
//     llama_sampler * smpl = llama_sampler_chain_init(sparams);

//     llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

//     // print the prompt token-by-token

//     for (auto id : prompt_tokens) {
//         char buf[128];
//         int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
//         if (n < 0) {
//             fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
//             return 1;
//         }
//         std::string s(buf, n);
//         printf("%s", s.c_str());
//     }

//     // prepare a batch for the prompt

//     llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

//     // main loop

//     const auto t_main_start = ggml_time_us();
//     int n_decode = 0;
//     llama_token new_token_id;

//     for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + n_predict; ) {
//         // evaluate the current batch with the transformer model
//         if (llama_decode(ctx, batch)) {
//             fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
//             return 1;
//         }

//         n_pos += batch.n_tokens;

//         // sample the next token
//         {
//             new_token_id = llama_sampler_sample(smpl, ctx, -1);

//             // is it an end of generation?
//             if (llama_vocab_is_eog(vocab, new_token_id)) {
//                 break;
//             }

//             char buf[128];
//             int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
//             if (n < 0) {
//                 fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
//                 return 1;
//             }
//             std::string s(buf, n);
//             printf("%s", s.c_str());
//             fflush(stdout);

//             // prepare the next batch with the sampled token
//             batch = llama_batch_get_one(&new_token_id, 1);

//             n_decode += 1;
//         }
//     }

//     printf("\n");

//     const auto t_main_end = ggml_time_us();

//     fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
//             __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

//     fprintf(stderr, "\n");
//     llama_perf_sampler_print(smpl);
//     llama_perf_context_print(ctx);
//     fprintf(stderr, "\n");

//     llama_sampler_free(smpl);
//     llama_free(ctx);
//     llama_model_free(model);

//     return 0;
// }

#include "llama.h" // llama.cpp 핵심 라이브러리 헤더
#include <cstdio>   // C 표준 입출력 (printf, fprintf)
#include <cstring>  // C 문자열 처리 함수 (strcmp)
#include <string>   // C++ 문자열 클래스
#include <vector>   // C++ 벡터 컨테이너

// 프로그램 사용법을 출력하는 함수
static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    // 커맨드 라인 사용 예시 출력: 모델 파일 지정, 예측 토큰 수(-n), GPU 레이어 수(-ngl), 프롬프트(선택)
    printf("\n     %s -m model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]\n", argv[0]);
    printf("\n");
}

// 메인 함수
int main(int argc, char ** argv) {
    // 모델 gguf 파일 경로를 저장할 변수
    std::string model_path;
    // 텍스트 생성을 시작할 프롬프트 문자열 (기본값 설정)
    std::string prompt = "Hello my name is";
    // GPU에 오프로드할 레이어 수 (기본값: 99 - 가능한 많이)
    int ngl = 99;
    // 예측(생성)할 토큰의 수 (기본값: 32)
    int n_predict = 32;

    // 커맨드 라인 인자(argument) 파싱
    {
        int i = 1; // 프로그램 이름(argv[0]) 다음부터 시작
        for (; i < argc; i++) { // 모든 인자 순회
            if (strcmp(argv[i], "-m") == 0) { // '-m' 인자 확인 (모델 경로)
                if (i + 1 < argc) { // 다음 인자가 있는지 확인
                    model_path = argv[++i]; // 다음 인자를 모델 경로로 저장하고 인덱스 증가
                } else { // '-m' 뒤에 경로가 없으면 사용법 출력 후 종료
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) { // '-n' 인자 확인 (예측 토큰 수)
                if (i + 1 < argc) { // 다음 인자가 있는지 확인
                    try { // 문자열을 정수로 변환 시도
                        n_predict = std::stoi(argv[++i]); // 다음 인자를 정수로 변환하고 저장, 인덱스 증가
                    } catch (...) { // 변환 실패 시 사용법 출력 후 종료
                        print_usage(argc, argv);
                        return 1;
                    }
                } else { // '-n' 뒤에 숫자가 없으면 사용법 출력 후 종료
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) { // '-ngl' 인자 확인 (GPU 레이어 수)
                if (i + 1 < argc) { // 다음 인자가 있는지 확인
                    try { // 문자열을 정수로 변환 시도
                        ngl = std::stoi(argv[++i]); // 다음 인자를 정수로 변환하고 저장, 인덱스 증가
                    } catch (...) { // 변환 실패 시 사용법 출력 후 종료
                        print_usage(argc, argv);
                        return 1;
                    }
                } else { // '-ngl' 뒤에 숫자가 없으면 사용법 출력 후 종료
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                // 옵션 인자(-m, -n, -ngl)가 아니면 프롬프트 시작으로 간주하고 루프 탈출
                break;
            }
        }
        // 모델 경로가 지정되지 않았으면 필수이므로 사용법 출력 후 종료
        if (model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        // 루프 탈출 후 남은 인자들이 있다면 프롬프트로 처리
        if (i < argc) {
            prompt = argv[i++]; // 첫 번째 남은 인자를 프롬프트로 설정
            // 여러 단어로 이루어진 프롬프트를 위해 나머지 인자들을 공백과 함께 이어붙임
            for (; i < argc; i++) {
                prompt += " ";
                prompt += argv[i];
            }
        }
    }

    // 동적 백엔드 로드 (예: CUDA, Metal 등)
    ggml_backend_load_all();

    // 모델 초기화
    // 기본 모델 파라미터 가져오기
    llama_model_params model_params = llama_model_default_params();
    // 파싱한 GPU 레이어 수 설정
    model_params.n_gpu_layers = ngl;

    // 지정된 경로의 모델 파일 로드
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    // 로드된 모델에서 어휘 사전(vocabulary) 가져오기
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 모델 로드 실패 시 오류 메시지 출력 후 종료
    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // 프롬프트 토큰화
    // 프롬프트에 있는 토큰 수를 계산 (버퍼=NULL, 크기=0 전달 시 토큰 수 반환)
    // true, true: 각각 BOS 추가 여부, 특수 토큰 처리 여부
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // 계산된 토큰 수만큼 벡터 메모리 할당
    std::vector<llama_token> prompt_tokens(n_prompt);
    // 실제 토큰화 수행하여 prompt_tokens 벡터 채우기
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1; // 토큰화 실패 시 오류
    }

    // 컨텍스트(context) 초기화
    // 기본 컨텍스트 파라미터 가져오기
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx: 컨텍스트 크기 설정 (프롬프트 길이 + 예측 길이 - 1 이상 필요)
    ctx_params.n_ctx = n_prompt + n_predict - 1;
    // n_batch: 한 번의 llama_decode 호출로 처리할 최대 토큰 수 (여기서는 프롬프트 전체를 한 번에 처리하도록 설정)
    ctx_params.n_batch = n_prompt;
    // 성능 카운터 활성화 (no_perf = false)
    ctx_params.no_perf = false;

    // 모델과 파라미터를 사용하여 컨텍스트 생성
    llama_context * ctx = llama_init_from_model(model, ctx_params);

    // 컨텍스트 생성 실패 시 오류 메시지 출력 후 종료
    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // 샘플러(sampler) 초기화
    // 기본 샘플러 체인 파라미터 가져오기
    auto sparams = llama_sampler_chain_default_params();
    // 성능 카운터 활성화
    sparams.no_perf = false;
    // 샘플러 체인 초기화
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    // 샘플러 체인에 탐욕적(greedy) 샘플러 추가 (가장 확률 높은 토큰만 선택)
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // 프롬프트를 토큰 단위로 출력
    printf("Prompt: ");
    for (auto id : prompt_tokens) { // 프롬프트 토큰 벡터 순회
        char buf[128]; // 토큰 텍스트를 저장할 버퍼
        // 토큰 ID를 텍스트 조각(piece)으로 변환
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) { // 변환 실패 시 오류
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n); // 변환된 텍스트 조각으로 문자열 생성
        printf("%s", s.c_str()); // 출력
    }
     printf("\n"); // 프롬프트 출력 후 줄바꿈

    // 프롬프트 처리를 위한 초기 배치 준비
    // prompt_tokens 벡터 전체를 포함하는 배치 생성
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // 메인 생성 루프
    const auto t_main_start = ggml_time_us(); // 생성 시작 시간 기록
    int n_decode = 0; // 디코딩된 토큰 수 카운터
    llama_token new_token_id; // 새로 샘플링된 토큰 ID 저장 변수

    printf("Generated: ");
    // 루프 조건: 현재 위치(n_pos) + 현재 배치 토큰 수 < 목표 길이 (프롬프트 + 예측)
    // 즉, 목표 길이만큼 생성될 때까지 반복
    for (int n_pos = 0; n_pos + batch.n_tokens <= n_prompt + n_predict; ) {
        // 현재 배치(batch)를 모델(ctx)로 처리 (순방향 연산)
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1); // 실패 시 오류
            return 1;
        }

        // 처리된 토큰 수만큼 현재 위치(n_pos) 업데이트
        n_pos += batch.n_tokens;

        // 다음 토큰 샘플링
        {
            // 샘플러(smpl)와 컨텍스트(ctx)를 사용하여 다음 토큰 ID 샘플링
            // idx = -1 은 일반적으로 마지막 토큰 위치의 로짓을 사용하라는 의미
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // 샘플링된 토큰이 문장 끝(End of Generation) 토큰인지 확인
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                printf(" [EOS]"); // EOS 토큰이면 표시하고 루프 종료
                break;
            }

            // 샘플링된 토큰 ID를 텍스트로 변환하여 출력
            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) { // 변환 실패 시 오류
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str()); // 출력
            fflush(stdout); // 버퍼를 비워 즉시 출력되도록 함

            // 다음 루프 반복을 위해 방금 샘플링된 토큰 1개만 포함하는 새 배치 준비
            batch = llama_batch_get_one(&new_token_id, 1);

            // 디코딩된 토큰 수 증가
            n_decode += 1;
        }
    }

    printf("\n"); // 최종 출력 후 줄바꿈

    const auto t_main_end = ggml_time_us(); // 생성 종료 시간 기록

    // 생성 결과 및 성능 통계 출력
    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    // 샘플러 성능 통계 출력
    llama_perf_sampler_print(smpl);
    // 컨텍스트 성능 통계 출력
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    // 자원 해제
    llama_sampler_free(smpl); // 샘플러 메모리 해제
    llama_free(ctx);          // 컨텍스트 메모리 해제
    llama_model_free(model);  // 모델 메모리 해제

    // 정상 종료
    return 0;
}
