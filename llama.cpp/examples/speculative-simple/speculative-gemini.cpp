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
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // 2. 기본 초기화
    common_init();
    if (params.speculative.model.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);

    // 3. 모델 및 컨텍스트 포인터 선언
    llama_model * model_tgt = NULL;
    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // 4. Target 모델 로딩
    common_init_result llama_init_tgt = common_init_from_params(params);
    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();
    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // 5. Draft 모델 로딩
    common_params params_dft = params; // Create a copy for draft model params
    params_dft.devices         = params.speculative.devices;
    params_dft.model           = params.speculative.model;
    params_dft.n_ctx           = params.speculative.n_ctx;
    // Ensure draft batch size can accommodate n_draft tokens for prompt processing
    params_dft.n_batch         = params.speculative.n_ctx > 0 ? std::max(params.n_batch, (uint32_t)params.speculative.n_max) : std::max(params.n_batch, (uint32_t)params.speculative.n_max);
    params_dft.n_gpu_layers    = params.speculative.n_gpu_layers;
    if (params.speculative.cpuparams.n_threads > 0) {
        params_dft.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }
    params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;

    common_init_result llama_init_dft = common_init_from_params(params_dft);
    ctx_dft   = llama_init_dft.context.get();


    // 6. 모델 호환성 검사
    if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        return 1;
    }

    // 7. 프롬프트 토큰화
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));
        return 1;
    }
    // Ensure target batch size can accommodate the prompt
    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
         LOG_WARN("%s: adjusting batch size from %d to %d to accommodate the prompt\n", __func__, llama_n_batch(ctx_tgt), (int) inp.size());
         llama_batch_set_batch_size(llama_init_tgt.batch, inp.size()); // Assuming batch is accessible or adjust common_init_from_params if needed
         // Or, alternatively, error out if dynamic batch size adjustment isn't feasible/desired
         // LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));
         // return 1;
    }


    LOG("\n\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }
    fflush(stdout); // Ensure prompt is printed immediately

    // 8. Speculative Decoding 파라미터 및 카운터 초기화
    int n_draft         = params.speculative.n_max;
    int n_draft_min     = params.speculative.n_min;
    float p_min         = params.speculative.p_min;
    int n_predict       = 0; // Total generated tokens (target perspective)
    int n_drafted       = 0; // Total attempted draft tokens
    int n_accept        = 0; // Total accepted draft tokens
    bool has_eos        = false;

    const auto t_enc_start = ggml_time_us();

    // 9. Target 모델용 샘플러 초기화
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);
    std::cout << "Model Initialized" << std::endl;

    // 10. 초기 프롬프트 처리 (KV 캐시 워밍업) - **CHANGED: Process the *entire* prompt**
    // Feed the entire prompt to the target model to build the initial KV cache state.
    llama_batch batch_prompt = llama_batch_get_one(inp.data(), inp.size());
    if (llama_decode(ctx_tgt, batch_prompt) != 0) {
         LOG_ERR("%s: llama_decode failed during prompt processing\n", __func__);
         return 1;
    }
    int n_past = inp.size(); // KV cache position is now after the full prompt

    // We also need to process the prompt in the draft model's context
    // to prepare its KV cache for generating drafts later.
    llama_batch batch_prompt_dft = llama_batch_get_one(inp.data(), inp.size());
     if (llama_decode(ctx_dft, batch_prompt_dft) != 0) { // Use draft context
         LOG_ERR("%s: llama_decode failed during prompt processing for draft model\n", __func__);
         return 1;
     }
    // Ensure draft KV cache position matches target
    llama_kv_cache_seq_cp(ctx_tgt, 0, ctx_dft, 0, 0, n_past); // Optional, but good practice for consistency


    // Stores all tokens processed by the target model (prompt + generated)
    llama_tokens prompt_tgt = inp;
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    const auto t_enc_end = ggml_time_us(); // End prompt processing time
    const auto t_dec_start = ggml_time_us(); // Start token generation time

    // --- CHANGED: Generate the *first* token using the Target Model ---
    llama_token id_last; // This will hold the last *accepted* token
    {
        // Sample the next token using the target model's state after processing the prompt
        id_last = common_sampler_sample(smpl, ctx_tgt, -1); // Sample based on logits at n_past - 1
        common_sampler_accept(smpl, ctx_tgt, id_last); // Accept the token in the sampler state

        // Print the first generated token
        const std::string first_token_str = common_token_to_piece(ctx_tgt, id_last);
        LOG("%s", first_token_str.c_str());
        fflush(stdout);

        // Update state
        prompt_tgt.push_back(id_last); // Add the first generated token to the target history
        n_predict++;                   // Increment the prediction count

        // Check for EOS right away
        if (llama_vocab_is_eog(vocab, id_last)) {
            has_eos = true;
        }

        // IMPORTANT: Advance the KV cache pointer in the target context for the token we just generated.
        // We do this by adding it as a batch of size 1.
        llama_batch batch_first = llama_batch_init(1, 0, 1);
        common_batch_add(batch_first, id_last, n_past++, { 0 }, true); // Add token at current n_past, then increment n_past
        if (llama_decode(ctx_tgt, batch_first) != 0) {
             LOG_ERR("%s: llama_decode failed processing the first target-generated token\n", __func__);
             llama_batch_free(batch_first);
             return 1;
        }
         // Also advance the draft model's KV cache to stay synchronized
         llama_kv_cache_seq_cp(ctx_tgt, 0, ctx_dft, 0, n_past - 1, n_past); // Copy the new KV state for id_last
        llama_batch_free(batch_first);
    }
    // --- End of Initial Target Token Generation ---


    // 11. Speculator 초기화 (Remains the same, uses draft context)
    struct common_speculative_params params_spec;
    params_spec.n_draft = n_draft;
    params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft;
    params_spec.p_min   = p_min;
    struct common_speculative * spec = common_speculative_init(ctx_dft);
    // struct common_speculative * spec_target = common_speculative_init(ctx_tgt); // Not needed based on common_speculative_gen_draft usage

    // 12. Target 모델용 배치(Batch) 초기화
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);


    // 13. 메인 생성 루프 (Starts *after* the first token is generated by target)
    while (true) {
        // Exit immediately if EOS was generated by the first target token or predict limit is 0/1
         if (has_eos || (params.n_predict >= 0 && n_predict >= params.n_predict)) {
             break;
         }

        // 13.1. Draft 토큰 생성 (Uses draft model based on the *last accepted* token)
        // **CHANGED: Pass the full target history for context if needed by gen_draft**
        // Note: common_speculative_gen_draft internally uses ctx_dft and its KV cache,
        // which we synchronized after the initial prompt and the first token.
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);

        // 13.2. Target 모델 입력 배치 준비
        common_batch_clear(batch_tgt);
        // Add the last *accepted* token. n_past was already incremented after processing this token.
        // So, the *next* token (the first one we need target logits for) is at n_past.
        // However, the batch API expects the *current* token and its position.
        // Let's re-think the batching logic slightly for clarity.
        // The target model needs to predict the token *after* id_last.
        // Its KV cache is currently at n_past.
        // When verifying drafts, we feed id_last again + the draft tokens.

        common_batch_clear(batch_tgt);
        // Position for id_last (which is already in KV cache) is n_past - 1
        common_batch_add(batch_tgt, id_last, n_past - 1, { 0 }, true); // This recalculates logits for id_last

        // Add draft tokens starting from the current n_past
        if (draft.size() >= (size_t) n_draft_min) {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }
        } else {
            draft.clear(); // Ignore drafts that are too short
        }


        // 13.3. Target 모델 디코딩 (Forward Pass for verification)
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: llama_decode failed during draft verification\n", __func__);
            return 1;
        }


        // 13.4. 샘플링 및 Draft 토큰 수락 검증
        // **CHANGED: Pass n_past - 1 as the sequence position corresponding to id_last's logits**
        // common_sampler_sample_and_accept_n needs the index in the batch's logits array
        // that corresponds to the prediction *after* id_last. Since id_last was added first
        // to the batch (at index 0), its logits correspond to predicting the token at n_past.
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft, 0 /* logit index for token after id_last */);

        GGML_ASSERT(ids.size() > 0); // At least the next token sampled by target is always "accepted"

        // 13.5. 카운터 업데이트
        // How many tokens did we actually accept beyond the first one sampled by target?
        const int n_accepted_in_batch = (int)ids.size() - 1;
        const int n_drafted_in_batch = draft.empty() ? 0 : (int)draft.size();

        if (!draft.empty()) {
            n_drafted += n_drafted_in_batch;
            n_accept  += n_accepted_in_batch;
        }
        n_predict += ids.size(); // Increment total predicted tokens

        // 13.6. 수락된 토큰 처리 및 출력
        for (size_t i = 0; i < ids.size(); ++i) {
            // The token id_last corresponds to the state *before* this loop iteration.
            // The first token in `ids` is the one sampled by the target model for the position *after* id_last.
            id_last = ids[i]; // Update id_last to the newly accepted token

            prompt_tgt.push_back(id_last); // Add accepted token to history

            // EOS 토큰 검사
            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
            }

            // 수락된 토큰을 텍스트로 변환하여 출력
            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
             // Color draft tokens green if accepted (i > 0 means it came from the draft verification part)
            if (params.use_color && i > 0 && n_accepted_in_batch > 0 && i <= (size_t)n_accepted_in_batch) {
                 // Simple green color for accepted draft tokens
                LOG("\u001b[32m%s\u001b[0m", token_str.c_str());
            } else {
                 LOG("%s", token_str.c_str());
            }
            fflush(stdout); // 즉시 출력

            if (has_eos) {
                break; // Stop processing tokens in this batch if EOS is found
            }
        }

        // 13.7. KV 캐시 업데이트 및 정리 (Rollback)
        // Update n_past to reflect the number of accepted tokens
        const int n_advanced = ids.size(); // Number of tokens successfully processed/accepted in this step
        n_past += n_advanced;

        // Remove KV cache entries for the rejected draft tokens.
        // The target KV cache sequence `0` should be trimmed to the new `n_past`.
        llama_kv_cache_seq_rm(ctx_tgt, 0, n_past, -1);

        // Synchronize the draft KV cache to match the accepted state of the target cache.
        llama_kv_cache_seq_cp(ctx_tgt, 0, ctx_dft, 0, n_past - n_advanced, n_past); // Copy KV state for accepted tokens
        llama_kv_cache_seq_rm(ctx_dft, 0, n_past, -1); // Ensure draft cache is also trimmed


        LOG_DBG("accepted %d/%d draft tokens, target KV cache now at n_past = %d\n", n_accepted_in_batch, n_drafted_in_batch, n_past);


        // 13.8. 루프 종료 조건 검사
        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    } // end of while(true)

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    // 14. 결과 로깅 및 성능 출력 (Same as before)
    LOG("\n\n");
    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict); // Total tokens generated by target perspective
    LOG_INF("n_drafted = %d\n", n_drafted); // Draft tokens generated *after* the first target token
    LOG_INF("n_accept  = %d\n", n_accept);  // Draft tokens accepted *after* the first target token
    if (n_drafted > 0) {
        LOG_INF("accept rate = %.3f%%\n", 100.0f * n_accept / n_drafted);
    } else {
         LOG_INF("accept rate = N/A (no drafts attempted)\n");
    }

    LOG_INF("\n");
    LOG_INF("draft performance:\n");
    llama_print_timings(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target performance:\n");
    llama_print_timings(ctx_tgt);
    // common_perf_print might not exist, use llama_print_timings instead.
    // common_perf_print(ctx_tgt, smpl); // If common_perf_print is defined and useful

    // 15. 자원 해제
    common_sampler_free(smpl);
    common_speculative_free(spec);
    // common_speculative_free(spec_target); // Not used
    llama_batch_free(batch_tgt);
    // llama_init_tgt and llama_init_dft handle model/context freeing via smart pointers
    llama_backend_free();

    LOG("\n\n");

    return 0;
}
