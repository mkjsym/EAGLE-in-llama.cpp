// ... (이전 코드) ...

struct llm_build_context {
    // ... (기존 멤버 변수) ...

    struct ggml_cgraph * append_pooling(struct ggml_cgraph * gf) {
        // find result_norm tensor for input
        struct ggml_tensor * inp = nullptr;
        for (int i = ggml_graph_n_nodes(gf) - 1; i >= 0; --i) {
            inp = ggml_graph_node(gf, i);
            if (strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
                break;
            } 
             // 최종 layer hidden_state 추출을 위해 "l_out" 추가 검색
            else if(strcmp(inp->name, "l_out") == 0 && ggml_graph_node_layer(gf,i) == this->n_layer -1) {
              break;
            }
           else {
                inp = nullptr;
            }
        }

        // 최종 layer의 hidden_state를 찾지 못했다면 에러 처리
        GGML_ASSERT(inp != nullptr && "missing result_norm/result_embd/l_out tensor");


        struct ggml_tensor * cur;

        // result_norm, result_embd 이면 pooling 로직 실행
        if(strcmp(inp->name, "result_norm") == 0 || strcmp(inp->name, "result_embd") == 0) {
          switch (pooling_type) {
              case LLAMA_POOLING_TYPE_NONE:
                  {
                      cur = inp;
                  } break;
              case LLAMA_POOLING_TYPE_MEAN:
                  {
                      struct ggml_tensor * inp_mean = build_inp_mean();
                      cur = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, inp)), inp_mean);
                  } break;
              case LLAMA_POOLING_TYPE_CLS:
              case LLAMA_POOLING_TYPE_LAST:
                  {
                      struct ggml_tensor * inp_cls = build_inp_cls();
                      cur = ggml_get_rows(ctx0, inp, inp_cls);
                  } break;
              case LLAMA_POOLING_TYPE_RANK:
                  {
                      struct ggml_tensor * inp_cls = build_inp_cls();
                      inp = ggml_get_rows(ctx0, inp, inp_cls);

                      // classification head
                      // https://github.com/huggingface/transformers/blob/5af7d41e49bbfc8319f462eb45253dcb3863dfb7/src/transformers/models/roberta/modeling_roberta.py#L1566
                      GGML_ASSERT(model.cls       != nullptr);
                      GGML_ASSERT(model.cls_b     != nullptr);

                      cur = ggml_add (ctx0, ggml_mul_mat(ctx0, model.cls, inp), model.cls_b);
                      cur = ggml_tanh(ctx0, cur);

                      // some models don't have `cls_out`, for example: https://huggingface.co/jinaai/jina-reranker-v1-tiny-en
                      // https://huggingface.co/jinaai/jina-reranker-v1-tiny-en/blob/cb5347e43979c3084a890e3f99491952603ae1b7/modeling_bert.py#L884-L896
                      if (model.cls_out) {
                          GGML_ASSERT(model.cls_out_b != nullptr);

                          cur = ggml_add (ctx0, ggml_mul_mat(ctx0, model.cls_out, cur), model.cls_out_b);
                      }
                  } break;
              default:
                  {
                      GGML_ABORT("unknown pooling type");
                  }
          }
          cb(cur, "result_embd_pooled", -1);
          ggml_build_forward_expand(gf, cur);
          
        }else {
            // 최종 layer의 hidden state 이면 cur에 저장
            cur = inp;

             cb(cur, "final_layer_hidden_state", -1);
            ggml_build_forward_expand(gf, cur);
        }

        return gf;
    }

    // ... (나머지 코드) ...
};
