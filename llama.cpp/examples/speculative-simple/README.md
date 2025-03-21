# llama.cpp/examples/speculative-simple

Demonstration of basic greedy speculative decoding

```bash
llama.cpp/build/bin/llama-speculative-simple -m  llama.cpp/models/gemma-7b.gguf -md llama.cpp/models/EAGLE.gguf -f test.txt -c 0 -ngl 99 --color 
    --sampling-seq k --top-k 1 -fa --temp 0.0 \
    -ngld 99 --draft-max 16 --draft-min 5 --draft-p-min 0.9
```
