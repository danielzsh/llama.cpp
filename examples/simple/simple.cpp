#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -p \"Hello my name is\" -n 32\n", argv[0]);
    LOG_TEE("\n");
}

llama_token_data_array get_logits(llama_model * model, llama_context * ctx, llama_batch batch) {
    for (int i = 0; i < batch.n_tokens - 1; i++) batch.logits[i] = false;
    batch.logits[batch.n_tokens - 1] = true;
    llama_decode(ctx, batch); 
    // printf("\nbatch n_tokens: %d\n", batch.n_tokens);
    auto   n_vocab = llama_n_vocab(model);
    auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
    return candidates_p;
}
float get_uncertainty(llama_token_data_array candidates_p) {
    float sum_exp = 0, max_logit = -INFINITY;
    for (int i = 0; i < candidates_p.size; i++) {
        sum_exp += exp(candidates_p.data[i].logit);
        max_logit = std::max(max_logit, candidates_p.data[i].logit);
    }
    float uncertainty = log(sum_exp) - max_logit;
    return uncertainty;
}
//creates models (hence the name)
std::pair<llama_model *, llama_context *> create_model(const char * file, gpt_params params) {
    llama_model_params model_params = llama_model_params_from_gpt_params(params);
    llama_model * model = llama_load_model_from_file(file, model_params);
    assert(model);
    llama_context_params ctx_params = llama_context_params_from_gpt_params(params);
    llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    assert(ctx);
    return {model, ctx};
}

// i forgot the type
/*
Mp denotes the accurate LLM, Me denotes the efficient SLM, input1..k denotes the user request prompts, T denotes the output
length, and T hresholdc denotes the adaptive threshold to balance output quality and throughput. We
also provide the vanilla sampling method of Auto-Regressive Sampling in Alg. 1 for comparison.


Input: M_p(.|.), M_e(.|.), input_1..k , T ,Threshold_c

Initialize n ← t

for i = 1 to T do
    Q_i ← M_e(x|input + {x_1, ..., x_i−1})
    C_i ← Conf idence(Q_i)
    if C_i > Threshold_c then
        Sample x_i ∼ Q_i
    else
        P_i ← M_p(x|input + {x_1, ..., x_i−1})
        Sample x_i ∼ P_i
    end if
end for

return x_1, ..., x_T
*/

int main(int argc, char ** argv) {
    gpt_params params;

    params.prompt = "Hello my name is";
    params.n_predict = 32;
    // params.n_threads = 1;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    // total length of the sequence including the prompt
    const int n_predict = params.n_predict;

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

    // initialize the model
    auto [model_sm, ctx_sm] = create_model("llama-2-7b.Q8_0.gguf", params);
    auto [model_lg, ctx_lg] = create_model("llama-2-13b.Q8_0.gguf", params);
    printf("llama 7b token count: %d\n", llama_n_vocab(model_sm));
    printf("llama 13b token count: %d\n", llama_n_vocab(model_lg));

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = ::llama_tokenize(ctx_sm, params.prompt, true);

    const int n_ctx    = llama_n_ctx(ctx_sm);
    const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size());

    LOG_TEE("\n%s: n_predict = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_predict, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        LOG_TEE("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        LOG_TEE("%s:        either reduce n_predict or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    fprintf(stderr, "\n");

    for (auto id : tokens_list) {
        fprintf(stderr, "%s", llama_token_to_piece(ctx_sm, id).c_str());
        // assert(llama_token_to_piece(ctx_sm, id) == llama_token_to_piece(ctx_lg, id));
    }

    fflush(stderr);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt

    // if (llama_decode(ctx_sm, batch) != 0) {
    //     LOG_TEE("%s: llama_decode() failed\n", __func__);
    //     return 1;
    // }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();
    printf("%d %d\n", n_cur, n_predict);
    // stuff happens here
    while (n_cur < n_predict) {
        // printf("\n%d\n", batch.n_tokens);
        // sample the next token
        {
            // auto   n_vocab = llama_n_vocab(model_sm);
            // auto * logits  = llama_get_logits_ith(ctx_sm, batch.n_tokens - 1);

            // std::vector<llama_token_data> candidates;
            // candidates.reserve(n_vocab);

            // for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            //     candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            // }

            // llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };
            llama_context * ctx = ctx_sm;
            auto candidates_p = get_logits(model_sm, ctx_sm, batch);
            
            // double max_p = INFINITY;
            // if (n_cur == n_predict) printf("finished printing logits\n");
            // softmax: probability = e^logit / sum(e^logit)
            // nll: -ln(probability)
            // nll(softmax(logit)) = ln(sum(e^logit)) - logit
            // uncertaintly = minimum nll
            // probability = 1 means that nll = 0, hence why high uncertainty is bad 
            // calculate uncertainty 
            float uncertainty = get_uncertainty(candidates_p);
            // printf("\nuncertainty: %f\n", uncertainty);

            if (uncertainty > 1.0) {
                LOG_TEE(" *");
                candidates_p = get_logits(model_lg, ctx_lg, batch);
                ctx = ctx_lg;
            }
            
            // sample the most likely token
            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);
             
            //negative log likelihood (NLL) confidence score
            
            // Confidence(Pi) = NLL(t1)
            
            // is it an end of generation?
            if (llama_token_is_eog(model_sm, new_token_id) || n_cur == n_predict) {
                LOG_TEE("\n");

                break;
            }
            assert(llama_token_to_piece(ctx_sm, new_token_id) == llama_token_to_piece(ctx_lg, new_token_id));
            LOG_TEE("|%s|", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            llama_batch_clear(batch);

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);
            llama_kv_update(ctx_sm, batch);
            llama_kv_update(ctx_lg, batch);
            n_decode += 1;
        }

        n_cur += 1;
    }

    LOG_TEE("\n");

    const auto t_main_end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));
    // for (int i = 0; i < 75; i++) printf("%d: %ld\n", i, t[i]);
    int64_t tot = 0;
    double kqv = 0;
    int c = 0;
    // for (int i = 0; i < sz; i++) if (s[i]) {
    //     std::string S(s[i]);
    //     rt[S.substr(0, S.find("-"))] += r[i] - l[i];

    //     if (S.substr(0, S.find("-")) == "Kcur") {
    //         kqv += r[i] - l[i];
    //         ++c;
    //     }
    //     tot += r[i] - l[i];
    // }
    // printf("%d repetitive k_cur computation of key avoided\n", tot_tokens);
    // printf("%f average time spent on Kcur\n", kqv / c);
    // printf("%d total k_cur computations\n", c); 
    std::vector<std::pair<std::string, int64_t>> v;

    llama_print_timings(ctx_sm);

    fprintf(stderr, "\n");

    llama_batch_free(batch);

    llama_free(ctx_sm);
    llama_free_model(model_sm);

    llama_backend_free();

    return 0;
}
