class RMSNorm {
public:
    RMSNorm(const Tensor& weight);

    void forward(
        const Tensor& input,
        Tensor& output
    );

private:
    Tensor weight_;
};

class Attention {
public:
    Attention(
        const ModelConfig& config,
        const AttentionWeights& weights,
        KVCache& kv_cache
    );

    void forward(
        const Tensor& x,
        Tensor& output,
        int seq_len,
        int start_pos
    );

private:
    Tensor wq_;
    Tensor wk_;
    Tensor wv_;
    Tensor wo_;

    KVCache& kv_cache_;
};

class FeedForward {
public:
    FeedForward(
        const ModelConfig& config,
        const FeedForwardWeights& weights
    );

    void forward(
        const Tensor& x,
        Tensor& output
    );

private:
    Tensor w1_;
    Tensor w2_;
    Tensor w3_;
};

class TransformerBlock {
public:
    TransformerBlock(
        int layer_id,
        const ModelConfig& config,
        const LayerWeights& weights,
        KVCache& kv_cache
    );

    void forward(
        const Tensor& x,
        Tensor& output,
        int seq_len,
        int start_pos
    );

private:
    RMSNorm attention_norm_;
    Attention attention_;

    RMSNorm ffn_norm_;
    FeedForward feed_forward_;
};