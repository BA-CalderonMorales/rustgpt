use llm::{
    Dataset as RootDataset, DatasetType as RootDatasetType, EMBEDDING_DIM,
    Embeddings as RootEmbeddings, HIDDEN_DIM, LLM as RootLlm, Layer as RootLayer, MAX_SEQ_LEN,
    Vocab as RootVocab,
    adam::Adam,
    dataset_loader::{Dataset, DatasetType},
    embeddings::Embeddings,
    feed_forward::{FeedForward, OptimizerKind, RmsProp, Sgd},
    layer_norm::LayerNorm,
    llm::{LLM, Layer},
    output_projection::OutputProjection,
    self_attention::SelfAttention,
    transformer::TransformerBlock,
    vocab::Vocab,
};

fn assert_module_layer<T: Layer>() {}
fn assert_root_layer<T: RootLayer>() {}

#[test]
fn existing_public_modules_reexports_constants_and_fields_compile() {
    assert_eq!(MAX_SEQ_LEN, 80);
    assert_eq!(EMBEDDING_DIM, 128);
    assert_eq!(HIDDEN_DIM, 256);

    let adam = Adam::new((1, 2));
    assert_eq!(adam.m.dim(), (1, 2));
    assert_eq!(adam.v.dim(), (1, 2));

    let dataset = Dataset {
        pretraining_data: vec!["pretraining".to_string()],
        chat_training_data: vec!["chat".to_string()],
    };
    assert_eq!(dataset.pretraining_data.len(), 1);
    assert_eq!(dataset.chat_training_data.len(), 1);
    let root_dataset: RootDataset = dataset;
    assert_eq!(root_dataset.pretraining_data[0], "pretraining");
    let _module_json = DatasetType::JSON;
    let _module_csv = DatasetType::CSV;
    let _root_json = RootDatasetType::JSON;
    let _root_csv = RootDatasetType::CSV;

    let embeddings = Embeddings::default();
    assert_eq!(embeddings.token_embeddings.ncols(), EMBEDDING_DIM);
    assert_eq!(embeddings.positional_embeddings.nrows(), MAX_SEQ_LEN);
    assert!(embeddings.cached_input.is_none());
    assert_eq!(
        embeddings.token_optimizer.m.dim(),
        embeddings.token_embeddings.dim()
    );
    assert_eq!(
        embeddings.positional_optimizer.m.dim(),
        embeddings.positional_embeddings.dim()
    );
    let _root_embeddings: RootEmbeddings = embeddings;

    let _feed_forward = FeedForward::new(2, 4);
    let _optimizer_kinds = [
        OptimizerKind::Adam,
        OptimizerKind::Sgd,
        OptimizerKind::RmsProp,
    ];
    let _sgd = Sgd;
    let rms_prop = RmsProp::new((1, 2));
    assert_eq!(rms_prop.squared_gradients.dim(), (1, 2));

    let _layer_norm = LayerNorm::new(2);

    let llm = LLM::default();
    assert!(!llm.vocab.words.is_empty());
    assert!(!llm.network.is_empty());
    let root_llm: RootLlm = llm;
    assert!(!root_llm.network.is_empty());

    let output_projection = OutputProjection::new(2, 3);
    assert_eq!(output_projection.w_out.dim(), (2, 3));
    assert_eq!(output_projection.b_out.dim(), (1, 3));
    assert_eq!(output_projection.optimizer.m.dim(), (2, 3));
    assert!(output_projection.cached_input.is_none());

    let self_attention = SelfAttention::new(2);
    assert_eq!(self_attention.embedding_dim, 2);
    let _transformer = TransformerBlock::new(2, 4);

    let vocab = Vocab::default();
    assert_eq!(vocab.encode.get("hello"), Some(&0));
    assert_eq!(vocab.decode.get(&0).map(String::as_str), Some("hello"));
    assert_eq!(vocab.words[0], "hello");
    let root_vocab: RootVocab = vocab;
    assert_eq!(root_vocab.words[0], "hello");

    assert_module_layer::<Embeddings>();
    assert_module_layer::<FeedForward>();
    assert_module_layer::<LayerNorm>();
    assert_module_layer::<OutputProjection>();
    assert_module_layer::<SelfAttention>();
    assert_module_layer::<TransformerBlock>();
    assert_root_layer::<RootEmbeddings>();

    let _module_llm_constructor: fn(Vocab, Vec<Box<dyn Layer>>) -> LLM = LLM::new;
    let _root_llm_constructor: fn(RootVocab, Vec<Box<dyn RootLayer>>) -> RootLlm = RootLlm::new;
}
