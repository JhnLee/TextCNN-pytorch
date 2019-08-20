''' Configuration file for CNN base '''

class cnn_config:

    tokenizer_path = 'data/m32k.model'
    vocab_path = 'data/m32k.vocab'
    w2v_path = 'w2v_pretrained'
    trainpath = './data/korean_single_train.csv'
    devpath = './data/korean_single_dev.csv'
    model_dir = 'model'

    num_classes = 7
    vocab_size = 32000
    embedding_size = 150
    epoch = 50
    batch_size = 256
    max_grad_norm = 5

    summary_step = 50



