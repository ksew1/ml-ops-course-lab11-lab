import os


class Settings:
    def __init__(self):
        self.s3_bucket = os.getenv("S3_BUCKET", "mlops-lab11-models-ksew1")

        self.s3_model_prefix = "sentence_transformer.model"
        self.s3_classifier_key = "classifier.joblib"

        self.root_dir = os.getcwd()
        self.artifacts_dir = os.path.join(self.root_dir, "artifacts")
        self.model_dir = os.path.join(self.root_dir, "model")

        self.sentence_transformer_dir = os.path.join(
            self.artifacts_dir, "sentence_transformer.model"
        )
        self.classifier_joblib_path = os.path.join(
            self.artifacts_dir, "classifier.joblib"
        )

        self.onnx_embedding_model_path = os.path.join(self.model_dir, "model.onnx")
        self.onnx_classifier_path = os.path.join(self.model_dir, "classifier.onnx")
        self.onnx_tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")

        self.embedding_dim = 384

    def make_dirs(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
