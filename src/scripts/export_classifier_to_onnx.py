import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from settings import Settings


def export_classifier_to_onnx(settings: Settings):
    print(f"Loading classifier from {settings.classifier_joblib_path}...")
    classifier = joblib.load(settings.classifier_joblib_path)

    initial_type = [("float_input", FloatTensorType([None, settings.embedding_dim]))]

    print("Converting classifier to ONNX...")
    onnx_model = convert_sklearn(classifier, initial_types=initial_type)

    print(f"Saving ONNX classifier to {settings.onnx_classifier_path}...")
    with open(settings.onnx_classifier_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == "__main__":
    s = Settings()
    s.make_dirs()
    export_classifier_to_onnx(s)
