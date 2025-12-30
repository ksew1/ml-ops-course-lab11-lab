import torch
from transformers import AutoTokenizer, AutoModel
from settings import Settings


class SentenceEmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled


def export_model_to_onnx(settings: Settings):
    print(f"Loading model from {settings.sentence_transformer_dir}...")
    base_model = AutoModel.from_pretrained(settings.sentence_transformer_dir)
    tokenizer = AutoTokenizer.from_pretrained(settings.sentence_transformer_dir)

    model = SentenceEmbeddingModel(base_model)
    model.eval()

    dummy_text = "This is a sample input for ONNX export."
    inputs = tokenizer(dummy_text, return_tensors="pt")

    print(f"Exporting to {settings.onnx_embedding_model_path}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs.get("attention_mask")),
            settings.onnx_embedding_model_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["sentence_embedding"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "sentence_embedding": {0: "batch_size"},
            },
            opset_version=17,
        )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(settings.model_dir)

    print("Export complete.")


if __name__ == "__main__":
    s = Settings()
    s.make_dirs()
    export_model_to_onnx(s)
