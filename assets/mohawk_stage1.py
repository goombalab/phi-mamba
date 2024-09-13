import torch
import torch.nn.init as init

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from modules.lm_head import LMHeadModel
from modules.modeling_phi import PhiForCausalLM
from utils.config import Config

device = "cuda"

teacher_model = PhiForCausalLM.from_pretrained(
    "microsoft/phi-1_5", attn_implementation="eager"
).to(device)
teacher_model.eval()
teacher_model.requires_grad_(False)

model_config = Config.from_json("assets/sample_config.json")
student_model = LMHeadModel(model_config).to(device)

dataset = load_dataset("stas/openwebtext-10k")["train"]
dataloader = DataLoader(dataset, batch_size=4)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

# Stage 1 skeleton
student_model.requires_grad_(True)
for idx, data in enumerate(dataset):
    input_ids = (
        tokenizer(data["text"], return_tensors="pt", truncation=True)
        .to(device)
        .input_ids
    )

    _, seq_len = input_ids.size()

    teacher_outputs = teacher_model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attention_results=True,
        output_attentions=True,
        use_cache=False,
    )

    for layer_idx, student_layer in enumerate(student_model.backbone.layers):
        student_input = teacher_outputs.all_hidden_states[layer_idx]

        # Forward pass
        student_output = student_layer(
            hidden_states=student_input,
            run_mlp_component=False,
            return_mixer_matrix=True,
        )
        transfer_matrix = student_output["transfer_matrix"]
        attn_matrix = teacher_outputs.all_attn_matrices[layer_idx]

        assert transfer_matrix.size() == attn_matrix.size()

        loss = torch.linalg.matrix_norm(
            transfer_matrix - attn_matrix, ord="fro"
        ).mean()

        loss.backward()
        print(f"Iter {idx}, Layer {layer_idx}, Loss: {loss.item()}")
print("DONE")