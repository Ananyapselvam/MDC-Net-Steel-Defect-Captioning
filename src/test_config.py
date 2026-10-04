from config import CFG
import torch

print("Device:", CFG.device)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Image size:", CFG.img_size)
print("Vocabulary size:", CFG.vocab_size)

print("PAD:", CFG.pad_idx)
print("SOS:", CFG.sos_idx)
print("EOS:", CFG.eos_idx)
print("UNK:", CFG.unk_idx)

print("Model:", CFG.model_name)
print("Batch size:", CFG.batch_size)