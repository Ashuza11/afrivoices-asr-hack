# AfriVoices East Africa ASR

A single unified Automatic Speech Recognition (ASR) model for six East African languages, built for the [AfriVoices East Africa ASR Hackathon](https://www.kaggle.com/competitions/afrivoices-east-africa-asr-hackathon).

## Languages

| Language | ISO Code | Dialects |
|---|---|---|
| Swahili | swa | Nairobi, Kisii, Wajir, Mombasa, Nakuru, Dar-es-Salaam |
| Kikuyu | kik | Gĩ-Kabete, Ki-Mathira, Ki-Muranga, Ki-Ndia |
| Luo (Dholuo) | luo | Nyandwat, Milambo |
| Somali | som | Maxatire, Mogadishu |
| Kalenjin | kln | Nandi, Kipsigis |
| Maasai | mas | Kimasaai, Kisamburu |

## Approach

Fine-tuning [openai/whisper-small](https://huggingface.co/openai/whisper-small) (244M parameters) on the combined multilingual dataset with:
- Temperature-based language sampling (α=0.7) to balance low-resource languages
- Curriculum learning: scripted speech first, then unscripted
- SpecAugment + speed perturbation for data augmentation
- fp16 mixed precision + gradient checkpointing on Kaggle GPU

Target: CPU-only inference, ≤8 GB RAM, real-time factor ≤2x on Raspberry Pi 4.

## Hardware constraints (competition rules)

- Parameters: < 1 billion
- Inference: CPU-only, ≤ 8 GB RAM
- Latency: RTF ≤ 2x on Raspberry Pi 4
- License: Apache-2.0

## Repository structure

```
eval/
  compute_wer.py          WER/CER evaluation harness
notebooks/
  kaggle_01_baseline_submission.py   Zero-shot baseline on Kaggle
  kaggle_02_data_and_finetune.py     Full multilingual fine-tuning pipeline
  lesson2_spectrogram_wer.py         Log-mel spectrogram + WER demo
  lesson3_ctc_vs_seq2seq.py          CTC vs seq2seq latency comparison
src/
  train_whisper.py        Whisper fine-tuning script (local smoke-test + Kaggle)
```

## Data sources

- `MCAA1-MSU/anv_data_ke` — Kikuyu, Kalenjin, Luo, Maasai, Somali (Maxatire)
- `DigitalUmuganda/Afrivoice_Swahili` — Swahili
- `DigitalUmuganda/Afrivoice` — Somali (Mogadishu)

All datasets are licensed under CC BY 4.0.

## License

Apache-2.0
