import unittest
import tempfile
from pathlib import Path

import torch
from transformers import Wav2Vec2Config

from round7.data import LANGUAGES
from round7.model import LanguageConditionedXLSRForCTC


class LanguageConditionedModelTests(unittest.TestCase):
    def tiny_model(self):
        config = Wav2Vec2Config(
            vocab_size=12,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            conv_dim=(4,),
            conv_stride=(2,),
            conv_kernel=(3,),
            num_conv_pos_embeddings=8,
            num_conv_pos_embedding_groups=2,
            final_dropout=0.0,
            pad_token_id=0,
            ctc_zero_infinity=True,
        )
        config.language_codes = list(LANGUAGES)
        return LanguageConditionedXLSRForCTC(config).eval()

    def test_forward_requires_language_ids(self):
        model = self.tiny_model()
        with self.assertRaisesRegex(ValueError, "language_ids is required"):
            model(torch.randn(2, 64))

    def test_language_conditioning_changes_logits(self):
        torch.manual_seed(7)
        model = self.tiny_model()
        audio = torch.randn(1, 64).repeat(2, 1)
        output = model(audio, language_ids=torch.tensor([0, 1]))
        self.assertEqual(output.logits.shape[0], 2)
        self.assertFalse(torch.allclose(output.logits[0], output.logits[1]))

    def test_ctc_loss_is_finite(self):
        model = self.tiny_model()
        output = model(
            torch.randn(2, 96),
            language_ids=torch.tensor([0, 5]),
            labels=torch.tensor([[3, 4, -100], [5, 6, 7]]),
        )
        self.assertTrue(torch.isfinite(output.loss))

    def test_checkpoint_round_trip(self):
        model = self.tiny_model()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            reloaded = LanguageConditionedXLSRForCTC.from_pretrained(Path(directory))
            self.assertEqual(reloaded.config.language_codes, list(LANGUAGES))
            self.assertEqual(reloaded.lm_head.out_features, 12)
