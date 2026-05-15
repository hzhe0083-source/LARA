import unittest

import torch

from Lara.model.framework.qwen import QwenActionTokenizer


class QwenActionTokenizerTest(unittest.TestCase):
    def test_validate_token_mask_accepts_expected_counts(self):
        mask = torch.tensor(
            [
                [True, False, True, False],
                [False, True, False, True],
            ]
        )

        QwenActionTokenizer._validate_token_mask(mask, expected_count=2, stream_name="latent action")

    def test_validate_token_mask_rejects_mismatched_counts(self):
        mask = torch.tensor(
            [
                [True, False, True, False],
                [False, True, True, True],
            ]
        )

        with self.assertRaisesRegex(ValueError, "Unexpected latent action token count per sample"):
            QwenActionTokenizer._validate_token_mask(mask, expected_count=2, stream_name="latent action")


if __name__ == "__main__":
    unittest.main()
