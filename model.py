"""
CSE 251B — Example model.py

This is a minimal example showing the required interface for your submission.
Your model.py must contain a load_model() function with this signature.

This example implements a tiny 2-layer Transformer (~1M params) for demonstration.
It will have terrible perplexity — it's just to show the required structure.
"""

import torch
import torch.nn as nn


# --- Your model definition (replace with your architecture) ---

class TinyGPT(nn.Module):
    """A minimal GPT for demonstration purposes."""

    def __init__(self, vocab_size=50257, n_embd=128, n_head=4, n_layer=2, block_size=1024):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd, nhead=n_head,
                dim_feedforward=4 * n_embd, dropout=0.1,
                activation="gelu", batch_first=True,
            )
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        Args:
            input_ids: LongTensor of shape (batch_size, seq_len)
        Returns:
            logits: FloatTensor of shape (batch_size, seq_len, 50257)
        """
        B, T = input_ids.shape
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = tok_emb + pos_emb

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1).bool()

        for block in self.blocks:
            x = block(x, src_mask=mask, is_causal=True)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# --- Required: load_model function ---

def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """
    Load your trained model from a checkpoint.

    This function is called by evaluate.py. It must return a model where:
        model(input_ids) -> logits
        - input_ids: LongTensor of shape (batch, seq_len)
        - logits: FloatTensor of shape (batch, seq_len, 50257)

    Args:
        checkpoint_path: Path to your checkpoint.pt file
        device: Device to load onto ("cuda" or "cpu")

    Returns:
        model: nn.Module in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # If you save config alongside weights, load it:
    # config = checkpoint["config"]
    # model = TinyGPT(**config)
    # model.load_state_dict(checkpoint["model_state_dict"])

    # Simple case: checkpoint is just the state_dict
    model = TinyGPT()  # Use your config here
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


# --- Optional: quick sanity check ---

if __name__ == "__main__":
    print("Creating example model...")
    model = TinyGPT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Test forward pass
    dummy_input = torch.randint(0, 50257, (2, 1024))
    logits = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 1024, 50257), "Output shape mismatch!"
    print("Interface check passed.")

    # Save example checkpoint
    torch.save(model.state_dict(), "checkpoint.pt")
    print("Saved example checkpoint.pt")
