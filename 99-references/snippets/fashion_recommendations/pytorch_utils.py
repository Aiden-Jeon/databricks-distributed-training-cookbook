"""
PyTorch utilities for LSTM recommendation model
Includes model architecture, training, and inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import mlflow

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pl = None


class LSTMRecommender(nn.Module):
    """
    LSTM-based recommendation model

    Architecture:
    - Embedding layer for article IDs
    - 2-layer LSTM with dropout
    - Fully connected layers
    - Output: probability distribution over all articles
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0
    ):
        """
        Args:
            vocab_size: Size of article vocabulary
            embedding_dim: Dimension of article embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            padding_idx: Index for padding token
        """
        super(LSTMRecommender, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx

        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        input_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            input_seq: Input sequences [batch_size, seq_length]
            hidden: Hidden state tuple (h_n, c_n) or None

        Returns:
            output: Logits [batch_size, vocab_size]
            hidden: Final hidden state
        """
        # Embedding: [batch_size, seq_length, embedding_dim]
        embedded = self.embedding(input_seq)

        # LSTM: [batch_size, seq_length, hidden_dim]
        lstm_out, hidden = self.lstm(embedded, hidden)

        # Take the last output
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]

        # Fully connected layers
        x = F.relu(self.fc1(last_output))
        x = self.dropout1(x)
        logits = self.fc2(x)  # [batch_size, vocab_size]

        return logits, hidden

    def predict_top_k(
        self,
        input_seq: torch.Tensor,
        k: int = 12,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-K articles for input sequences

        Args:
            input_seq: Input sequences [batch_size, seq_length]
            k: Number of top predictions to return
            temperature: Temperature for softmax (lower = more confident)

        Returns:
            top_k_indices: Top K article indices [batch_size, k]
            top_k_scores: Scores for top K articles [batch_size, k]
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_seq)
            # Apply temperature scaling
            scaled_logits = logits / temperature
            # Get probabilities
            probs = F.softmax(scaled_logits, dim=-1)
            # Get top K
            top_k_scores, top_k_indices = torch.topk(probs, k, dim=-1)

        return top_k_indices, top_k_scores


class LSTMTrainer:
    """Trainer for LSTM recommendation model"""

    def __init__(
        self,
        model: LSTMRecommender,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Args:
            model: LSTMRecommender model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=model.padding_idx
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        log_interval: int = 100
    ) -> float:
        """
        Train for one epoch

        Args:
            train_loader: Training data loader
            log_interval: Log every N batches

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (input_seq, target, customer_ids) in enumerate(progress_bar):
            # Move to device
            input_seq = input_seq.to(self.device)
            target = target.to(self.device)

            # Forward pass
            logits, _ = self.model(input_seq)

            # Calculate loss
            # Target: [batch_size, num_targets], we'll use first target
            # For simplicity, predict the first next item
            target_first = target[:, 0]  # [batch_size]
            loss = self.criterion(logits, target_first)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate on validation set

        Args:
            val_loader: Validation data loader

        Returns:
            avg_loss: Average validation loss
            accuracy: Top-1 accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_seq, target, customer_ids in tqdm(val_loader, desc="Validation"):
                # Move to device
                input_seq = input_seq.to(self.device)
                target = target.to(self.device)

                # Forward pass
                logits, _ = self.model(input_seq)

                # Calculate loss
                target_first = target[:, 0]
                loss = self.criterion(logits, target_first)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == target_first).sum().item()
                total += target_first.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        mlflow_logging: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            mlflow_logging: Whether to log to MLflow

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_accuracy = self.evaluate(val_loader)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Log to MLflow
            if mlflow_logging:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_checkpoint("best_model.pt")
                if mlflow_logging:
                    mlflow.log_metric("best_val_loss", best_val_loss)
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epoch(s)")

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        # Load best model
        self.load_checkpoint("best_model.pt")

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class LSTMTrainerDistributed:
    """Distributed trainer for LSTM recommendation model using DDP"""

    def __init__(
        self,
        model: nn.Module,  # Already wrapped with DDP
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        rank: int = 0
    ):
        """
        Args:
            model: DDP-wrapped LSTMRecommender model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            rank: Process rank in distributed training
        """
        self.model = model
        self.device = device
        self.rank = rank
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Get padding_idx from the wrapped model
        padding_idx = model.module.padding_idx if hasattr(model, 'module') else model.padding_idx
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def train_epoch(
        self,
        train_loader: DataLoader,
        log_interval: int = 100
    ) -> float:
        """
        Train for one epoch (distributed)

        Args:
            train_loader: Training data loader with DistributedSampler
            log_interval: Log every N batches

        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Only show progress bar on rank 0
        iterator = tqdm(train_loader, desc="Training") if self.rank == 0 else train_loader

        for batch_idx, (input_seq, target, customer_ids) in enumerate(iterator):
            # Move to device
            input_seq = input_seq.to(self.device)
            target = target.to(self.device)

            # Forward pass
            logits, _ = self.model(input_seq)

            # Calculate loss
            target_first = target[:, 0]  # [batch_size]
            loss = self.criterion(logits, target_first)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            num_batches += 1

            if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches
                if hasattr(iterator, 'set_postfix'):
                    iterator.set_postfix({'loss': f'{avg_loss:.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float]:
        """
        Evaluate on validation set (distributed)

        Args:
            val_loader: Validation data loader with DistributedSampler

        Returns:
            avg_loss: Average validation loss
            accuracy: Top-1 accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        iterator = tqdm(val_loader, desc="Validation") if self.rank == 0 else val_loader

        with torch.no_grad():
            for input_seq, target, customer_ids in iterator:
                # Move to device
                input_seq = input_seq.to(self.device)
                target = target.to(self.device)

                # Forward pass
                logits, _ = self.model(input_seq)

                # Calculate loss
                target_first = target[:, 0]
                loss = self.criterion(logits, target_first)
                total_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == target_first).sum().item()
                total += target_first.size(0)

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        early_stopping_patience: int = 3,
        mlflow_logging: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping (distributed)

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            early_stopping_patience: Stop if no improvement for N epochs
            mlflow_logging: Whether to log to MLflow (only rank 0)

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
            if self.rank == 0:
                print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            if hasattr(val_loader.sampler, 'set_epoch'):
                val_loader.sampler.set_epoch(epoch)

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_accuracy = self.evaluate(val_loader)

            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Log to MLflow (only rank 0)
            if mlflow_logging and self.rank == 0:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            if self.rank == 0:
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print(f"Val Accuracy: {val_accuracy:.4f}")

            # Early stopping (only on rank 0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if self.rank == 0 and mlflow_logging:
                    mlflow.log_metric("best_val_loss", best_val_loss)
            else:
                patience_counter += 1
                if self.rank == 0:
                    print(f"No improvement for {patience_counter} epoch(s)")

            if patience_counter >= early_stopping_patience:
                if self.rank == 0:
                    print(f"Early stopping after {epoch + 1} epochs")
                break

        return history


def generate_recommendations(
    model: LSTMRecommender,
    input_sequences: List[torch.Tensor],
    customer_ids: List[str],
    encoder,
    device: torch.device,
    k: int = 12,
    batch_size: int = 256
) -> Dict[str, List[str]]:
    """
    Generate recommendations for a list of customers

    Args:
        model: Trained LSTM model
        input_sequences: List of input sequence tensors
        customer_ids: List of customer IDs
        encoder: ArticleEncoder for decoding
        device: Device to run inference on
        k: Number of recommendations per customer
        batch_size: Batch size for inference

    Returns:
        Dictionary mapping customer_id to list of recommended article_ids
    """
    model.eval()
    recommendations = {}

    # Process in batches
    for i in range(0, len(input_sequences), batch_size):
        batch_seqs = input_sequences[i:i + batch_size]
        batch_customers = customer_ids[i:i + batch_size]

        # Stack sequences into batch
        batch_tensor = torch.stack(batch_seqs).to(device)

        # Get predictions
        with torch.no_grad():
            top_k_indices, top_k_scores = model.predict_top_k(batch_tensor, k=k)

        # Convert to article IDs
        for j, customer_id in enumerate(batch_customers):
            article_indices = top_k_indices[j].cpu().numpy()
            article_ids = encoder.decode_batch(article_indices.tolist())
            # Filter out padding tokens
            article_ids = [aid for aid in article_ids if aid != '<PAD>']
            recommendations[customer_id] = article_ids[:k]

    return recommendations


# PyTorch Lightning Module
if LIGHTNING_AVAILABLE:
    class LSTMRecommenderLightning(pl.LightningModule):
        """
        PyTorch Lightning wrapper for LSTM recommendation model
        Handles training, validation, and optimization in a clean interface
        """

        def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 64,
            hidden_dim: int = 128,
            num_layers: int = 2,
            dropout: float = 0.3,
            padding_idx: int = 0,
            learning_rate: float = 0.001,
            weight_decay: float = 1e-5
        ):
            """
            Args:
                vocab_size: Size of article vocabulary
                embedding_dim: Dimension of article embeddings
                hidden_dim: Hidden dimension of LSTM
                num_layers: Number of LSTM layers
                dropout: Dropout rate
                padding_idx: Index for padding token
                learning_rate: Learning rate for optimizer
                weight_decay: L2 regularization weight
            """
            super().__init__()
            self.save_hyperparameters()

            # Create the LSTM model
            self.model = LSTMRecommender(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                padding_idx=padding_idx
            )

            # Loss function
            self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

        def forward(self, input_seq, hidden=None):
            """Forward pass"""
            return self.model(input_seq, hidden)

        def training_step(self, batch, batch_idx):
            """Training step"""
            input_seq, target, customer_ids = batch

            # Forward pass
            logits, _ = self.model(input_seq)

            # Calculate loss (predict first next item)
            target_first = target[:, 0]
            loss = self.criterion(logits, target_first)

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == target_first).float().mean()

            # Log metrics
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return loss

        def validation_step(self, batch, batch_idx):
            """Validation step"""
            input_seq, target, customer_ids = batch

            # Forward pass
            logits, _ = self.model(input_seq)

            # Calculate loss
            target_first = target[:, 0]
            loss = self.criterion(logits, target_first)

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == target_first).float().mean()

            # Log metrics
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            return loss

        def configure_optimizers(self):
            """Configure optimizer"""
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
            return optimizer

        def predict_top_k(self, input_seq, k=12, temperature=1.0):
            """Predict top-K articles for input sequences"""
            return self.model.predict_top_k(input_seq, k=k, temperature=temperature)
