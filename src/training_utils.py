import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_in_cpu(
    model, train_loader, optimizer, num_epochs, loss_fn, debug=False, plot_eval=False
):
    """
    Training function for the GraphSAGE model with unsupervised loss.
    This function tracks the loss evolution per batch and per epoch,
    and then plots the loss evolution at the end of training.

    Parameters:
        model: Instance of the GraphSAGE model.
        train_loader: DataLoader for training batches.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        loss_fn: Loss function.
        debug (bool): If True, prints additional debug information.
        plot_eval (bool): If True, plots training evaluation metrics (default: False).
    """
    # Set model in training mode
    model.train()

    # Lists for storing the losses
    epoch_loss_history = []  # Average loss per epoch
    batch_loss_history = []  # Loss for each batch over the entire training

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")

    for epoch in epoch_pbar:
        total_loss = 0.0
        all_embeddings = []

        # Batch progress bar for each epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in batch_pbar:
            if debug:
                print(
                    f"[DEBUG] Batch shapes - batch.x: {batch.x.shape}, batch.edge_index: {batch.edge_index.shape}"
                )

            # Clear gradients from the previous iteration
            optimizer.zero_grad()

            # Forward pass: obtain embeddings for the batch
            z = model(batch.x, batch.edge_index)
            if debug:
                print(f"[DEBUG] Model output (z) shape: {z.shape}")
                print(f"[DEBUG] First 5 embeddings: {z[:5]}")
                if torch.isnan(z).any():
                    print("[DEBUG] Detected NaN values in embeddings")

            # Compute loss using the unsupervised loss function
            loss = loss_fn(z, batch.edge_index)
            if debug:
                print(f"[DEBUG] Loss value for this batch: {loss.item():.4f}")
                if torch.isnan(loss):
                    print("[DEBUG] Loss is NaN in this batch!")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss and store embeddings for further analysis
            loss_value = loss.item()
            total_loss += loss_value
            batch_loss_history.append(loss_value)
            all_embeddings.append(z.detach())

            # Update batch progress bar with the current batch loss
            batch_pbar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
            if debug:
                print(f"[DEBUG] Accumulated loss so far: {total_loss:.4f}")

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        epoch_loss_history.append(avg_loss)

        # Analyze embedding statistics for the epoch (optional)
        with torch.no_grad():
            if debug:
                print(f"[DEBUG] Number of embeddings collected: {len(all_embeddings)}")
            epoch_embeddings = torch.cat(all_embeddings, dim=0)
            if debug:
                print(f"[DEBUG] Epoch embeddings shape: {epoch_embeddings.shape}")
            norms = torch.norm(epoch_embeddings, dim=1)
            if debug:
                print(
                    f"[DEBUG] Embeddings norms statistics -> "
                    f"min: {norms.min().item():.4f}, max: {norms.max().item():.4f}, "
                    f"mean: {norms.mean().item():.4f}, std: {norms.std().item():.4f}"
                )
            mean_norm = norms.mean().item()
            std_norm = norms.std().item()

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            {
                "avg_loss": f"{avg_loss:.4f}",
                "mean_norm": f"{mean_norm:.4f}",
                "std_norm": f"{std_norm:.4f}",
            }
        )

    if plot_eval:
        # Plot the loss evolution after training
        plt.figure(figsize=(12, 5))

        # Plot average loss per epoch
        plt.subplot(1, 2, 1)
        plt.plot(
            range(1, num_epochs + 1), epoch_loss_history, marker="o", linestyle="-"
        )
        plt.title("Average Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # Plot loss per batch over the entire training
        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(batch_loss_history) + 1),
            batch_loss_history,
            marker=".",
            linestyle="-",
            color="orange",
        )
        plt.title("Loss per Batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Return the metrics in case further analysis is needed
    return {
        "epoch_loss_history": epoch_loss_history,
        "batch_loss_history": batch_loss_history,
    }


def train_in_gpu(
    model, train_loader, optimizer, num_epochs, loss_fn, debug=False, plot_eval=False
):
    """
    Training function for the GraphSAGE model with unsupervised loss on GPU.
    This function tracks the loss evolution per batch and per epoch,
    and then plots the loss evolution at the end of training.

    Parameters:
        model: Instance of the GraphSAGE model.
        train_loader: DataLoader for training batches.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        loss_fn: Loss function.
        debug (bool): If True, prints additional debug information.
        plot_eval (bool): If True, plots training evaluation metrics (default: False).
    """
    # Set model in training mode
    model.train()

    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Lists for storing the losses
    epoch_loss_history = []  # Average loss per epoch
    batch_loss_history = []  # Loss for each batch over the entire training

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")

    # Iterate through epochs
    for epoch in epoch_pbar:
        total_loss = 0.0
        all_embeddings = []

        # Batch progress bar for each epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in batch_pbar:
            if debug:
                print(
                    f"[DEBUG] Batch shapes - batch.x: {batch.x.shape}, batch.edge_index: {batch.edge_index.shape}"
                )

            # Move batch to GPU
            batch = batch.to(device)

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass with autocast for mixed precision
            with autocast():
                z = model(batch.x, batch.edge_index)
                if debug:
                    print(f"[DEBUG] Model output (z) shape: {z.shape}")
                    print(f"[DEBUG] First 5 embeddings: {z[:5]}")
                    if torch.isnan(z).any():
                        print("[DEBUG] Detected NaN values in embeddings")

                # Compute loss using the unsupervised loss function
                loss = loss_fn(z, batch.edge_index)
                if debug:
                    print(f"[DEBUG] Loss value for this batch: {loss.item():.4f}")
                    if torch.isnan(loss):
                        print("[DEBUG] Loss is NaN in this batch!")

            # Backward pass and optimization with GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss and store embeddings for later analysis
            loss_value = loss.item()
            total_loss += loss_value
            batch_loss_history.append(loss_value)
            all_embeddings.append(z.detach().cpu())

            # Update batch progress bar with current batch loss
            batch_pbar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
            if debug:
                print(f"[DEBUG] Accumulated loss so far: {total_loss:.4f}")

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        epoch_loss_history.append(avg_loss)

        # Analyze embedding statistics for the epoch (optional)
        with torch.no_grad():
            if debug:
                print(f"[DEBUG] Number of embeddings collected: {len(all_embeddings)}")
            epoch_embeddings = torch.cat(all_embeddings, dim=0)
            if debug:
                print(f"[DEBUG] Epoch embeddings shape: {epoch_embeddings.shape}")
            norms = torch.norm(epoch_embeddings, dim=1)
            if debug:
                print(
                    f"[DEBUG] Embeddings norms statistics -> "
                    f"min: {norms.min().item():.4f}, max: {norms.max().item():.4f}, "
                    f"mean: {norms.mean().item():.4f}, std: {norms.std().item():.4f}"
                )
            mean_norm = norms.mean().item()
            std_norm = norms.std().item()

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            {
                "avg_loss": f"{avg_loss:.4f}",
                "mean_norm": f"{mean_norm:.4f}",
                "std_norm": f"{std_norm:.4f}",
            }
        )

    if plot_eval:
        # Plot the loss evolution after training
        plt.figure(figsize=(12, 5))

        # Plot average loss per epoch
        plt.subplot(1, 2, 1)
        plt.plot(
            range(1, num_epochs + 1), epoch_loss_history, marker="o", linestyle="-"
        )
        plt.title("Average Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)

        # Plot loss per batch over the entire training
        plt.subplot(1, 2, 2)
        plt.plot(
            range(1, len(batch_loss_history) + 1),
            batch_loss_history,
            marker=".",
            linestyle="-",
            color="orange",
        )
        plt.title("Loss per Batch")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # Return the metrics in case further analysis is needed
    return {
        "epoch_loss_history": epoch_loss_history,
        "batch_loss_history": batch_loss_history,
    }
