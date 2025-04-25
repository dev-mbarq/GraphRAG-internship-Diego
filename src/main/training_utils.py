import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR


def train_in_cpu(
    model, train_loader, optimizer, num_epochs, loss_fn, debug=False, plot_eval=False
):
    """
    Training function for the GraphSAGE model with unsupervised loss.
    This function tracks the loss evolution per batch and per epoch, as well as
    the L2 norm (mean and std) of the generated embeddings per batch and per epoch.
    If plot_eval is True, it plots:
      - The loss evolution (average loss per epoch and loss per batch)
      - The embedding L2 norms (mean and std per epoch, and mean and std per batch)

    Parameters:
        model: Instance of the GraphSAGE model.
        train_loader: DataLoader for training batches.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        loss_fn: Loss function.
        debug (bool): If True, prints additional debug information.
        plot_eval (bool): If True, plots training evaluation metrics.
    """
    # Set model in training mode
    model.train()

    # Lists for storing loss values and norm metrics
    epoch_loss_history = []  # Average loss per epoch
    batch_loss_history = []  # Loss for each batch
    epoch_norm_mean_history = []  # Mean L2 norm per epoch
    epoch_norm_std_history = []  # Std L2 norm per epoch
    batch_norm_mean_history = []  # Mean L2 norm per batch
    batch_norm_std_history = []  # Std L2 norm per batch

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")

    for epoch in epoch_pbar:
        total_loss = 0.0
        all_embeddings = []  # To accumulate embeddings for epoch-level norm computation

        # Batch progress bar for the current epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in batch_pbar:
            if debug:
                print(
                    f"[DEBUG] Batch shapes - batch.x: {batch.x.shape}, batch.edge_index: {batch.edge_index.shape}"
                )

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass: get embeddings for the batch
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

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Accumulate loss and store the embeddings for later analysis
            loss_value = loss.item()
            total_loss += loss_value
            batch_loss_history.append(loss_value)
            all_embeddings.append(z.detach())

            # Compute batch-level L2 norm statistics from the embeddings
            batch_norms = torch.norm(z.detach(), dim=1)
            batch_mean_norm = batch_norms.mean().item()
            batch_std_norm = batch_norms.std().item()
            batch_norm_mean_history.append(batch_mean_norm)
            batch_norm_std_history.append(batch_std_norm)

            # Update batch progress bar with current batch loss
            batch_pbar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
            if debug:
                print(f"[DEBUG] Accumulated loss so far: {total_loss:.4f}")

        # Compute epoch average loss
        avg_loss = total_loss / len(train_loader)
        epoch_loss_history.append(avg_loss)

        # Compute epoch-level L2 norm statistics by concatenating all embeddings
        with torch.no_grad():
            if debug:
                print(
                    f"[DEBUG] Number of embeddings collected in epoch: {len(all_embeddings)}"
                )
            epoch_embeddings = torch.cat(all_embeddings, dim=0)
            if debug:
                print(f"[DEBUG] Epoch embeddings shape: {epoch_embeddings.shape}")
            norms = torch.norm(epoch_embeddings, dim=1)
            if debug:
                print(
                    f"[DEBUG] Epoch norms -> min: {norms.min().item():.4f}, max: {norms.max().item():.4f}, "
                    f"mean: {norms.mean().item():.4f}, std: {norms.std().item():.4f}"
                )
            epoch_mean_norm = norms.mean().item()
            epoch_std_norm = norms.std().item()

            epoch_norm_mean_history.append(epoch_mean_norm)
            epoch_norm_std_history.append(epoch_std_norm)

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            {
                "avg_loss": f"{avg_loss:.4f}",
                "mean_norm": f"{epoch_mean_norm:.4f}",
                "std_norm": f"{epoch_std_norm:.4f}",
            }
        )

    # Plot evaluations if requested
    if plot_eval:
        loss_fig, norm_fig = plot_training_evolution(
            num_epochs=num_epochs,
            epoch_loss_history=epoch_loss_history,
            batch_loss_history=batch_loss_history,
            epoch_norm_mean_history=epoch_norm_mean_history,
            epoch_norm_std_history=epoch_norm_std_history,
            batch_norm_mean_history=batch_norm_mean_history,
            batch_norm_std_history=batch_norm_std_history,
        )

    # Return all collected metrics for further analysis if needed
    return {
        "epoch_loss_history": epoch_loss_history,
        "batch_loss_history": batch_loss_history,
        "epoch_norm_mean_history": epoch_norm_mean_history,
        "epoch_norm_std_history": epoch_norm_std_history,
        "batch_norm_mean_history": batch_norm_mean_history,
        "batch_norm_std_history": batch_norm_std_history,
        "loss_fig": loss_fig,
        "norm_fig": norm_fig,
    }


def train_in_gpu(
    model, train_loader, optimizer, num_epochs, loss_fn, debug=False, plot_eval=False
):
    """
    Training function for the GraphSAGE model with unsupervised loss on GPU.
    This function tracks the loss evolution per batch and per epoch, as well as the L2 norm
    (mean and std) of the generated embeddings per batch and per epoch.
    If plot_eval is True, it plots:
      - Loss evolution (average loss per epoch and loss per batch)
      - Embedding L2 norm evolution (mean and std plotted in two subplots: one for epochs and one for batches)

    Parameters:
        model: Instance of the GraphSAGE model.
        train_loader: DataLoader for training batches.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        loss_fn: Loss function.
        debug (bool): If True, prints additional debug information.
        plot_eval (bool): If True, plots training evaluation metrics.
    """
    # Set model in training mode
    model.train()

    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Lists for storing loss values and norm metrics
    epoch_loss_history = []  # Average loss per epoch
    batch_loss_history = []  # Loss for each batch over the entire training
    epoch_norm_mean_history = []  # Mean L2 norm per epoch
    epoch_norm_std_history = []  # Std L2 norm per epoch
    batch_norm_mean_history = []  # Mean L2 norm per batch
    batch_norm_std_history = []  # Std L2 norm per batch

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")

    for epoch in epoch_pbar:
        total_loss = 0.0
        all_embeddings = []  # To aggregate embeddings for epoch-level norm metrics

        # Batch progress bar for the current epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in batch_pbar:
            if debug:
                print(
                    f"[DEBUG] Batch shapes - batch.x: {batch.x.shape}, batch.edge_index: {batch.edge_index.shape}"
                )

            # Move batch to GPU
            batch = batch.to(device)

            # Clear gradients from the previous iteration
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
            # Save embeddings to CPU for further analysis
            all_embeddings.append(z.detach().cpu())

            # Compute batch-level L2 norm statistics for current batch embeddings
            batch_norms = torch.norm(z.detach(), dim=1)
            batch_mean_norm = batch_norms.mean().item()
            batch_std_norm = batch_norms.std().item()
            batch_norm_mean_history.append(batch_mean_norm)
            batch_norm_std_history.append(batch_std_norm)

            # Update batch progress bar with the current batch loss
            batch_pbar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
            if debug:
                print(f"[DEBUG] Accumulated loss so far: {total_loss:.4f}")

        # Compute average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        epoch_loss_history.append(avg_loss)

        # Compute epoch-level L2 norm statistics by concatenating all embeddings from the epoch
        with torch.no_grad():
            if debug:
                print(
                    f"[DEBUG] Number of embeddings collected in epoch: {len(all_embeddings)}"
                )
            epoch_embeddings = torch.cat(all_embeddings, dim=0)
            if debug:
                print(f"[DEBUG] Epoch embeddings shape: {epoch_embeddings.shape}")
            norms = torch.norm(epoch_embeddings, dim=1)
            if debug:
                print(
                    f"[DEBUG] Epoch norms statistics -> min: {norms.min().item():.4f}, max: {norms.max().item():.4f}, "
                    f"mean: {norms.mean().item():.4f}, std: {norms.std().item():.4f}"
                )
            epoch_mean_norm = norms.mean().item()
            epoch_std_norm = norms.std().item()
            epoch_norm_mean_history.append(epoch_mean_norm)
            epoch_norm_std_history.append(epoch_std_norm)

        # Update epoch progress bar with metrics
        epoch_pbar.set_postfix(
            {
                "avg_loss": f"{avg_loss:.4f}",
                "mean_norm": f"{epoch_mean_norm:.4f}",
                "std_norm": f"{epoch_std_norm:.4f}",
            }
        )

    # Plot evaluations if requested
    if plot_eval:
        loss_fig, norm_fig = plot_training_evolution(
            num_epochs=num_epochs,
            epoch_loss_history=epoch_loss_history,
            batch_loss_history=batch_loss_history,
            epoch_norm_mean_history=epoch_norm_mean_history,
            epoch_norm_std_history=epoch_norm_std_history,
            batch_norm_mean_history=batch_norm_mean_history,
            batch_norm_std_history=batch_norm_std_history,
        )

    # Return the metrics in case further analysis is needed
    return {
        "epoch_loss_history": epoch_loss_history,
        "batch_loss_history": batch_loss_history,
        "epoch_norm_mean_history": epoch_norm_mean_history,
        "epoch_norm_std_history": epoch_norm_std_history,
        "batch_norm_mean_history": batch_norm_mean_history,
        "batch_norm_std_history": batch_norm_std_history,
        "loss_fig": loss_fig,
        "norm_fig": norm_fig,
    }


# def train_model_in_gpu_V2(
#    model,
#    train_loader,
#    optimizer,
#    num_epochs: int,
#    loss_fn,
#    warmup_steps: int,
#    total_steps: int,
#    warmup_start_lr: float = 0.0,
#    warmup_end_lr: float = None,
#    min_lr: float = 1e-6,
#    max_grad_norm: float = 1.0,
#    debug: bool = False,
#    plot_eval: bool = False
# ):
#    """
#    Version 2 of the training loop with:
#      - Linear warm‑up of the LR
#      - Cosine‑annealing decay of the LR
#      - Gradient clipping
#      - Fixed scheduler ordering and non‑zero warmup_start_lr
#    """
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model.to(device).train()
#    scaler = GradScaler()
#
#    # If no explicit warmup_end_lr, use the optimizer's initial LR
#    if warmup_end_lr is None:
#        warmup_end_lr = optimizer.param_groups[0]["lr"]
#
#    # Ensure warmup_start_lr is strictly > 0
#    if warmup_start_lr <= 0.0:
#        # pick a small fraction (0.1%) of warmup_end_lr, but at least 1e-6
#        warmup_start_lr = max(min_lr, warmup_end_lr * 1e-3)
#
#    # Build schedulers
#    warmup_scheduler = LinearLR(
#        optimizer,
#        start_factor=warmup_start_lr / warmup_end_lr,  # must be in (0,1]
#        end_factor=1.0,
#        total_iters=warmup_steps
#    )
#    decay_scheduler = CosineAnnealingLR(
#        optimizer,
#        T_max=max(total_steps - warmup_steps, 1),
#        eta_min=min_lr
#    )
#
#    # Prepare histories
#    epoch_loss_history      = []
#    batch_loss_history      = []
#    epoch_norm_mean_history = []
#    epoch_norm_std_history  = []
#    batch_norm_mean_history = []
#    batch_norm_std_history  = []
#
#    global_step = 0
#    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
#
#    for epoch in epoch_pbar:
#        total_loss    = 0.0
#        all_embeddings = []
#
#        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
#        for batch in batch_pbar:
#            batch = batch.to(device)
#            optimizer.zero_grad()
#
#            # Forward
#            with autocast():
#                z    = model(batch.x, batch.edge_index)
#                loss = loss_fn(z, batch.edge_index)
#
#            # Backward + gradient clipping + optimizer step
#            scaler.scale(loss).backward()
#            clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
#            scaler.step(optimizer)
#            scaler.update()
#
#            # **Scheduler step after optimizer.step()**
#            if global_step < warmup_steps:
#                warmup_scheduler.step()
#            else:
#                decay_scheduler.step()
#            global_step += 1
#
#            # Record metrics
#            loss_value = loss.item()
#            total_loss += loss_value
#            batch_loss_history.append(loss_value)
#
#            norms = torch.norm(z.detach(), dim=1)
#            batch_norm_mean_history.append(norms.mean().item())
#            batch_norm_std_history.append(norms.std().item())
#
#            batch_pbar.set_postfix({"batch_loss": f"{loss_value:.4f}"})
#            all_embeddings.append(z.detach().cpu())
#
#        # Epoch‑level metrics
#        avg_loss = total_loss / len(train_loader)
#        epoch_loss_history.append(avg_loss)
#
#        with torch.no_grad():
#            epoch_emb = torch.cat(all_embeddings, dim=0)
#            norms     = torch.norm(epoch_emb, dim=1)
#            epoch_norm_mean_history.append(norms.mean().item())
#            epoch_norm_std_history.append(norms.std().item())
#
#        epoch_pbar.set_postfix({
#            "avg_loss":  f"{avg_loss:.4f}",
#            "mean_norm": f"{epoch_norm_mean_history[-1]:.4f}",
#            "std_norm":  f"{epoch_norm_std_history[-1]:.4f}"
#        })
#
#    return {
#        "epoch_loss_history":      epoch_loss_history,
#        "batch_loss_history":      batch_loss_history,
#        "epoch_norm_mean_history": epoch_norm_mean_history,
#        "epoch_norm_std_history":  epoch_norm_std_history,
#        "batch_norm_mean_history": batch_norm_mean_history,
#        "batch_norm_std_history":  batch_norm_std_history,
#    }


def plot_training_evolution(
    num_epochs,
    epoch_loss_history,
    batch_loss_history,
    epoch_norm_mean_history,
    epoch_norm_std_history,
    batch_norm_mean_history,
    batch_norm_std_history,
):
    """
    Plot training evolution metrics and return the figures for saving.
    Returns:
        tuple: (loss_fig, norm_fig) containing both matplotlib figures
    """
    # Loss Evolution Plots
    loss_fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_loss_history, marker="o", linestyle="-")
    plt.title("Average Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

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

    # Embedding Norm Evolution Plots
    norm_fig = plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, num_epochs + 1),
        epoch_norm_mean_history,
        marker="o",
        linestyle="-",
        label="Mean Norm",
    )
    plt.plot(
        range(1, num_epochs + 1),
        epoch_norm_std_history,
        marker="o",
        linestyle="--",
        label="Std Norm",
    )
    plt.title("Embedding Norms per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(batch_norm_mean_history) + 1),
        batch_norm_mean_history,
        marker=".",
        linestyle="-",
        label="Mean Norm",
    )
    plt.plot(
        range(1, len(batch_norm_std_history) + 1),
        batch_norm_std_history,
        marker=".",
        linestyle="--",
        label="Std Norm",
    )
    plt.title("Embedding Norms per Batch")
    plt.xlabel("Batch")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return loss_fig, norm_fig
