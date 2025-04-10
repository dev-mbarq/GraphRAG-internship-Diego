import torch
from torch.cuda.amp import autocast, GradScaler
from loss_functions import unsupervised_loss
from tqdm import tqdm


def train_in_cpu(model, train_loader, optimizer, num_epochs, loss_fn, debug=False):
    """
    Training function for the GraphSAGE model with unsupervised loss.

    Parameters:
        model: Instance of the GraphSAGE model.
        train_loader: Loader for training batches.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        loss_fn: Loss function.
        debug (bool): If True, prints additional debug information.
    """
    # Set model in training mode
    model.train()

    # Initialize progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")

    # Iterate through epochs
    for epoch in epoch_pbar:
        # Initialize metrics for this epoch
        total_loss = 0.0
        all_embeddings = []

        # Create progress bar for batches within this epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        # Iterate through batches
        for batch in batch_pbar:
            if debug:
                print(
                    f"[DEBUG] Batch shapes - batch.x: {batch.x.shape}, batch.edge_index: {batch.edge_index.shape}"
                )

            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Forward pass: get embeddings for the batch
            z = model(batch.x, batch.edge_index)
            if debug:
                print(f"[DEBUG] Model output (z) shape: {z.shape}")
                # Print the first 5 embeddings to inspect
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

            # Accumulate loss and store embeddings for later analysis
            total_loss += loss.item()
            all_embeddings.append(z.detach())

            # Update batch progress bar with current batch loss
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            if debug:
                print(f"[DEBUG] Accumulated loss so far: {total_loss:.4f}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Analyze embedding statistics for the epoch
        with torch.no_grad():
            if debug:
                print(f"[DEBUG] Number of embeddings collected: {len(all_embeddings)}")
            # Concatenate all embeddings from this epoch
            epoch_embeddings = torch.cat(all_embeddings, dim=0)
            if debug:
                print(f"[DEBUG] Epoch embeddings shape: {epoch_embeddings.shape}")
            # Calculate L2 norms of embeddings
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


def train_in_gpu(model, train_loader, optimizer, num_epochs, loss_fn, debug=False):
    """
    Training function for the GraphSAGE model with unsupervised loss on GPU.

    Parameters:
        model: Instance of the GraphSAGE model.
        train_loader: Loader for training batches.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
        loss_fn: Loss function.
        debug (bool): If True, prints additional debug information.
    """
    # Set model in training mode
    model.train()

    # Move model to GPU
    device = torch.device("cuda")
    model.to(device)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Initialize progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Training epochs")

    # Iterate through epochs
    for epoch in epoch_pbar:
        # Initialize metrics for this epoch
        total_loss = 0.0
        all_embeddings = []

        # Create progress bar for batches within this epoch
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        # Iterate through batches
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
                    # Print the first 5 embeddings to inspect
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
            total_loss += loss.item()
            all_embeddings.append(z.detach().cpu())

            # Update batch progress bar with current batch loss
            batch_pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
            if debug:
                print(f"[DEBUG] Accumulated loss so far: {total_loss:.4f}")

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)

        # Analyze embedding statistics for the epoch
        with torch.no_grad():
            if debug:
                print(f"[DEBUG] Number of embeddings collected: {len(all_embeddings)}")
            # Concatenate all embeddings from this epoch
            epoch_embeddings = torch.cat(all_embeddings, dim=0)
            if debug:
                print(f"[DEBUG] Epoch embeddings shape: {epoch_embeddings.shape}")
            # Calculate L2 norms of embeddings
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
