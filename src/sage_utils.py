#!/usr/bin/env python
import torch

def get_new_sage_embedding(model, new_feature, device="cuda", self_loop=True):
    """
    Generate a new node embedding using the trained GraphSAGE model.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained GraphSAGE model.
    new_feature : torch.Tensor
        The new node's initial features of shape (1024,).
    device : torch.device
        The device (e.g., cuda) on which the model is located.
    self_loop : bool, optional
        If True, adds a self-loop edge (node connected to itself) to simulate
        neighbor aggregation when no neighbors exist (default is True).

    Returns:
    --------
    torch.Tensor
        The new node's embedding of shape (out_channels,).
    """
    # Set the model to evaluation mode.
    model.eval()
    
    # Move the new node's features to the specified device.
    new_feature = new_feature.to(device)
    
    if self_loop:
        # Create a self-loop edge_index. This indicates that the node is connected to itself.
        # The edge_index tensor must have shape [2, num_edges]; here we create a single edge (0,0).
        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
    else:
        # Alternatively, if you prefer no edges, create an empty edge_index.
        # Note: Without a self-loop, the model might not transform the features as intended.
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    
    # GraphSAGE expects a batch dimension, so unsqueeze new_feature to shape [1, 1024].
    with torch.no_grad():
        new_embedding = model(new_feature.unsqueeze(0), edge_index)
    
    # Remove the batch dimension to return a tensor of shape [out_channels].
    return new_embedding.squeeze(0)

## Example usage:
#if __name__ == "__main__":
#    # Create a dummy GraphSAGE model for demonstration.
#    class DummyGraphSAGE(torch.nn.Module):
#        def __init__(self):
#            super().__init__()
#            self.lin = torch.nn.Linear(1024, 256)
#        def forward(self, x, edge_index):
#            return self.lin(x)
#    
#    model = DummyGraphSAGE()
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    model = model.to(device)
#    
#    # Create a dummy new node feature vector (replace with your actual new feature)
#    new_feature = torch.randn(1024)
#    
#    # Get the new node embedding using the function.
#    new_node_embedding = get_new_node_embedding(model, new_feature, device, self_loop=True)
#    print("New node embedding shape:", new_node_embedding.shape)
