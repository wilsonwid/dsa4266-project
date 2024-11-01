import torch
import torch.nn as nn
import datetime as dt
import argparse
from rcnn import RecurrentConvolutionalNetwork

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "rcnn"
NOW = dt.datetime.now()
FILENAME = f"{NOW.strftime("%Y-%m-%d-%H-%M-%S")}"
SAVE_DIR = 

def train_model(
    data: torch.Tensor,
    model: nn.Module,
    epochs: int,
    filename: str
):
    """
    Trains the model and saves the weights into a `.pt` file.

    Args:
        data (torch.Tensor): The data to train the model.
        model (nn.Module): The model to train.
        epochs (int): Number of epochs.
        filename (str): Filename to save the model to.

    Returns:

    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i in range(100):
            x = torch.randn(1, 5, 32, 32).to(device)
            y = torch.randint(0, 2, (1,)).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), f"{MODEL_NAME}_{filename}.pt")


def get_arguments() -> argparse.Namespace:
    """
    Parses the arguments for the training script.

    Args:
        None

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="train_rcnn",
        description="Trains the RCNN Model"
    )
    parser.add_argument("--input_dim", type=int, default=5,
                        help="Input dimension")
    parser.add_argument("--num_rec_layers", type=int, default=5, 
                        help="Number of recurrent layers")
    parser.add_argument("--num_kernels", type=int, default=128,
                        help="Number of kernels in each RCL layer")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="Kernel size (as int)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride (as int)")
    parser.add_argument("--padding", type=int, default=1,
                        help="Padding (as int)")
    parser.add_argument("--dropout_prob", type=float, default=0.25,
                        help="Dropout probability")
    parser.add_argument("--nonlinearity", type=str, default="relu",
                        help="Nonlinearity function (as str)")
    parser.add_argument("--bias", type=bool, default=False,
                        help="Bias (as bool)")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of steps")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--filename", type=str, default=FILENAME,
                        help="Filename to save the model")

    return parser.parse_args()
    


if __name__ == "__main__":
    args = get_arguments()
    model = RecurrentConvolutionalNetwork(
        input_channels=args.input_dim,
        num_recurent_layers=args.num_rec_layers,
        num_kernels=args.num_kernels,
        kernel_size=args.kernel_size,
        stride=args.stride,
        padding=args.padding,
        dropout_prob=args.dropout_prob,
        nonlinearity=args.nonlinearity,
        bias=args.bias,
        steps=args.steps,
        num_classes=args.num_classes
    )
    
    train_model(model, args.epochs, args.filename)
