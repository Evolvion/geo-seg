from rich.console import Console
from rich.table import Table


def get_console() -> Console:
    return Console()


def log_metrics(
    console: Console, epoch: int, train_loss: float, val_iou: float
) -> None:
    table = Table(title=f"Epoch {epoch}")
    table.add_column("train_loss", justify="right")
    table.add_column("val_iou", justify="right")
    table.add_row(f"{train_loss:.4f}", f"{val_iou:.4f}")
    console.print(table)
