import typer

from .predict import predict

app = typer.Typer()
app.command()(predict)

if __name__ == "__main__":
    app()