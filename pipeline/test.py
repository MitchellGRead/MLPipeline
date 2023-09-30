# flake8: noqa
import model_registration as mr
import typer

import wandb
from ml.api.data_handler_interface import DataHandlerInterface

app = typer.Typer()


@app.command()
def test(data_loc: str = "./data/complex_physics/", for_model: str = "Complex_Physics_GNS_Model"):

    wandb.login()


if __name__ == "__main__":
    app()

    """machine api.wandb.ai
  login user
  password 6a4ead1fd99902098b517872824c1375f88b40fe
    """
