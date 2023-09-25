import model_registration as mr
import typer

from ml.api.data_handler_interface import DataHandlerInterface

app = typer.Typer()


@app.command()
def test(data_loc: str = "./data/complex_physics/", for_model: str = "Complex_Physics_GNS_Model"):
    data_handler = mr.getDataHandler(for_model)(data_loc)
    assert isinstance(data_handler, DataHandlerInterface)

    data_handler.get_data().take_batch(50, batch_format="default")
    # print(data.show())
    # print(data["data"])
    # print(type(data["data"]))
    # print(data["data"][1])

    data_handler.get(0)
    # print(data_handler.get(0))


if __name__ == "__main__":
    app()
