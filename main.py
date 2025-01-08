from typing import List, Literal, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit
from cellpose import models

class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D).",
        json_schema_extra={"widget_type": "image"},
    )
    model_name: Literal["cyto", "nuclei", "cyto2"] = Field(
        default="cyto",
        title="Model",
        description="The model used for instance segmentation",
        json_schema_extra={"widget_type": "dropdown"},
    )
    diameter: int = Field(
        default=20,
        title="Cell diameter (px)",
        description="The approximate size of the objects to detect",
        ge=0,
        le=100,
        json_schema_extra={
            "widget_type": "int",
            "step": 1,
        },
    )
    flow_threshold: float = Field(
        default=0.3,
        title="Flow threshold",
        description="The flow threshold",
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.05,
        },
    )
    cellprob_threshold: float = Field(
        default=0.5,
        title="Probability threshold",
        description="The detection probability threshold",
        ge=0.0,
        le=1.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 0.01,
        },
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim != 2:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array

# Define the run_algorithm() method for your algorithm
class Server(serverkit.Server):
    def __init__(
        self,
        algorithm_name: str="cellpose",
        parameters_model: Type[BaseModel]=Parameters
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        image: np.ndarray,
        model_name: str,
        diameter: int,
        flow_threshold: float,
        cellprob_threshold: float,
        **kwargs,
    ) -> List[tuple]:
        """Runs the algorithm."""
        model = models.Cellpose(
            gpu=False,  # For now
            model_type=model_name,
        )

        if diameter == 0:
            diameter = None
            print(
                "Diameter is set to None. The size of the cells will be estimated on a per image basis"
            )

        segmentation, flows, styles, diams = model.eval(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0],  # Grayscale image only (for now)
        )

        segmentation_params = {"name": "Cellpose result"}
        
        return [
            (segmentation, segmentation_params, "labels"),
        ]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Load one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images

server = Server()
app = server.app

if __name__=='__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000)