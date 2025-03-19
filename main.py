from typing import List, Literal, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit
from cellpose import models
import os

custom_model_path = "/models"
custom_models = [f.name for f in os.scandir(custom_model_path)]
base_models = ["cyto", "nuclei", "cyto2"]
models_list = base_models
if len(custom_models) > 0:
    print(f"Found custom models: {custom_models}")
    models_list += custom_models
print(f"Available models: {models_list}")
models_list = Literal[tuple(models_list)]


class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D).",
        json_schema_extra={"widget_type": "image"},
    )
    model_name: models_list = Field(
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

        self.last_model_name = None
        self.last_model = None
        self.last_diameter = None
        self.last_flow_threshold = None
        self.last_cellprob_threshold = None

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
        if diameter == 0:
            diameter = None
            print(
                "Diameter is set to None. The size of the cells will be estimated on a per image basis"
            )

        if model_name != self.last_model_name or diameter != self.last_diameter or flow_threshold != self.last_flow_threshold or cellprob_threshold != self.last_cellprob_threshold:
            if model_name in custom_models:
                model_path = os.path.join(custom_model_path, model_name)
                print(f"Loading custom model from {model_path}")
                model = models.CellposeModel(gpu = True, model_type= model_path)
            else: 
                model = models.Cellpose(
                    gpu=True,
                    model_type=model_name,
                )
            self.last_model_name = model_name
            self.last_model = model
            self.last_diameter = diameter
            self.last_flow_threshold = flow_threshold
            self.last_cellprob_threshold = cellprob_threshold
        else:
            print("Using cached model")
            model = self.last_model

        segmentation, flows, styles, diams = model.eval(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            channels=[0, 0],  # Grayscale image only (for now)
        )

        segmentation_params = {"name": "Cellpose result"}
        
        return [
            (segmentation, segmentation_params, "instance_mask"),
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