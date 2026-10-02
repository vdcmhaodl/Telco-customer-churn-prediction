from typing import Literal

from pydantic import(
    BaseModel,
    ConfigDict,
    Field,
)

YesNo = Literal["Yes", "No"]

InternetAddon = Literal[
    "Yes",
    "No",
    "No internet service",
]

class CustomerFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )
    
    gender: Literal[
        "Female",
        "Male",
    ]
    
    SeniorCitizen: Literal[0, 1]
    
    Partner: YesNo
    Dependents: YesNo
    
    tenure: int = Field(
        ge=0
    )
    
    PhoneService: YesNo
    
    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]
    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]
    
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    
    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]
    
    PaperlessBilling: YesNo
    
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    
    MonthlyCharges: float = Field(
        ge=0
    )
    TotalCharges: float = Field(
        ge=0
    )
    
class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int