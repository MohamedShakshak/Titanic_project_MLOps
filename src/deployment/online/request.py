from pydantic import BaseModel, Field, field_validator

class PassengerFeatures(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str
    Age: float = Field(..., gt=0, lt=120)
    SibSp: int = Field(default=0, ge=0)
    Parch: int = Field(default=0, ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str
    Name: str = Field(default="Unknown, Mr. Unknown")
    Ticket: str = Field(default="000000")
    Cabin: str = Field(default="")

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        v = v.lower()
        if v not in ("male", "female"):
            raise ValueError("Sex must be male or female")
        return v

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, v: str) -> str:
        v = v.upper()
        if v not in ("S", "C", "Q"):
            raise ValueError("Embarked must be S, C, or Q")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 1, "Sex": "female", "Age": 29.0,
                "SibSp": 0, "Parch": 0, "Fare": 211.3,
                "Embarked": "S", "Name": "Cumings, Mrs. John Bradley",
                "Ticket": "PC 17599", "Cabin": "C85",
            }
        }
    }
