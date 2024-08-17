from pydantic import BaseModel

class UserImage(BaseModel):
    image: str  # base64 encoded image

class GameState(BaseModel):
    user_id: str
    game_image: str  # base64 encoded game result image