import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi import FastAPI, File, UploadFile

from models import UserImage, GameState
from image_process import make_bald_advanced, make_bald_jinu, merge_images
import uuid

router = APIRouter()
user_images = {}

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@router.post("/start_game")
async def start_game(request: Request, user_image: UserImage):
    try:
        body = await request.body()
        logger.debug(f"Received request body: {body}")
        logger.debug(f"Parsed UserImage: {user_image}")

        if not user_image.image:
            raise ValueError("Image data is missing")

        bald_image = make_bald_advanced(user_image.image)
        user_id = str(uuid.uuid4())
        user_images[user_id] = bald_image
        return {"user_id": user_id, "bald_image": bald_image}
    except ValueError as e:
        logger.error(f"ValueError in start_game: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in start_game: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
@router.post("/start_game2")
async def start_game(request: Request,file: UploadFile | None = None):
    print(file)
    # file: UploadFile = request.files['image']
    
    try:
        # body = await request.body()

        if not file: 
            logger.debug(f"Received request body: {file.filename}")
            raise ValueError("Image data is missing")

        # bald_image_path = make_bald_jinu(file, file.filename)
        # user_id = str(uuid.uuid4())
        # user_images[user_id] = bald_image_path
        # return {"user_id": user_id, "bald_image": bald_image_path}
        base64encoded = make_bald_jinu(file, file.filename)
        user_id = str(uuid.uuid4())
        # user_images[user_id] = bald_image_path
        return {"user_id": user_id, "bald_image": base64encoded}
    except ValueError as e:
        logger.error(f"ValueError in start_game: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in start_game: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@router.post("/finish_game")
async def finish_game(request: Request, game_state: GameState):
    try:
        body = await request.body()
        logger.debug(f"Received request body: {body}")
        logger.debug(f"Parsed GameState: {game_state}")

        if game_state.user_id not in user_images:
            raise HTTPException(status_code=404, detail="User not found")
    
        bald_image_base64 = user_images[game_state.user_id]
        game_image_base64 = game_state.game_image

        final_image_base64 = merge_images(bald_image_base64, game_image_base64)
        return {"final_image": final_image_base64}
    except ValueError as e:
        logger.error(f"ValueError in finish_game: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in finish_game: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")