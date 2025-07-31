import aiofiles
from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile
from app.config import settings

async def save_upload_file(user_id: str, batch: str, f: UploadFile) -> str:
    folder = settings.upload_dir / user_id / batch
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f.filename
    async with aiofiles.open(dest, "wb") as out:
        while chunk := await f.read(1 << 20):
            await out.write(chunk)
    return str(dest)
