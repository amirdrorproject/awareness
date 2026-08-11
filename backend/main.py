from datetime import datetime, timezone

from fastapi import FastAPI

app = FastAPI()


@app.get("/time")
def get_time():
    return {"time": datetime.now(timezone.utc).isoformat()}
