  1 from typing import Union, Optional
  2
  3 from fastapi import FastAPI
  4 from pydantic import BaseModel
  5 app = FastAPI()
  6
  7
  8 @app.get("/")
  9 def read_root():
 10     return {"Hello": "World"}
 11
 12
 13 @app.get("/items/{item_id}")
 14 def read_item(item_id: int, q: Union[str, None] = None):
 15     return {"item_id": item_id, "q": q}
 16
 17 # New models and route
 18 class JoinRequest(BaseModel):
 19     site_id: str
 20     version: Optional[str] = None
 21
 22 class JoinReply(BaseModel):
 23     welcome: str
 24     round: int
 25
 26 @app.post("/join", response_model=JoinReply)
 27 def join(req: JoinRequest):
28     return JoinReply(welcome=f"Hello {req.site_id}", round=0    )
~
