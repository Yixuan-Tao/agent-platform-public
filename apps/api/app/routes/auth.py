"""平台网关的用户登录 / 会话 / 账号管理接口。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from agent_platform.core.security import AuthorizationError, current_identity, require_admin

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str


@router.post("/login")
def login(payload: LoginRequest, request: Request) -> dict:
    try:
        token = request.app.state.access_store.authenticate(payload.username, payload.password)
        return {"access_token": token, "token_type": "bearer"}
    except AuthorizationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me")
def me(identity: dict = Depends(current_identity)) -> dict:
    return identity


@router.post("/logout")
def logout(request: Request, authorization: str = Header(default="")) -> dict:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="缺少 Bearer 会话令牌")
    token = authorization[len(prefix):]
    request.app.state.access_store.revoke(token)
    return {"ok": True}


@router.get("/users", dependencies=[Depends(require_admin)])
def list_users(request: Request) -> dict:
    return {"users": request.app.state.access_store.list_users()}


@router.post("/users", dependencies=[Depends(require_admin)])
def create_user(payload: UserCreateRequest, request: Request) -> dict:
    try:
        request.app.state.access_store.create_user(payload.username, payload.password, payload.role)
        return {"username": payload.username, "role": payload.role}
    except (AuthorizationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/users/{username}", dependencies=[Depends(require_admin)])
def delete_user(username: str, request: Request) -> dict:
    try:
        request.app.state.access_store.delete_user(username)
        return {"ok": True}
    except AuthorizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
