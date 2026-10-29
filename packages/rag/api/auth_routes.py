"""局域网控制台登录与会话接口。"""
from __future__ import annotations
import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from utils.access_control import AccessStore, AuthorizationError
from config import config

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreateRequest(BaseModel):
    username: str
    password: str
    role: str


def _bearer(authorization: str = Header(default="")) -> str:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="缺少 Bearer 会话令牌")
    return authorization[len(prefix):]


def _automation_identity(token: str) -> dict | None:
    if config.ADMIN_API_KEY and hmac.compare_digest(token, config.ADMIN_API_KEY):
        return {"username": "automation", "role": "admin"}
    return None


def current_identity(request: Request, authorization: str = Header(default="")) -> dict:
    try:
        token = _bearer(authorization)
        return _automation_identity(token) or request.app.state.access_store.identity(token)
    except AuthorizationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def require_admin(request: Request, authorization: str = Header(default="")) -> dict:
    try:
        token = _bearer(authorization)
        return _automation_identity(token) or request.app.state.access_store.identity(token, "admin")
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def require_viewer(request: Request, authorization: str = Header(default="")) -> dict:
    """局域网模式下要求有效会话；本机模式保持既有无登录体验。"""
    if not config.LAN_ENABLED:
        return {}
    return current_identity(request, authorization)


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
    token = _bearer(authorization)
    if _automation_identity(token):
        raise HTTPException(status_code=400, detail="自动化 API Key 请通过移除或轮换环境变量撤销")
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
