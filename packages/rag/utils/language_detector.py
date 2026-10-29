"""
语种检测模块 — 自动识别中/英/德文
"""
from langdetect import detect, DetectorFactory
from utils.logger import log

# 固定随机种子，保证检测结果一致
DetectorFactory.seed = 0

# 支持的语言
SUPPORTED_LANGUAGES = {"zh-cn": "zh", "zh-tw": "zh", "en": "en", "de": "de"}


def detect_language(text: str) -> str:
    """
    检测文本语种，返回 "zh" | "en" | "de"
    默认返回 "en"（无法识别时）
    """
    if not text or len(text.strip()) < 10:
        return "en"
    try:
        lang = detect(text)
        return SUPPORTED_LANGUAGES.get(lang, "en")
    except Exception:
        log.warning("语种检测失败，默认使用 en")
        return "en"
