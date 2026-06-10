import urllib.parse
import re

def normalize_entity_id(eid) -> str:
    """
    统一实体 ID 规范化：
    - 针对 WDC-LSPM: 纯数字 ID 统一转为字符串并 strip
    - 针对 URL: 截断为最后一段
    """
    if eid is None:
        return "nil"
        
    # 1. 强制转字符串
    text = str(eid).strip()
    
    # 2. 处理 NIL
    if text.lower() in ["nil", "none", "null", ""]:
        return "nil"
        
    # 3. 处理 WDC 数字 ID (这是关键！)
    # 如果看起来像纯数字，直接返回，不做 URL 处理
    if text.isdigit():
        return text
        
    # 4. 处理 Wiki/URL 前缀
    if text.startswith("wiki:"):
        text = text[len("wiki:"):]
        text = urllib.parse.unquote(text)
    
    # 去 URL 前缀 (http://...)
    if "://" in text:
        text = text.split("/")[-1].split(":")[-1]
    
    # 清理括号
    if "(" in text:
        text = text.split("(")[0]
        
    # 统一格式
    text = text.replace("_", " ").strip().lower()
    
    # 压缩空格
    text = re.sub(r"\s+", " ", text).strip()
    
    return text
