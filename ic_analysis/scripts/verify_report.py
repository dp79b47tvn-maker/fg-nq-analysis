"""
發布 output/report.html 之前的自動檢查——這份報告是純靜態HTML（沒有<script>、
沒有注入的JS DATA變數，跟bond_data_pipeline的dashboard.html不是同一種結構），
所以檢查項目跟那邊的verify_dashboard.py不同，是針對這份報告實際會出錯的地方：

  1. 結構完整：<div>/</div>、<table>/</table> 數量要配對（這份報告改版時吃過
     div標籤沒配對的虧，整頁排版會壞掉但肉眼不一定馬上看得出來）。
  2. 沒有殘留的f-string沒填值（"None"字面值、沒被替換掉的"{變數名}"這種痕跡）。
  3. 每一張嵌入的base64圖表，解碼後真的是合法PNG（不是產圖失敗留下空字串或壞資料）。
  4. 必要章節標題都在（確認report.html不是只產出一半）。

用法：
    python3 scripts/verify_report.py
成功回傳 exit code 0，失敗印出具體原因並回傳非0，CI/發布前應該先跑這個腳本。
"""
import base64
import re
import sys
from pathlib import Path

REPORT_PATH = Path(__file__).resolve().parent.parent / "output" / "report.html"

REQUIRED_SECTIONS = [
    "摘要：全樣本結果",
    "策略回測",
    "不同持有期的訊號強度",
    "兩段子時期對照",
    "資料來源與可信度",
    "方法論",
    "結論",
]


def fail(msg):
    print(f"✗ {msg}")
    return False


def main():
    ok = True
    if not REPORT_PATH.exists():
        return fail(f"找不到 {REPORT_PATH}，report.html 可能還沒產生")
    html = REPORT_PATH.read_text(encoding="utf-8")

    # ---- 1. div / table 標籤配對 ----
    div_open, div_close = html.count("<div"), html.count("</div>")
    if div_open != div_close:
        ok = fail(f"<div> 數量不配對：開 {div_open} 個、關 {div_close} 個")
    else:
        print(f"✓ <div> 標籤配對（{div_open} 對）")

    table_open, table_close = html.count("<table>"), html.count("</table>")
    if table_open != table_close:
        ok = fail(f"<table> 數量不配對：開 {table_open} 個、關 {table_close} 個")
    else:
        print(f"✓ <table> 標籤配對（{table_open} 對）")

    # ---- 2. f-string殘留痕跡 ----
    if "None" in html:
        ok = fail("html裡出現字面值 'None'，可能有f-string套用到空值沒處理")
    else:
        print("✓ 沒有殘留的 'None' 字面值")

    if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_']*[\[\.]?", html.replace("{{", "").replace("}}", "")):
        # 粗略檢查：CSS的 {{ }} 已經排除，如果還抓到疑似變數名的花括號，可能是f-string漏填
        stray = re.findall(r"\{[a-zA-Z_][a-zA-Z0-9_'\[\].]{0,40}\}", html)
        if stray:
            ok = fail(f"疑似殘留未填值的f-string變數：{stray[:5]}")
        else:
            print("✓ 沒有明顯的f-string殘留痕跡")

    # ---- 3. 嵌入的base64圖表都是合法PNG ----
    imgs = re.findall(r"data:image/png;base64,([A-Za-z0-9+/=]+)'", html)
    bad = 0
    for b64 in imgs:
        try:
            data = base64.b64decode(b64)
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                bad += 1
        except Exception:
            bad += 1
    if bad:
        ok = fail(f"{bad} / {len(imgs)} 張嵌入圖表不是合法PNG")
    elif not imgs:
        ok = fail("html裡完全沒有嵌入圖表，report.html可能只產出了骨架")
    else:
        print(f"✓ {len(imgs)} 張嵌入圖表都是合法PNG")

    # ---- 4. 必要章節都在 ----
    missing = [s for s in REQUIRED_SECTIONS if s not in html]
    if missing:
        ok = fail(f"缺少章節：{missing}")
    else:
        print(f"✓ {len(REQUIRED_SECTIONS)} 個必要章節都在")

    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
