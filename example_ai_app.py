"""
example_ai_app.py
-----------------
Vi du thuc te ve cach tich hop TokenSaver vao ung dung goi AI.
Ho tro ca Google Gemini va OpenAI.

Cach chay:
    python example_ai_app.py
"""

import os
import sys

# Import cac module tu TokenSaver
from tokensaver import (
    TokenCounter,
    BudgetGuard,
    BudgetExceededError,
    PromptCompressor,
    HTMLCleaner,
    SessionLogger,
)


def run_demo():
    print("=" * 60)
    print("DEMO TICH HOP TOKENSAVER VAO UNG DUNG AI THUC TE")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. XAC DINH NHA CUNG CAP (GEMINI HOAC OPENAI)
    # -------------------------------------------------------------
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        provider = "gemini"
        model = "gemini-3.8-flash"
        print("-> Phat hien GEMINI_API_KEY: Su dung Google Gemini (gemini-3.8-flash)")
    elif openai_key:
        provider = "openai"
        model = "gpt-4o-mini"
        print("-> Phat hien OPENAI_API_KEY: Su dung OpenAI (gpt-4o-mini)")
    else:
        provider = "gemini"
        model = "gemini-3.8-flash"
        print("-> Chua tim thay API Key: Chay che do mo phong (Simulation)")

    # -------------------------------------------------------------
    # 2. THIET LAP NGAN SACH & THEO DOI (BUDGET & MONITORING)
    # -------------------------------------------------------------
    guard = BudgetGuard(
        session_limit_tokens=20_000,
        session_limit_usd=0.05,
        warn_at_percent=0.8,
    )

    counter = TokenCounter(provider=provider, model=model)
    compressor = PromptCompressor(mode="simple")
    logger = SessionLogger(tag="demo-app")

    # -------------------------------------------------------------
    # 3. PROMPT THO (CHUA TOI UU - NHIEU TOKEN RAC)
    # -------------------------------------------------------------
    raw_prompt = """
    // Day la doan code tinh tien don hang trong he thong
    // Ham nhan vao danh sach cac item
    function calculateTotal(items) {
        // khoi tao tong ban dau bang 0
        let total = 0;
        
        // lap qua tung phan tu trong mang
        for (let i = 0; i < items.length; i++) {
            // cong don gia tien
            total += items[i].price;
        }
        
        // tra ve ket qua
        return total;
    }
    
    Please note that it is important to note that I need an analysis.
    Of course, explain this code in Vietnamese and provide suggestions to optimize it.
    """

    print("\n[1] DANH GIA PROMPT BAN DAU:")
    raw_check = counter.count(raw_prompt)
    print(f"  - So tokens tho: {raw_check.tokens} tokens")
    print(f"  - Chi phi uoc tinh: ${raw_check.cost_usd:.6f}")

    # -------------------------------------------------------------
    # 4. TOI UU HOA PROMPT (NEN & LOC BO TOKEN RAC)
    # -------------------------------------------------------------
    compressed_res = compressor.compress(raw_prompt)
    optimized_prompt = compressed_res.compressed_text

    print("\n[2] SAU KHI NEN BANG TOKENSAVER:")
    print(f"  - Tokens sau khi nen: {compressed_res.compressed_tokens} tokens")
    print(f"  - Tiet kiem duoc: {compressed_res.saved_tokens} tokens ({compressed_res.saved_percent:.1f}%)")

    # -------------------------------------------------------------
    # 5. KIEM TRA NGAN SACH TRUOC KHI GOI API (PRE-FLIGHT CHECK)
    # -------------------------------------------------------------
    print("\n[3] KIEM TRA NGAN SACH (BUDGET GUARD)...")
    try:
        guard.check(
            estimated_input_tokens=compressed_res.compressed_tokens,
            provider=provider,
            model=model,
        )
        print("  -> Ngan sach hop le, cho phep gui toi AI.")
    except BudgetExceededError as e:
        print(f"  -> CANH BAO! Vuot qua ngan sach cho phep: {e}")
        return

    # -------------------------------------------------------------
    # 6. GOI API THUC TE
    # -------------------------------------------------------------
    print("\n[4] GUI TOI AI VA NHAN PHAN HOI THUC TE...")
    ai_reply = ""
    actual_in_tokens = compressed_res.compressed_tokens
    actual_out_tokens = 0

    if gemini_key:
        import warnings
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=gemini_key)

        # Gioi han output tokens de tiet kiem chi phi (co the tang len neu can)
        MAX_OUTPUT_TOKENS = 512

        # Danh sach cac model thu lan luot neu model chinh bi qua tai (503)
        candidate_models = [model, "gemini-2.5-flash", "gemini-1.5-flash"]
        called_success = False

        for candidate in candidate_models:
            try:
                print(f"  -> Dang ket noi toi model: {candidate}...")
                # An canh bao AFC khong can thiet cua Google SDK
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    response = client.models.generate_content(
                        model=candidate,
                        contents=optimized_prompt,
                        config=genai_types.GenerateContentConfig(
                            max_output_tokens=MAX_OUTPUT_TOKENS,
                        ),
                    )
                ai_reply = response.text
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    actual_in_tokens = response.usage_metadata.prompt_token_count or actual_in_tokens
                    actual_out_tokens = response.usage_metadata.candidates_token_count or 100
                else:
                    actual_out_tokens = counter.count(ai_reply).tokens
                print(f"  (Goi Google Gemini API [{candidate}] thanh cong!)")
                model = candidate
                called_success = True
                break
            except Exception as e:
                err_msg = str(e)
                if "503" in err_msg or "high demand" in err_msg or "404" in err_msg:
                    print(f"  [Tam thoi ban] {candidate} dang qua tai, thu model tiep theo...")
                    continue
                else:
                    ai_reply = f"Loi ket noi: {e}"
                    print(f"  (Loi khi goi Gemini: {e})")
                    break

        if not called_success and not ai_reply:
            ai_reply = "Cac model hien deu dang qua tai, vui long thu lai sau vai giay."
    elif openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": optimized_prompt}]
            )
            ai_reply = response.choices[0].message.content
            actual_in_tokens = response.usage.prompt_tokens
            actual_out_tokens = response.usage.completion_tokens
            print("  (Goi OpenAI API thanh cong!)")
        except Exception as e:
            print(f"  (Loi khi goi OpenAI: {e})")
            ai_reply = f"Loi ket noi: {e}"
    else:
        print("  (Che do mo phong: Khong co API key)")
        ai_reply = "Doan code tren co the toi uu bang cach dung ham reduce() trong JavaScript..."
        actual_out_tokens = 60

    print(f"\n-> Phan hoi tu AI:\n{ai_reply.strip()}")

    # -------------------------------------------------------------
    # 7. GHI NHAN CHI TIEU THUC TE (POST-CALL LOGGING)
    # -------------------------------------------------------------
    cost = counter._cost(actual_in_tokens, actual_out_tokens)

    guard.record(
        input_tokens=actual_in_tokens,
        output_tokens=actual_out_tokens,
        provider=provider,
        model=model,
    )

    logger.log(
        provider=provider,
        model=model,
        input_tokens=actual_in_tokens,
        output_tokens=actual_out_tokens,
        cost_usd=cost,
    )

    print("\n[5] TONG KET CHI PHI & NGAN SACH HIEN TAI:")
    print(f"  - Input Tokens: {actual_in_tokens} | Output Tokens: {actual_out_tokens}")
    print(f"  - Chi phi dot nay: ${cost:.6f}")
    print(f"  - Trang thai Budget: {guard.usage.summary()}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
