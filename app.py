import os
import time
import json
import requests
import streamlit as st
from dotenv import load_dotenv
# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
# Phase-1 (OpenAI)
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Phase-2 (OCR / Computer Vision Read v3.2 GA)
AZURE_OCR_KEY = os.getenv("AZURE_OCR_KEY")
AZURE_OCR_ENDPOINT = os.getenv("AZURE_OCR_ENDPOINT")  # e.g. https://xxx.cognitiveservices.azure.com/

# -----------------------------
# Helpers
# -----------------------------
def _safe_sum(items):
    total = 0.0
    for it in items or []:
        try:
            total += float(it.get("price", 0) or 0)
        except Exception:
            pass
    return total

# -----------------------------
# Phase-1: classify items via OpenAI
# -----------------------------
def classify_items_with_ai(grocery_text: str):
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}

    prompt = f"""
    You are a budgeting assistant. The user will give you a grocery list with items, quantities, and prices.
    Classify each item into 'Essential' or 'Non-Essential'.
    Return the result as JSON like this:
    {{
        "essentials": [{{"item": "Milk", "quantity":1, "price":3}}],
        "non_essentials": [{{"item": "Chips", "quantity":1, "price":4}}],
        "suggestions": ["Suggestion 1", "Suggestion 2"]
    }}
    Here is the list:
    {grocery_text}
    """

    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful AI that outputs JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        content = payload["choices"][0]["message"]["content"]
        return json.loads(content)
    except json.JSONDecodeError:
        st.error("AI did not return valid JSON. Showing raw content:")
        st.write(payload["choices"][0]["message"]["content"] if "payload" in locals() else "(no content)")
        return None
    except requests.HTTPError:
        st.error("Azure OpenAI HTTP error")
        try:
            st.json(resp.json())
        except Exception:
            st.write(str(resp.text))
        return None
    except Exception as e:
        st.error(f"Unexpected error calling Azure OpenAI: {e}")
        return None

# -----------------------------
# Phase-2: OCR via Computer Vision Read v3.2
# -----------------------------
def extract_text_from_receipt(file_bytes: bytes) -> str:
    analyze_url = f"{AZURE_OCR_ENDPOINT}vision/v3.2/read/analyze"
    headers = {"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY, "Content-Type": "application/octet-stream"}

    try:
        submit = requests.post(analyze_url, headers=headers, data=file_bytes, timeout=60)
        submit.raise_for_status()
        op_location = submit.headers.get("Operation-Location") or submit.json().get("operationLocation")
        if not op_location:
            return "⚠️ OCR did not return a valid Operation-Location."

        for _ in range(60):
            poll = requests.get(op_location, headers={"Ocp-Apim-Subscription-Key": AZURE_OCR_KEY}, timeout=30)
            poll.raise_for_status()
            result = poll.json()
            status = result.get("status", "").lower()

            if status == "succeeded":
                lines = []
                analyze = result.get("analyzeResult", {})
                for page in analyze.get("readResults", []):
                    for line in page.get("lines", []):
                        txt = line.get("text", "").strip()
                        if txt:
                            lines.append(txt)
                return "\n".join(lines) if lines else "⚠️ No text found on this receipt."
            if status == "failed":
                return "⚠️ OCR failed to process the receipt."
            time.sleep(1)

        return "⚠️ OCR timed out while reading the receipt."
    except Exception as e:
        return f"⚠️ OCR error: {e}"

# -----------------------------
# Preprocess OCR via OpenAI to clean & structure
# -----------------------------
def preprocess_ocr_text(raw_text: str) -> str:
    """
    Send raw OCR text to OpenAI to extract items with quantity and price in table format.
    """
    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_OPENAI_KEY}

    prompt = f"""
    You are a grocery assistant. The user provides a raw receipt text:
    {raw_text}

    Extract only the purchased items with their quantity (default 1 if not present) and total price.
    Ignore any other irrelevant text (store info, barcodes, date, etc.).
    Return the result as a formatted text list like this (human readable):
    Item - Quantity - Price
    Milk - 1 - 3
    Chips - 2 - 5
    """
    body = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that formats receipt text."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        return payload["choices"][0]["message"]["content"]
    except Exception as e:
        return f"⚠️ Error preprocessing OCR text: {e}"

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Smart Grocery Saver", page_icon="🛒", layout="centered")
st.title("🛒 Smart Grocery Saver")

tab_text, tab_ocr = st.tabs(["✍️ Manual Entry (Phase 1)", "🧾 Receipt OCR (Phase 2)"])

# ----- Phase 1: manual entry -----
with tab_text:
    st.write("Enter your grocery list (one per line as `Item - price`).")
    sample_text = """Milk - 3
Bread - 2
Coca Cola - 5
Chips - 4
Rice - 10"""
    grocery_input = st.text_area("Grocery List:", sample_text, height=200)

    if st.button("Analyze (Text)"):
        if not grocery_input.strip():
            st.warning("Please enter your grocery list.")
        else:
            with st.spinner("Analyzing your grocery list..."):
                data = classify_items_with_ai(grocery_input)

            if data:
                essentials = data.get("essentials", [])
                non_essentials = data.get("non_essentials", [])
                suggestions = data.get("suggestions", [])

                essentials_total = _safe_sum(essentials)
                non_essentials_total = _safe_sum(non_essentials)
                total_spent = essentials_total + non_essentials_total

                saving_remove = non_essentials_total
                saving_half = non_essentials_total / 2.0

                st.subheader("📊 Summary")
                st.write(f"**Total Spent:** Rs.{total_spent:.2f}")
                st.write(f"**Essentials:** Rs.{essentials_total:.2f}")
                st.write(f"**Non-Essentials:** Rs.{non_essentials_total:.2f}")

                if total_spent > 0:
                    st.subheader("💰 Potential Savings")
                    st.write(f"Remove all non-essentials: Save **Rs.{saving_remove:.2f}** ({(saving_remove/total_spent)*100:.1f}%)")
                    st.write(f"Reduce non-essentials by 50%: Save **Rs.{saving_half:.2f}** ({(saving_half/total_spent)*100:.1f}%)")

                if suggestions:
                    st.subheader("📝 Suggestions")
                    for s in suggestions:
                        st.write(f"- {s}")

                st.subheader("✅ Essentials")
                st.table(essentials)

                st.subheader("🚫 Non-Essentials")
                st.table(non_essentials)

# ----- Phase 2: receipt OCR -----
with tab_ocr:
    st.write("Upload a receipt image or PDF; we’ll read it and analyze the items.")
    file = st.file_uploader("Upload (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])

    if file is not None:
        file_bytes = file.read()
        with st.spinner("Running OCR..."):
            ocr_text = extract_text_from_receipt(file_bytes)

        st.subheader("📝 Extracted Text")
        st.text(ocr_text if ocr_text else "(No text extracted)")

        if ocr_text and not ocr_text.startswith("⚠️"):
            with st.spinner("Cleaning OCR text..."):
                cleaned_text = preprocess_ocr_text(ocr_text)
            st.subheader("📝 Preprocessed Text")
            st.text(cleaned_text)

            if st.button("Analyze (Receipt Text)"):
                with st.spinner("Analyzing extracted items..."):
                    data = classify_items_with_ai(cleaned_text)

                if data:
                    essentials = data.get("essentials", [])
                    non_essentials = data.get("non_essentials", [])
                    suggestions = data.get("suggestions", [])

                    essentials_total = _safe_sum(essentials)
                    non_essentials_total = _safe_sum(non_essentials)
                    total_spent = essentials_total + non_essentials_total

                    saving_remove = non_essentials_total
                    saving_half = non_essentials_total / 2.0

                    st.subheader("📊 Summary")
                    st.write(f"**Total Spent:** Rs.{total_spent:.2f}")
                    st.write(f"**Essentials:** Rs.{essentials_total:.2f}")
                    st.write(f"**Non-Essentials:** Rs.{non_essentials_total:.2f}")

                    if total_spent > 0:
                        st.subheader("💰 Potential Savings")
                        st.write(f"Remove all non-essentials: Save **Rs.{saving_remove:.2f}** ({(saving_remove/total_spent)*100:.1f}%)")
                        st.write(f"Reduce non-essentials by 50%: Save **Rs.D:{saving_half:.2f}** ({(saving_half/total_spent)*100:.1f}%)")

                    if suggestions:
                        st.subheader("📝 Suggestions")
                        for s in suggestions:
                            st.write(f"- {s}")

                    st.subheader("✅ Essentials")
                    st.table(essentials)

                    st.subheader("🚫 Non-Essentials")
                    st.table(non_essentials)