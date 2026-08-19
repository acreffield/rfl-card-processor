import cv2
import easyocr
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="RFL Card Processor", layout="wide")
st.title("🎴 RFL Card Batch Processor")

show_debug = st.checkbox("🔍 Enable Debug Mode", value=True)


@st.cache_resource
def load_ocr_reader():
  return easyocr.Reader(["en"])


reader = load_ocr_reader()

uploaded_file = st.file_uploader("Upload Scanned PDF Batch", type=["pdf"])

if uploaded_file is not None:
  with st.spinner("Processing PDF batch... Please wait."):
    pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    card_data = []

    for page_idx, page in enumerate(pdf_doc):
      zoom = 300 / 72
      mat = fitz.Matrix(zoom, zoom)
      pix = page.get_pixmap(matrix=mat)

      img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
          pix.h, pix.w, pix.n
      )
      if pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      elif pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      h, w = gray.shape

      text_instances = page.search_for("Student Name")

      for idx, rect in enumerate(text_instances):
        x0, y0, x1, y1 = (
            int(rect.x0 * zoom),
            int(rect.y0 * zoom),
            int(rect.x1 * zoom),
            int(rect.y1 * zoom),
        )

        # 1. NAME CROP
        name_y1 = y1 + 2
        name_y2 = y1 + 110
        name_x1 = x0
        name_x2 = min(x0 + 700, w)

        name_crop = gray[name_y1:name_y2, name_x1:name_x2]

        results = reader.readtext(name_crop, detail=0)
        student_name = "Unread Name"
        if results:
          cleaned = " ".join(results).strip()
          if len(cleaned) > 1:
            student_name = cleaned

        # 2. EXPANDED GRID CROP (Height widened to capture bottom 2 rows)
        grid_y1 = name_y2 + 5
        grid_y2 = min(
            grid_y1 + int(h * 0.58), h
        )  # Expanded down to capture all rows
        grid_x1 = int(w * 0.32)
        grid_x2 = int(w * 0.99)

        grid_crop_gray = gray[grid_y1:grid_y2, grid_x1:grid_x2]

        # Convert to binary
        blurred = cv2.GaussianBlur(grid_crop_gray, (3, 3), 0)
        _, binary_grid = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY_INV)

        # --- LINE ERASE FILTER: Removes all straight table borders ---
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        h_lines = cv2.morphologyEx(
            binary_grid, cv2.MORPH_OPEN, h_kernel, iterations=2
        )

        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        v_lines = cv2.morphologyEx(
            binary_grid, cv2.MORPH_OPEN, v_kernel, iterations=2
        )

        table_grid_lines = cv2.add(h_lines, v_lines)
        ink_only_grid = cv2.subtract(binary_grid, table_grid_lines)

        # --- COUNT REMAINING HANDWRITTEN INK ---
        num_rows = 7
        num_cols = 5

        row_height = (grid_y2 - grid_y1) / num_rows
        col_width = (grid_x2 - grid_x1) / num_cols

        total_signatures = 0

        for row in range(num_rows):
          for col in range(num_cols):
            cx1, cx2 = int(col * col_width), int((col + 1) * col_width)
            cy1, cy2 = int(row * row_height), int((row + 1) * row_height)

            pad_x, pad_y = int(col_width * 0.10), int(row_height * 0.10)
            inner_cell = ink_only_grid[
                cy1 + pad_y : cy2 - pad_y, cx1 + pad_x : cx2 - pad_x
            ]

            if cv2.countNonZero(inner_cell) > 40:
              total_signatures += 1

        if show_debug:
          st.write(
              f"--- **Card {idx + 1} on Page {page_idx + 1} Debug** ---"
          )
          st.image(
              name_crop,
              caption=f"Name Crop (Extracted: '{student_name}')",
              width=450,
          )
          st.image(
              ink_only_grid,
              caption=(
                  f"Line-Erased Image (Full 7 Rows Covered) | Counted:"
                  f" {total_signatures} points"
              ),
              width=450,
          )

        card_data.append({
            "Student Name": student_name,
            "Behavior Event": "Green points from RFL cards",
            "Points": total_signatures,
            "PDF Page": page_idx + 1,
            "Card Index": idx + 1,
        })

    df = pd.DataFrame(card_data)

  st.success("Processing complete!")
  edited_df = st.data_editor(df, num_rows="dynamic")

  csv_data = edited_df.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📥 Download Bromcom Import File (CSV)",
      data=csv_data,
      file_name="bromcom_green_points_import.csv",
      mime="text/csv",
  )