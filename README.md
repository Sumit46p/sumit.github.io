# 📄 Document Scanner with OpenCV

This Python script takes an image of a document (e.g. a bill, form, or paper), detects the largest rectangular contour (presumably the document), and applies a **perspective transformation** to create a flat, top-down view of the document.

---

## 🔧 Features

- Loads and resizes an input image
- Converts to grayscale and applies adaptive thresholding
- Uses morphological operations to enhance edges
- Detects contours and locates the document using polygon approximation
- Applies a perspective transform to obtain a flattened view

---

## 📁 Input

Place your document image in the project folder and update the `im_path` in the script:

```python
im_path = "bill.jpg"
