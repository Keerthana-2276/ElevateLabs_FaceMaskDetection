
# Face Mask Detection from Local Images 🧠😷

This project performs **face mask detection** on static images using a pre-trained Keras model and OpenCV's Haar cascade face detection. It detects faces in an image and classifies each face as wearing a mask or not.

---

## 📁 Project Structure

```
FaceMaskDetector/
├── mymodel.h5                       # Pre-trained Keras model
├── facemask.py                     # Main script for image mask detection
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── mask.png                        # Input test image (changeable)
├── temp.jpg                        # Temp image used for prediction
└── README.md                       # You're here!
```

---

## ⚙️ Requirements

Install the required dependencies:

```bash
pip install keras tensorflow opencv-python
```

> Optional for future visualization:
```bash
pip install matplotlib
```

---

## ▶️ How to Run

1. Clone the repository or download the code.
2. Place your input image as `mask.png` in the same directory or update the image path in the script.
3. Run the detection script:

```bash
python facemask.py
```

---

## 📸 How It Works

- Uses **Haar cascade** to detect faces from the image.
- Each face is cropped and resized to `150x150` to match model input.
- The Keras model (`mymodel.h5`) predicts whether the face has a mask.
- Annotates the image with labels: `MASK` (✅ green) or `NO MASK` (❌ red).

---

## 🧪 Testing

Use the provided sample images or generate your own using tools like DALL·E or open-source face datasets.

---

## 📌 Notes

- Works best on **medium-resolution** images.
- Automatically resizes oversized images for better detection performance.
- Model was trained separately and not included in this repo due to size limits.

---

## 🤝 Contributing

Feel free to fork this repo and contribute new features like:
- Real-time webcam support
- Mask confidence heatmaps
- Integration with webcam feeds or video

---

## 📜 License

This project is under the [MIT License](LICENSE).
