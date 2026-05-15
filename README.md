# Predicting Ravichandran Ashwin's Spin Delivery Type from Broadcast Footage

Final year BSc Computer Science dissertation, Oxford Brookes University. COMP6013 Computing Honours Project, 2025/26.

An LSTM-based system that predicts whether Ashwin is about to bowl an off break or a carrom ball, using only standard broadcast footage and before the ball leaves his hand.

## Problem

Existing cricket analytics tools like Hawk-Eye only track the ball after release. No publicly accessible system existed that could predict spin delivery type before release from broadcast footage. The two deliveries look near identical during Ashwin's bowling action. This project asks whether a neural network can detect subtle biomechanical cues that the human eye cannot process fast enough.

## Approach

End to end pipeline built using only free tools and publicly available footage.

1. **Data collection.** yt-dlp used to clip 2 to 3 second segments from YouTube broadcasts. Final dataset: 31 clips (20 off break, 11 carrom ball).
2. **Feature extraction.** MediaPipe Heavy pose landmarker extracts right elbow and right wrist coordinates frame by frame using OpenCV. 1,871 rows of (x, y) data saved to CSV.
3. **Model.** LSTM in PyTorch. Two stacked layers (64 hidden units), dropout 0.3, fully connected head (64 to 32 to 2). Trained on 20 frame sliding windows with class weighting to handle dataset imbalance.
4. **Prediction interface.** Google Colab notebook that accepts a new clip and outputs a delivery type prediction with confidence percentage.

## Results

| Metric | Value |
|--------|-------|
| Test accuracy | 69% |
| Off break recall | 75% (15/20) |
| Carrom ball recall | 56% (5/9) |

The model classifies genuinely in both directions. Early versions predicted every sequence as off break until class weighting was applied. Performance is comparable to Mutawa et al. (2025), who achieved 70% on a related cricket movement prediction task using a larger controlled dataset.

## Tech Stack

Python, PyTorch, MediaPipe Heavy, OpenCV, yt-dlp, pandas, NumPy, scikit-learn, Matplotlib, Seaborn. Developed in Google Colab with Drive storage.

## Key Design Decisions

- **LSTM over CNN.** Delivery prediction is a temporal problem. The pattern across 20 frames is the signal, not any single moment.
- **Elbow and wrist only.** Most relevant joints for the release action. Kept the model simple given the small dataset.
- **MediaPipe Heavy.** Most accurate of three available variants. Precision matters when biomechanical differences are subtle.
- **20 frame windows.** Captures the full bowling action without irrelevant pre or post motion.
- **Class weighting.** Corrected the off break only baseline once dataset imbalance was diagnosed.
- **2D coordinates only.** Z axis excluded as unreliable from a monocular broadcast camera.

## Limitations

- 2D camera angle cannot capture wrist rotation in the depth axis
- 31 clips is a small dataset for a neural network
- Only two landmarks tracked
- Dataset imbalance (20 vs 11) inherent to footage availability
- Multiple people often visible in frame (keeper, batter, fielders)
- Some clips include post release frames

## Future Work

- Collect 100 or more confirmed clips per delivery type
- Use side on camera footage at 90 degrees to the bowling crease
- Add shoulder, hand orientation, and finger position landmarks
- Extend to multiple bowlers and additional delivery types
- Frame level annotation for pre release only training
- Apply SHAP values or attention weights for explainability

## Author

Joval Calix Augustus Valentine. Final year BSc Computer Science, Oxford Brookes University. Supervisor: Shumao Ou. Submitted 17 April 2026.

## License

MIT
