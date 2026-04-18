# Optimization of Computer Vision Efficiency via Dual-Model Cascade Architectures

This repository contains the source code and dataset samples for the research project focused on optimizing inference efficiency in noisy urban environments (specifically Uralsk, Kazakhstan).

## Project Overview
The core idea is a **Dual-Model Cascade**. A lightweight model (MobileNetV3) performs the primary inference. If the confidence score falls below a predefined threshold ($\tau = 0.85$), the system triggers a more robust but computationally expensive model (EfficientNet-B0). 

This approach ensures high accuracy in difficult conditions (dirt on license plates, low lighting, shadows) while maintaining high speed during clear daylight conditions.

## Repository Structure
- `test_models.py`: The main Python script implementing the cascade logic.
- `data/`: Sample images from the "Uralsk Urban Dataset".
  - `day/`: Images captured during daylight.
  - `evening/`: Images captured during evening/low-light conditions.
- `requirements.txt`: List of necessary Python libraries.

## Implementation Details
The project utilizes:
- **Primary Model:** MobileNetV3-Small (Pre-trained on ImageNet).
- **Secondary Model:** EfficientNet-B0 (Pre-trained on ImageNet).
- **Inference Logic:** - If $P_{max} > 0.85$: Accept Light Model result.
  - If $P_{max} < 0.85$: Activate Heavy Model.

## How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
2. Install dependencies:
   pip install -r requirements.txt
3. Run the inference test:
python test_models.py

Author
Zhoshy Khalelov 10th-grade student, NIS Oral, Kazakhstan.
