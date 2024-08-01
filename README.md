# WBC-NCA

## Abstract
Diagnosis of hematological malignancies depends on accurate identification of white blood cells in peripheral blood smears. Deep learning techniques are emerging as a viable solution to scale and optimize this process by automatic cell classification. However, these techniques face several challenges such as limited generalizability, sensitivity to domain shifts, and lack of explainability. Here, we introduce a novel approach for white blood cell classification based on neural cellular automata (NCA). We test our approach on three datasets of white blood cell images and show that we achieve competitive performance compared to conventional methods. Our NCA-based method is significantly smaller in terms of parameters and exhibits robustness to domain shifts. Furthermore, the architecture is inherently explainable, providing insights into the decision process for each classification, which helps to understand and validate model predictions. Our results demonstrate that NCA can be used for image classification, and that they address key challenges of conventional methods, indicating a high potential for applicability in clinical practice.

## Model
![Architecture](/src/images/model_graphic.svg)
Neural cellular automata (NCA) can be used for the accurate classification of single white blood cells in patient blood smears. 
- A: Our approach consists of four steps: 
    1. image padding to increase the number of channels
    2. k NCA update steps to extract features from the image that manifest in the hidden channels
    3. pooling via channel-wise maximum
    4. a fully connected network to classify the image 
- B: The NCA step updates each cell based on its immediate surroundings according to equations. 
- C: Training the model end-to-end allows the NCA to learn an update rule that extracts useful features.

### Paper Preprint
https://arxiv.org/abs/2404.05584

## Contact
If you have any questions feel free to reach out via [e-mail](mailto:michael.deutges@gmail.com)