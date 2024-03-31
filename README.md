
## Adversarial Attack and Defense - Beginner guide

This repository serves as a comprehensive resource for studying, experimenting with, and understanding adversarial attacks and defenses in the field of machine learning and artificial intelligence. This repository aims to provide you with the tools and knowledge necessary to explore potential vulnerabilities and robustness in machine learning models. Additionally, it offers insights into common adversarial attack and defense methods.



## Contents
### Attack Methods
**FGSM (Fast Gradient Sign Method):** A simple yet effective method for generating adversarial examples by perturbing input data in the direction of the gradient of the loss function.

**I-FGSM (Iterative Fast Gradient Sign Method)**: An iterative variant of FGSM that applies FGSM multiple times with smaller perturbations to generate stronger adversarial examples.

**Transfer Attack**: A method that leverages adversarial examples generated from one model to attack another model, even if the models are trained on different datasets.
### Defense Methods
**Adversarial Training**: A technique that involves training the model on adversarially perturbed examples to enhance its robustness against adversarial attacks.

**Image Transformation:** A defense method that applies image transformations, such as rotation, scaling, and compression, to input data to reduce the effectiveness of adversarial attacks.

**Ensemble Defense**: A defense strategy that combines multiple models to make predictions, making it more challenging for attackers to generate effective adversarial examples.

## Learning Material
[Presentation](slide.pdf): A comprehensive slide that covers the fundamentals of adversarial attacks and defenses, including key concepts, methods, and real-world applications, written in Vietnamese.

Colab Notebooks: Interactive Colab notebooks for hands-on experimentation with attack and defense methods directly in your browser. 
- Attack methods

  [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhtlongcs/simple-adversarial-attack/blob/main/notebooks/FGSM_attacks.ipynb)
- Defense methods

  [![Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nhtlongcs/simple-adversarial-attack/blob/main/notebooks/FGSM_defense.ipynb) 

This repository: The most effective learning method is directly from the source code. Dive into the codebase to understand the implementation details of each attack and defense method.

## Contributing
We welcome contributions from the community to enhance this repository further. Whether it's adding new attack methods, improving existing defenses, or providing additional learning materials, your contributions are invaluable.

## Support
If you encounter any issues, have questions, or want to provide feedback, please don't hesitate to create an issue. We're here to help
