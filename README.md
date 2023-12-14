# gCLIP: A Multimodal Approach to Integrating Visual and Textual Cues in Games

## Abstract
In this paper, we delve into the crucial role of visual and textual representations in gaming. Traditional Visual and Text Encoders often fall short in this domain. To address this, we introduce a novel multimodal framework, gCLIP, to enhance game understanding. Our approach involves a unique dataset of game screens with human-labeled captions and an adaptation of the Contrastive Language–Image Pretraining (CLIP) model using soft-labels. This method is tailored for nuanced game content analysis. Applications of our methodology include zero-shot classification, game genre clustering, game screen classification, and player risk assessment, showcasing the versatility and impact of our approach. Our study aims to bridge a gap in game-related AI research, offering insights for game developers, players, and researchers. The full paper will be available in January 2024.

## Repository Contents
This repository contains the code necessary for training the gCLIP model, as well as documentation and examples.

### Installation
1. **Install Required Modules:**
   ```
   pip install -r requirements.txt
   ```

### Preparing the Dataset
2. **Download the Dataset:**
   - Obtain the gCLIP Dataset.
   - Place the downloaded data in the `./data` directory.

### Training the Model
3. **Train the gCLIP Model:**
   ```
   python -m src.train
   ```

## Additional Information
- The paper, "gCLIP: A Multimodal Approach to Integrating Visual and Textual Cues in Games," is scheduled for release in January 2024.
- This framework is aimed at enhancing the understanding and interaction within game environments.



## Reference

- [Paper] [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020v1.pdf)
- [Repo] [moein-shariatnia/OpenAI-CLIP](https://github.com/moein-shariatnia/OpenAI-CLIP)
- [Repo] [openai/CLIP](https://github.com/openai/CLIP)
- [Repo] [Distributed Training in PyTorch](https://github.com/youngerous/distributed-training-comparison)
