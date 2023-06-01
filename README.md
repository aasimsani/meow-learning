# Train your team on AI - quickly

This is a *pick-your-problem* style guide I created to educate everyone from my leadership team to my ML engineers on the process of how to create AI in production settings.

#### Previously CopyCat AI Guidelines

### *All reading topics are in reading order.*

**Readings covering a fundamentals overview of how Machine Learning Systems are made**

[cs329s_2022_02_slides_mlsd](https://docs.google.com/presentation/d/1BYxwxJCb7onDemOtAZTmMc50V3tF80BflkuKZCBLUxg/edit?usp=sharing)

[Rules of Machine Learning: | Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)

*Product Leadership team can stop here*

**Research**

Where to look for models/techniques and the like?

- Model Zoos
    - modelzoo.co
    - Hugging face
    - Torchvision
    - Torchaudio
    - Torchtext
    - TIMM
    - Tensorflow Hub
    - Nvidia Model Zoo
- AI company Githubs
    - Laion
    - Ultralytics
    - AirBnB
    - Facebook
    - Google
    - Microsoft
    - Netflix
    - and many more…
- AI lab blogs
    - CSAIL - MIT/Stanford
    - CMU Blog
    - Taiwan AI Labs
    - Deepmind blog
    - OpenAI Blog
    - Synced Review
    - BAIR
    - and many more…

Where not to look for models?

- Towards Data Science and other unmoderated blogs (unless they link to one of the above)
- Kaggle

Rules to figure out what is and is not promising

- Look at whether existing implementations exist
    - Look at code cleanliness first and foremost. Bad AI code is a major pain. Ask me for stories about PyTorch’s FasterRCCN being broken and how we wasted 1 month behind it.
    - Look at Git repository popularity. More people using something often means bugs have been caught or addressed.
    - Look at open issues in the git repo.
- Look at if people have used these kinds of models in production already
- Look at if you can find if these models have been used on data of similar complexity and scale as the use case
- Understand the math and the process. Is it actually going to do something meaningful to your data - especially in the case of self-supervised learning. E.g. random cropping images of buttons to learn features in an auto-encoder won’t make sense but doing it for a whole UI image might.

More covered in planning below.

**Planning**

Finding the best way to collect data + Finding the right metric

[cs329s_2022_03_slides_training_data](https://docs.google.com/presentation/d/1OYjrmhSBu3Poo5FcY6WywpU_eR7mtkpe1r8nbbWvArg/edit?usp=sharing)

**Baseline Testing - also covered in large in pre-requisite readings**

Make your test dataset before you train anything

Setting up an appropriate baseline is an important step that many candidates forget. There are three different baselines that you should think about:

- *Random baseline*: if your model predicts everything randomly, what's the expected performance?
- *Human baseline*: how well would humans perform on this task?
- *Simple heuristic*: for example, for the task of recommending the app to use next on your phone, the simplest model would be to recommend your most frequently used app. If this simple heuristic can predict the next app accurately 70% of the time, any model you build has to outperform it significantly to justify the added complexity.

**Training**

*Not essential for anyone but MLEs*

Understanding common data challenges in training

[cs329s_2022_04_slides_feature_engineering](https://docs.google.com/presentation/d/1Gq3VHW-0ci1gTh97OlckrCBqi3qgkjQNV0SO9t42Eyg/edit?usp=sharing)

Understanding training, model selection, and other processes

[cs329s_2022_05_slides_model_development](https://docs.google.com/presentation/d/1X_w55MfBhXGQbZkT_fbW9wdNrOs4sOuydRGPUI_yYCo/edit?usp=sharing)

Debugging AI models

[https://github.com/google-research/tuning_playbook](https://github.com/google-research/tuning_playbook)

[Full Stack Deep Learning - Lecture 7: Troubleshooting Deep Neural Networks](https://fullstackdeeplearning.com/spring2021/lecture-7/)

**Deployment**

Deployment checklist

[GitHub - modzy/model-deployment-checklist: An efficient, to-the-point, and easy-to-use checklist to following when deploying an ML model into production.](https://github.com/modzy/model-deployment-checklist?utm_source=substack&utm_medium=email)

Understanding infrastructure

[Full Stack Deep Learning - Lecture 6: MLOps Infrastructure & Tooling](https://fullstackdeeplearning.com/spring2021/lecture-6/)

The deployment “stack”

[Full Stack Deep Learning - Lecture 11: Deployment & Monitoring](https://fullstackdeeplearning.com/spring2021/lecture-11/)

**End to End and pre-deployment testing**

How to evaluate models before deploying in terms of infrastructural challenges

[cs329s_2022_06_slides_model_evaluation](https://docs.google.com/presentation/d/1RqbEbMDmxq53jhjVi9V30-DYMv0PiUqlNlTZsw9Vm9Y/edit?usp=sharing)

Testing and expandability guidelines

[Full Stack Deep Learning - Lecture 10: Testing & Explainability](https://fullstackdeeplearning.com/spring2021/lecture-10/)

**Monitoring**

How to make sure ML systems fail and you can see and know it

[cs329s_2022_10_slides_ml_failure_diagnosis](https://docs.google.com/presentation/d/1tuCIbk9Pye-RK1xqiiZXPzT8lIgDUL6CqBkFSYZXkbY/edit?usp=sharing)

**References and resources**

Full Standford - CS329S Course

[CS 329S | Syllabus](https://stanford-cs329s.github.io/syllabus.html)
