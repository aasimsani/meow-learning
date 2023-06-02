# MeowLearning: Train your team on Production AI - quickly

This is a *pick-your-problem* style guide I created to educate everyone from my leadership team to my ML engineers on the process of how to work with AI in production settings. This is stuff you *won't* learn as of today on Coursera or through most online courses.

#### Previously CopyCat's internal AI Guidelines

### *All reading topics are in reading order.*

## Fundementals
Readings covering a fundamentals overview of how Machine Learning Systems are made

[Machine Learning Systems Design - Part 1](https://docs.google.com/presentation/d/1bhjgRelQ0O5FnYCOGiCVWg_SkfRcZ9bffQsgk6yAaL0/edit?usp=sharing)

[Machine Learning Systems Design - Part 2](https://docs.google.com/presentation/d/1BYxwxJCb7onDemOtAZTmMc50V3tF80BflkuKZCBLUxg/edit?usp=sharing)

[Rules of Machine Learning: | Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)

*Product Leadership team can stop here*

## Research

**Where to look for models/techniques and the like?**

- Model Zoos
    - [Hugging face](https://huggingface.co/) and [their Github](https://github.com/orgs/huggingface/repositories)
    - [PyTorch Hub](https://pytorch.org/hub/research-models)
    - [Torchvision](https://pytorch.org/vision/stable/models.html)
    - [Torchaudio](https://pytorch.org/audio/stable/pipelines.html#module-torchaudio.pipelines)
    - [Torchtext - Hugging face is way better for this but just in case](https://pytorch.org/text/stable/models.html)
    - [TIMM - Vision models](https://timm.fast.ai/) - Also check [their Hugging Face Page](https://huggingface.co/docs/timm/reference/models)
    - [Tensorflow Hub](https://www.tensorflow.org/hub)
    - [Model Zoo.co](https://modelzoo.co/)
    - [Nvidia Model Zoo](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/overview.html)
    - [ONNX Model Zoo](https://github.com/onnx/models)
    - [NVIDIA NGC Model Zoo](https://ngc.nvidia.com/catalog/models)
    - [Facebook Research Model Zoo](https://github.com/facebookresearch)
    - [Keras Applications](https://keras.io/api/applications/)
    - [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
    - [MXNet/Gluon Model Zoo](https://mxnet.apache.org/versions/1.6/ecosystem)
    - [Apple Machine Learning Models](https://developer.apple.com/machine-learning/models/)
    - some more exist...
 
*Please note that the availability and content of these model zoos may vary, so it's always best to refer to the official documentation provided by each platform.*

- AI company Githubs
    - [Laion](https://www.laion.ai/blog)
    - [Ultralytics](https://www.ultralytics.com/blog)
    - [AirBnB](https://medium.com/airbnb-engineering/ai/home)
    - [Facebook AI](https://ai.facebook.com/blog/)
    - [Google AI](https://ai.googleblog.com/)
    - [Microsoft AI](https://blogs.microsoft.com/ai/)
    - [Netflix TechBlog](https://netflixtechblog.com/tagged/machine-learning)
    - and many more…
- AI lab blogs
    - [CSAIL - MIT/Stanford](https://www.csail.mit.edu/)
    - [CMU Blog](https://www.cmu.edu/news/index.html)
    - [Taiwan AI Labs](https://www.twai.com.tw/en/)
    - [Deepmind blog](https://deepmind.com/blog)
    - [OpenAI Blog](https://openai.com/blog/)
    - [Synced Review](https://syncedreview.com/)
    - [BAIR](https://bair.berkeley.edu/)
    - and many more…

- Github search - but follow the rules below to quickly filter out duds

**Where not to look for models?**
- Towards Data Science and other unmoderated blogs (unless they link to one of the above)
- Kaggle
- Github Snippets

**Rules to figure out what is and is not promising**

- Look at whether existing implementations exist
    - Look at code cleanliness first and foremost. Bad AI code is a major pain. Ask me for stories about PyTorch’s FasterRCCN being broken and how we wasted 1 month behind it.
    - Look at Git repository popularity. More people using something often means bugs have been caught or addressed.
    - Look at open issues in the git repo.
- Look at if people have used these kinds of models in production already
- Look at if you can find if these models have been used on data of similar complexity and scale as the use case
- Understand the math and the process. Is it actually going to do something meaningful to your data - especially in the case of self-supervised learning. E.g. random cropping images of buttons to learn features in an auto-encoder won’t make sense but doing it for a whole UI image might.

More covered in planning below.

## Planning

[Finding the best way to collect data + Finding the right metric](https://docs.google.com/presentation/d/1OYjrmhSBu3Poo5FcY6WywpU_eR7mtkpe1r8nbbWvArg/edit?usp=sharing)

## Baseline Testing 

**Also covered further in pre-requisite readings**

*Make your test dataset before you train anything*

Setting up an appropriate baseline is an important step that many candidates forget. There are three different baselines that you should think about:

- *Random baseline*: if your model predicts everything randomly, what's the expected performance?
- *Human baseline*: how well would humans perform on this task?
- *Simple heuristic*: for example, for the task of recommending the app to use next on your phone, the simplest model would be to recommend your most frequently used app. If this simple heuristic can predict the next app accurately 70% of the time, any model you build has to outperform it significantly to justify the added complexity.

## Data Tagging

It's extremely hard to find good advice or a one-size fits all solution on with data annotation and what works well but here are a few resources I've been able to find.

**Tagging Guidelines**
[Labelling Guidelines by Eugene Yan](https://eugeneyan.com/writing/labeling-guidelines/)

[How to Develop Annotation Guidelines by Prof. Dr. Nils Reiter](https://nilsreiter.de/blog/2017/howto-annotation)

**Data pipeline integrity**

* [Great Expectations](https://github.com/great-expectations/great_expectations): Helps data teams eliminate pipeline debt, through data testing, documentation, and profiling. - Your new best friend as a Data Scientist
* [Soda Core](https://github.com/sodadata/soda-core): Data profiling, testing, and monitoring for SQL accessible data. - Your kinda sorta-best friend
* [ydata-quality](https://github.com/ydataai/ydata-quality): Data Quality assessment with one line of code. - is cool but inflexible
* [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling): Extends the pandas DataFrame with df.profile_report() for quick data analysis.
* [DataProfiler](https://github.com/capitalone/DataProfiler): A Python library designed to make data analysis, monitoring and sensitive data detection easy. - Bit tough to use

**Data tagging platforms**
I love using [Scale AI](https://scale.com/) for tagging but if you're looking for something free then [LabelStudio](https://labelstud.io/) is a good start

## Training

*Section not essential for anyone but MLEs*

[Understanding common data challenges in training](https://docs.google.com/presentation/d/1Gq3VHW-0ci1gTh97OlckrCBqi3qgkjQNV0SO9t42Eyg/edit?usp=sharing)

[Understanding training, model selection, and other processes](https://docs.google.com/presentation/d/1X_w55MfBhXGQbZkT_fbW9wdNrOs4sOuydRGPUI_yYCo/edit?usp=sharing)

## Debugging AI models

*Section not essential for anyone but MLEs*

[Google Model Tuning Playbook](https://github.com/google-research/tuning_playbook)

[Full Stack Deep Learning - Lecture 7: Troubleshooting Deep Neural Networks](https://fullstackdeeplearning.com/spring2021/lecture-7/)

## Prototyping

It's often very useful to setup an internal prototyping/testing interface for any AI model + it's data that you plan to deploy

[Gradio](https://gradio.app/)

[Streamlit](https://streamlit.io/)

## Testing and expandability
[Infrastructure challenges and considerations](https://docs.google.com/presentation/d/1RqbEbMDmxq53jhjVi9V30-DYMv0PiUqlNlTZsw9Vm9Y/edit?usp=sharing)

[Full Stack Deep Learning - Lecture 10: Testing & Explainability](https://fullstackdeeplearning.com/spring2021/lecture-10/)

**Tools**
* [ZenoML](https://zenoml.com/) - Data and model result explainability - very new but simple and great for computer vision
* [Netron](https://github.com/lutzroeder/netron): Visualizer for neural network, deep learning, and machine learning models.
* [Deepchecks](https://github.com/deepchecks/deepchecks): Test Suites for Validating ML Models & Data. Deepchecks is a Python package for comprehensively validating your machine learning models and data with minimal effort.
* [Evidently](https://github.com/evidentlyai/evidently): Interactive reports to analyze ML models during validation or production monitoring.

* I'd also highly recommend some kind of hardware usage monitoring to see if models are actually efficient e.g. RAM, CPU, GPU % util - most if not Cloud Platforms have this.

## Deployment and Scaling beyond your own machine

**Deployment checklist**

[GitHub - modzy/model-deployment-checklist: An efficient, to-the-point, and easy-to-use checklist to following when deploying an ML model into production.](https://github.com/modzy/model-deployment-checklist?utm_source=substack&utm_medium=email)

**Understanding infrastructure in general**

This moves very fast and gets crazier by the month. Just take a look at the [MAD: Machine Learning, Artificial Intelligence & Data Landscape for 2023](https://mad.firstmark.com/).

But here are the essentials you need to make AI happen smoothly:

- Code versioning
- Model and Artifact versioning
- Data versioning
- Data storage and collection/collation pipeline
- Model training infra - GPU machines + and preferably platforms like [KubeFlow](https://www.kubeflow.org/docs/components/pipelines/) or [SageMaker](https://aws.amazon.com/sagemaker/)
- Experiment Tracking - My go to is [Weights and Biases](https://wandb.ai/) or [Tensorboard](https://www.tensorflow.org/tensorboard) but if you're looking for a more packaged solution [MLFlow](https://mlflow.org/) is great
- Monitoring/Logging
- Inference Deployment Mechanism (more below)

There are also a bunch of all-in-one platforms that do all or most of these things like [MLFlow](https://mlflow.org/), [Neptune](https://neptune.ai/), [Sagemaker](https://aws.amazon.com/sagemaker/), [Vertex](https://cloud.google.com/vertex-ai), or [Polyaxon](https://polyaxon.com/).

[Full Stack Deep Learning - Lecture 6: MLOps Infrastructure & Tooling](https://fullstackdeeplearning.com/spring2021/lecture-6/)

**Deployment**

The deployment “stack” - this also keeps moving quite fast but the basic principles remain the same. A quick Google Search doesn't hurt though.

1. Know your use-case's deployment platform e.g. Mobile, Web, Edge, etc.
2. Find the stack/toolkit/library that works on your platform and with company requirements e.g Tensorflow Lite (Mobile/Edge), Google's Vertex AI prediction (SaaS), Torchserve/TFX (Backend and enterprise grade) and BentoML (Backend but simpler)
3. Understand your use-case's constraints e.g. real-time for video or batch for recommendation engines
4. Optimize for time/cost/performance/hardware

[Full Stack Deep Learning - Lecture 11: Deployment & Monitoring](https://fullstackdeeplearning.com/spring2021/lecture-11/)

UPDATE: [Lecture 5: Deployment](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/)

## Monitoring

[How to make sure ML systems fail and you can see and know it](https://docs.google.com/presentation/d/1tuCIbk9Pye-RK1xqiiZXPzT8lIgDUL6CqBkFSYZXkbY/edit?usp=sharing)

**Tools**
* [Aporia](https://www.aporia.com/): Observability with customized monitoring and explainability for ML models.
* [Gantry](https://gantry.io/): ML Observability platform with analytics, alerting, and human feedback
* [Arize](https://arize.com/): An end-to-end ML observability and model monitoring platform.
* [WhyLabs](https://whylabs.ai/): AI Observability platform - they also have opensource components
* [Fiddler](https://www.fiddler.ai/): Monitor, explain, and analyze your AI in production.
* [Superwise](https://www.superwise.ai): Fully automated, enterprise-grade model observability in a self-service SaaS platform.

If these platforms don't work for you I recommend making your own pipeline using either the:
- [ELK Stack](https://www.elastic.co/what-is/elk-stack)
- [Grafana's Stack](https://grafana.com/)
- [Manifold](https://github.com/uber/manifold): A model-agnostic visual debugging tool for machine learning.
- Your own logger + Your own data stores + Your own BI (Metabase, Superset, etc.) - not recommended

**I'd also highly recommend some kind of hardware usage monitoring to see if models are actually efficient e.g. RAM, CPU, GPU % util - most if not Cloud Platforms have this.**

## Acknowledgements and references

[Stanford CS329S Course by Chip Hyuen - CS 329S | Syllabus](https://stanford-cs329s.github.io/syllabus.html)

[Full Stack Deep Learning by Josh Tobin and Sergey Karayev](https://fullstackdeeplearning.com/)

[Rules of Machine Learning: | Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)

[Google Model Tuning Playbook](https://github.com/google-research/tuning_playbook)

[Labelling Guidelines by Eugene Yan](https://eugeneyan.com/writing/labeling-guidelines/)

[How to Develop Annotation Guidelines by Prof. Dr. Nils Reiter](https://nilsreiter.de/blog/2017/howto-annotation)

