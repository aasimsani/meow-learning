# MeowLearning: Train your team on Production AI - quickly

This is a *pick-your-problem* style guide I created to educate everyone from my leadership team to my ML engineers on how to work with AI in production settings. This is the stuff you *won't* learn in most ML/AI courses.

#### Previously CopyCat's internal AI Guidelines

### *All reading topics are in reading order.*

## Fundamentals
Readings covering a fundamentals overview of how Machine Learning Systems are made

[Machine Learning Systems Design - Part 1](https://docs.google.com/presentation/d/1bhjgRelQ0O5FnYCOGiCVWg_SkfRcZ9bffQsgk6yAaL0/edit?usp=sharing)

[Machine Learning Systems Design - Part 2](https://docs.google.com/presentation/d/1BYxwxJCb7onDemOtAZTmMc50V3tF80BflkuKZCBLUxg/edit?usp=sharing)

[Rules of Machine Learning: | Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)

[Large Language Models | Full Stack Deep Learning](https://youtu.be/MyFrMFab6bo?list=PL1T8fO7ArWleyIqOy37OVXsP4hFXymdOZ)

#### My Golden Rule above everything else
*Focus on the speed at which you can run valid experiments, it is the only way to find a viable model for your problem.*

*Product Leadership team can stop here*

#### AutoML: Before anything else - because the Golden Rule is key!

Recent experiences have shown me that AutoML has come a long way in the past five years especially for Tabular Machine Learning. So my latest recommendation is to use it. Use it first.
- Get your data in order ~ clean | preprocess | shrink/project if needed
- Use AutoML
- See what baselines it gives - if it works out of the box I'm happy for you but very jealous! :p

**AutoML tools**

Tip: You can use a foundation model like CLIP-VIT or GPTx as a pre-processor to make any data into structured data (embedding) for tasks as a quick and dirty experiment.

*Structured ~ Tabular ~ Embedding ~ Preprocessed

1. [Lazy Predict](https://pypi.org/project/lazypredict/) - Structured
2. [AutoGluon](https://auto.gluon.ai/stable/index.html) - Structured | Image | Text | Multimodal | Time series
3. [H2O](https://github.com/h2oai/h2o-3) - Tabular possibly Structured
4. [MLJar](https://github.com/mljar/mljar-supervised) - Structured - Has auto-feature engineering
5. [AutoPytorch](https://github.com/automl/Auto-PyTorch) - Structured
6. [AutoSklearn](https://automl.github.io/auto-sklearn/master) - Structured
7. [TPOT](http://epistasislab.github.io/tpot/) - Structured - Has auto-feature engineering
8. [TPOT2](https://epistasislab.github.io/tpot2/) - Structured - Has auto-feature engineering
9. [AutoKeras](https://autokeras.com/) - Structured | Image | Text | Time Series | Multimodal
10. [FLAML](https://github.com/microsoft/FLAML) - Structured
11. [PyCaret](https://pycaret.org/) - Structured | Time Series | Text
12. [AutoGen](https://microsoft.github.io/autogen/) - LLMs
13. [TransmogrifyAI](https://github.com/salesforce/TransmogrifAI) - Structured
14. [Model Search by Google](https://github.com/google/model_search) - Structured | Image | Text | Audio | Time Series

A number of these are also extendable with your custom models which aren't just Tabular - FLAML, AutoGluon, AutoKeras

If you have the ($)_($)
15. [GCP Vertex AutoML](https://cloud.google.com/vertex-ai/docs/beginner/beginners-guide)
16. [AWS SageMaker AutoML](https://aws.amazon.com/machine-learning/automl/)

Theoretrically you can also use any model hub for "AutoML" if you combine it with a sweeping agent.

E.g. [HuggingFace Autotrain](https://huggingface.co/autotrain) + [Weights and Biases Sweeps](https://wandb.ai) - Technically not AutoML but so many models so it's so very easy to do




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
- AI researcher's personal blogs
- Paper author's Github

- Github search - but follow the rules below to quickly filter out duds

**Where not to look for models?**
- Towards Data Science and other unmoderated blogs (unless they link to one of the above)
- Kaggle
- Github Snippets

**Rules to figure out what is and is not promising**

- Look at whether existing implementations exist

- If no then I highly recommend finding another architecture, this process can be excruciating and time consuming but if you do have to:
    - Select a design pattern to write the Neural Network in - I love a class based system like PyTorch does and then using [PyTorch Lightning's](https://www.pytorchlightning.ai/index.html) prescribed format on top of it.
    - Read the research paper and see if they've specified all the bits and bobs of the architecture and the training process - if not email the authors - you may get lucky
    - Write the pseudocode - especially the math
    - Implementation
        - [Beginner Advice on Learning to Implement Machine Learning Models](http://jsatml.blogspot.com/2014/10/beginner-advice-on-learning-to.html)
        - Courses that help and are great reference material:
            - [Practical Deep Learning for Coders (FastAI)](https://course.fast.ai/)
            - [Neural Networks: Zero To Hero by Andrej Karpath](https://karpathy.ai/zero-to-hero.html)
            - [Cutting Edge Deep Learning For Coders (FastAI)](https://course18.fast.ai/part2.html)
            - [Dive into Deep Learning](https://d2l.ai)
        - Coursera and Udacity courses tend to have everything handed to you in a sliver platter so while they're good for basics they don't help with this much.
    - Optimize later - premature optimization is the bane of all good code

- If yes then
    - Look at code cleanliness first and foremost. Bad AI code is a major pain. Ask me for stories about PyTorch’s FasterRCCN being broken and how we wasted 1 month behind it.
        - Look at Git repository popularity. More people using something often means bugs have been caught or addressed.
        - Look at open issues in the git repo.
    - Look at if people have used these kinds of models in production already
    - Look at if you can find if these models have been used on data of similar complexity and scale as the use case
    - Understand the math and the process. Is it actually going to do something meaningful to your data - especially in the case of self-supervised learning. E.g. random cropping images of buttons to learn features in an auto-encoder won’t make sense but doing it for a whole UI image might.

- See if the dataset the model is trained and tested on is publicly available and feasibly downloadable - if not don't fret too much on this step since the goal is to make it work for your data and your problem.
- Test your implementation on the dataset and see if you can reproduce results within the ball park (~2-5% error difference is fine)
- See debugging your AI models section below

More covered in planning below.

## Planning

[Finding the best way to collect data + Finding the right metric](https://docs.google.com/presentation/d/1OYjrmhSBu3Poo5FcY6WywpU_eR7mtkpe1r8nbbWvArg/edit?usp=sharing)

**Understanding and Planning LLMs**

[Blazing through these lectures by FullStack Deep learning will let you get pretty much all you need to know about LLMs](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023)

*So simple question - Should I train and/or invest in working with an LLM?*
- I think the answer is "it" depends - but know that it's probably expensive. So see if you can get close with prompt tuning, and if you can't then fine-tuning it on your own data, then consider model distillation, and finally think about full training.

## Baseline Testing 

**Also covered further in pre-requisite readings**

*Make your test dataset before you train anything*

Setting up an appropriate baseline is an important step that many candidates forget. There are three different baselines that you should think about:

- *Random baseline*: if your model predicts everything randomly, what's the expected performance?
- *Human baseline*: how well would humans perform on this task?
- *Simple heuristic*: for example, for the task of recommending the app to use next on your phone, the simplest model would be to recommend your most frequently used app. If this simple heuristic can predict the next app accurately 70% of the time, any model you build has to outperform it significantly to justify the added complexity.

**Testing LLMs is hard**

But here's [the best we've been able to figure out](https://youtu.be/Fquj2u7ay40?t=1355) - [research here is always progressing](https://arxiv-sanity-lite.com/?q=testing+large+language+models+in+production&rank=search&tags=&pid=&time_filter=&svm_c=0.01&skip_have=no)

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

Here's the last thing I saw that showed the latest changes in the landscape: [A Shift in ML Deployment by James Detweiler (Felicis VC)](https://www.felicis.com/insight/a-shift-in-ml-deployment)

1. Know your use-case's deployment platform e.g. Mobile, Web, Edge, etc.
2. Find the stack/toolkit/library that works on your platform and with company requirements e.g Tensorflow Lite (Mobile/Edge), Google's Vertex AI prediction (SaaS), Torchserve/TFX (Backend and enterprise grade) and BentoML (Backend but simpler)
3. Understand your use-case's constraints e.g. real-time for video or batch for recommendation engines
4. Optimize for time/cost/performance/hardware

[Full Stack Deep Learning - Lecture 11: Deployment & Monitoring](https://fullstackdeeplearning.com/spring2021/lecture-11/)

UPDATE: [Lecture 5: Deployment](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/)

**LLM - Large Language Models** *by popular request*

I've found that deployment depends on model needs but Hugging Face has done a great job providing a API interface that "just works".
- [Deploying LLMs via HuggingFace](https://huggingface.co/blog/inference-endpoints-llm)
- Otherwise the same stuff as above

## Monitoring

[How to make sure you know when ML systems fail and you can see and know it](https://docs.google.com/presentation/d/1tuCIbk9Pye-RK1xqiiZXPzT8lIgDUL6CqBkFSYZXkbY/edit?usp=sharing)

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

## How do I keep up with the AI, Data Science and ML world?
1. I follow a few newsletters like the Gradient, and The Batch and augment them by the RSS feed below
2. I tend to sometimes look at [Arxiv Sanity](https://www.arxiv-sanity.com/)
3. I look at popular topics on Twitter and the common Hashtags.
4. I tend to loosely follow the RSS feeds of the following blogs (I've uploaded the OPML file for this in this repo):

- [Machine Learning Blog | ML@CMU | Carnegie Mellon University](https://blog.ml.cmu.edu)
- [KDnuggets](https://www.kdnuggets.com)
- [Meta Research](https://research.facebook.com/)
- [The TWIML AI Podcast (formerly This Week in Machine Learning & Artificial Intelligence)](https://twimlai.com)
- [MachineLearningMastery.com](https://machinelearningmastery.com/)
- [Synced](https://syncedreview.com)
- [fast.ai](https://www.fast.ai/)
- [MIT News - Computer Science and Artificial Intelligence Laboratory](https://www.csail.mit.edu/)
- [The Gradient](https://thegradient.pub/)
- [DeepMind](https://www.deepmind.com)
- [Paperspace Blog](https://blog.paperspace.com/)
- [PyTorch - Medium](https://medium.com/pytorch?source=rss----512b8efdf2e7---4)
- [MLOps Community](https://mlops.community)
- [ScienceDaily - Artificial Intelligence](https://www.sciencedaily.com/news/computers_math/artificial_intelligence/)
- [Taiwan AILabs](https://ailabs.tw)
- [The Official Blog of BigML.com](https://blog.bigml.com)
- [Arize AI](https://arize.com/)
- [The TensorFlow Blog](https://blog.tensorflow.org/)
- [The AI Blog](https://blogs.microsoft.com/ai/)
- [PyTorch Website](https://pytorch.org/)
- [The Stanford AI Lab Blog](http://ai.stanford.edu/blog/)
- [Google AI Blog](http://ai.googleblog.com/)
- [TruEra](https://truera.com/)
- [OpenAI](https://blog.openai.com)
- [The Berkeley Artificial Intelligence Research Blog](http://bair.berkeley.edu/blog/)
- [neptune.ai](https://neptune.ai/)
- [Apple Machine Learning Research](https://machinelearning.apple.com)

## Acknowledgements and references

[Stanford CS329S Course by Chip Hyuen - CS 329S | Syllabus](https://stanford-cs329s.github.io/syllabus.html)

[Full Stack Deep Learning by Josh Tobin and Sergey Karayev](https://fullstackdeeplearning.com/)

[Rules of Machine Learning: | Google Developers](https://developers.google.com/machine-learning/guides/rules-of-ml)

[Google Model Tuning Playbook](https://github.com/google-research/tuning_playbook)

[Labelling Guidelines by Eugene Yan](https://eugeneyan.com/writing/labeling-guidelines/)

[How to Develop Annotation Guidelines by Prof. Dr. Nils Reiter](https://nilsreiter.de/blog/2017/howto-annotation)

