# Introduction

This repo presents code for a deep-learning-based algorithm for
**detecting violence**, **fire** and **human presence** in indoor or outdoor environments. The algorithm can
accurately detect the following scenarios: fight, fire, car crash, and even
more.

To detect other scenarios you have to add **descriptive text label** of a
scenario in `settings.yaml` file under `labels` key. At this moment model can
detect 16`+1` scenarios, where one is default `Unknown` label. You can change,
add or remove labels according to your use case. The model is trained on wide
variety of data. The task for the model at training was to predict similar
vectors for image and text that describes well a scene on the image. Thus model
can generalize well on other scenarios too if you provide proper textual
information about a scene of interest.
<a name="howtorun"/>

# How to Run

First install requirements:
`pip install -r requirements.txt`

Dataset can be taken from: 

https://github.com/sukhitashvili/violence-detection/tree/main/data

https://drive.google.com/open?id=1qpnajiy9wa5dZStqIhFHVgy2hE_fK4fb

To test the model you can either run:
`python run.py --image-path ./data/7.jpg`

Or you can test it through web app:
`streamlit run app.py`

Or you can see the example code in `tutorial.ipynb` jupyter notebook

Or incorporate this model in your project using this code:

```python
from model import Model
import cv2

model = Model()
image = cv2.imread('./your_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
label = model.predict(image=image)['label']
print('Image label is: ', label)
```

<a name="results"></a>

# Results

Below are the resulting images. I used the model to make predictions
on each frame of the videos and print model's predictions on the left side of
frame of saved videos. In case of images, titles are model's predictions. You
can find code that produces that result in `tutorial.ipynb` jupyter notebook.

### Result Images

![WhatsApp Image 2024-04-24 at 4 15 09 PM](https://github.com/aditisharma132/SP_Bot/assets/63997962/36efe872-2ff9-4fb9-91e3-0ccf8f75ff8d)
![WhatsApp Image 2024-04-24 at 4 40 14 PM](https://github.com/aditisharma132/SP_Bot/assets/63997962/c1a38acd-aa99-4452-9f7f-441c6edd7a2f)
![Screenshot 2024-04-23 210344](https://github.com/aditisharma132/SP_Bot/assets/63997962/d10b07db-e497-426d-8987-1bdd91570afb)
![Screenshot 2024-04-23 210354](https://github.com/aditisharma132/SP_Bot/assets/63997962/16b74561-a58e-4a37-986a-eaddaae5f209)
![Screenshot 2024-04-23 210933](https://github.com/aditisharma132/SP_Bot/assets/63997962/9cb49fa5-8bae-4722-b5f5-f20f79846e18)

![Screenshot 2024-04-24 163430](https://github.com/aditisharma132/SP_Bot/assets/63997962/9ff61920-65fb-4a58-b8fd-664608872b1d)
![Screenshot 2024-04-24 163335](https://github.com/aditisharma132/SP_Bot/assets/63997962/947ee768-fe12-462b-ae0e-2c1b936da9ea)



<a name="work"></a>

# Further Work

For further enhancements like: Batch processing support for speedup, return of
multiple suggestions, threshold fine-tuning for specific data, ect. contact me:

My
Linkedin: ![Aditi Sharma][https://www.linkedin.com/in/aditi-sharma-663709202/]

