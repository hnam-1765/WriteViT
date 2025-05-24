 # WriteViT: Handwritten Text Generation with Vision Transformer

  <p align='center'>
  <b>
    <a href="https://arxiv.org/abs/2505.13235">ArXiv</a>
    | 
    <a href="https://arxiv.org/pdf/2505.13235">Paper</a>
  </b>
</p> 

 
 <p align="center">
<img src=Figures/Architecture.png width="500"/>
</p>

<!-- 
<img src="Figures/Result.gif" width="800"/>
 -->


  
## Software environment

- Python 3.7
- PyTorch >=1.4

## Setup & Training
Please refer to `INSTALL.md` for installation instructions of required libraries.

To visualize generated handwriting during training, you can modify the settings in `params.py`.



Download Dataset files and model from https://drive.google.com/drive/folders/1ZgYS6-6l6fjKY75RJipONBByujIgf-uE?usp=sharing

Quick setup with terminal:

```bash
git https://github.com/hnam-1765/WriteViT.git
cd WriteViT
pip install --upgrade --no-cache-dir gdown
gdown --id 1ZgYS6-6l6fjKY75RJipONBByujIgf-uE && unzip files.zip && rm files.zip
```

To train the model

```
python train.py
```

If you want to use ```wandb``` please install it and change your auth_key in the ```train.py``` file. 

You can also modify different hyperparameters in  ```params.py``` file.

The dataset is organized as a dictionary containing lists of writer samples 

```python
{
'train': [{writer_1:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_2:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
'test': [{writer_3:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]}, {writer_4:[{'img': <PIL.IMAGE>, 'label':<str_label>},...]},...], 
}
```
 <!-- ## Run Demo using Docker
```
 docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/ankankbhunia-hwt:latest python app.py
 ``` -->

## Handwriting generation results

 <p align="center">
<img src=Figures/Generation.png width="1000"/>
</p>


## Handwriting reconstruction results
 Reconstruction results.

 <p align="center">
<img src=Figures/Reconstruction.png width="1000"/>
</p>

<!-- 
<img src="Figures/result.jpg" >

<img src="Figures/recons2.jpg" >
 -->


 
