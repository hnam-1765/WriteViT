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


