pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
cd Dassl.pytorch/
pip install -r requirements.txt
python setup.py develop
cd ..

# update setuptools package 
pip install setuptools==59.5.0


pip install ftfy==6.1.1
pip install nltk==3.7
pip install timm==0.6.11
pip install transformers==4.23.1
pip install wget
pip install regex
pip install tqdm

