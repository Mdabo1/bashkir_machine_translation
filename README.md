# Russian - Bashkir bilingual neural translator
Files from load folder are available here - https://drive.google.com/drive/folders/1yzh4ZnqG_q5UcNPvE4ALnHTmCElEFwqV?usp=sharing  
## Training process
Trained for 90 epochs using 524k sentence pairs for 2 days total (batch size = 64, hidden size = 124, GPU: RTX 3060)  

Vocab size ru: 83333  
Vocab size ba: 72647  

NNLog for Russian-Bashkir pair: -0.145  
NNLog for Bashkir-Russian pair: -0.195


### Russian - Bashkir pair training loss

![Russian_Bashkir](https://github.com/Mdabo1/bashkir_machine_translation/assets/122386960/1712a5f6-6254-429d-bcc1-85833e927e68)

### Bashkir - Russian pair training loss

![Bashkir_Russian](https://github.com/Mdabo1/bashkir_machine_translation/assets/122386960/979f3ddc-93d9-4886-9491-94c0918cdda0)

## Evaluation 

BLEU score: 0.285

### Streamlit translation result example

Still big space for improvement ;)  

![1](https://github.com/Mdabo1/bashkir_machine_translation/assets/122386960/971e123d-bda2-4d52-a6cf-93796246f0db)

## Acknoledgments

AigizK for providing parallel corpora text https://github.com/AigizK/bashkort-parallel-corpora
