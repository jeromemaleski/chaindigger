# Chaindigger camera 
For interacting with chaindigger camera and files

## Interactively explore raw image and histogram
RAW_to_RGB_explore.ipynb

## Convert raw to rgb basic (basic debayer no interpolation)
RAW_to_RGB_chaindigger.py

## Convert raw to rgb (use color package debayer)
RAW_to_RGB_chaindigger_HQ.py

## Apply adaptive histogram equalization with CLAHE
The chaindigger operates under a wide range lighting conditions and shadow so CLAHE from skimage is used to adjust exposure and reduce shadows.

### Demonstrating exposure adjustment

No equalization or exposure adjustment\
![low light no equalization or exposure adjustment](/assets/noeqal.png)

Apply CLAHE clip=0.3\
![low light apply CLAHE](/assets/eqalization_CLAHE.png)

### Demonstrating shadow reduction

No equalization or exposure adjustment\
![shadow](/assets/shadow_noeq.png)

Apply CLAHE clip=0.3
![shadow apply CLAHE](/assets/shadow_CLAHE.png)
