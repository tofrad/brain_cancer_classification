# Brain Cancer Classification
Train a Neural Network to classify the type of braincancer from MRI images.

The data used for training is from following kaggle Project:

  https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c

<table>
  <tr>
    <th style="text-align: center">Examples from Brain Cancer Dataset</th>
  </tr>
  <tr>
    <td><img src= https://github.com/user-attachments/assets/05240ef8-e105-4bc0-84e0-2d3e0ace636c /></td>
  </tr>
  

<table>
  <tr>
    <th colspan="2" style="text-align: center">Class Distribution of the Train-Dataset containing 90% of images</th>
  </tr>
  <tr>
    <td style="width: 66%">
      <img src="https://github.com/user-attachments/assets/568dff8f-4624-40cd-a77c-44aa6f08d420" style="width: 100%"/>
    </td>
    <td style="width: 34%; vertical-align: top">
      <table>
        <tr><th>Class</th><th>Amount</th></tr>
        <tr><td>Meningioma</td><td>874</td></tr>
        <tr><td>Astrocitoma</td><td>580</td></tr>
        <tr><td>_NORMAL</td><td>522</td></tr>
        <tr><td>Schwannoma</td><td>465</td></tr>
        <tr><td>Neurocitoma</td><td>457</td></tr>
        <tr><td>Carcinoma</td><td>251</td></tr>
        <tr><td>Papiloma</td><td>237</td></tr>
        <tr><td>Oligodendroglioma</td><td>224</td></tr>
        <tr><td>Glioblastoma</td><td>204</td></tr>
        <tr><td>Ependimoma</td><td>150</td></tr>
        <tr><td>Tuberculoma</td><td>145</td></tr>
        <tr><td>Meduloblastoma</td><td>131</td></tr>
        <tr><td>Germinoma</td><td>100</td></tr>
        <tr><td>Granuloma</td><td>78</td></tr>
        <tr><td>Ganglioglioma</td><td>61</td></tr>
      </table>
    </td>
  </tr>
</table>

# Using ResNet50V2 with weights from imagenet
Using a pre-trained version of ResNet with the last 7 layers of the model made trainable. The 4D output is then average pooled with kernel size 4x4, flattended and densed into a output layer of size 15, corresponding to the 15 classes.

<table>
  <tr>
    <th colspan="1" style="text-align: center">Accuracy through training epochs</th>
    <th colspan="1" style="text-align: center">Confusion Matrix from Validation Datset</th>
  </tr>
  <tr>
    <td><img src= https://github.com/user-attachments/assets/7ec8c9dc-3304-450c-82c7-19ab7c0f12f9 /></td>
    <td><img src= https://github.com/user-attachments/assets/f756acf9-22ae-43f1-99e3-2cecd5ae1610 /></td>
  </tr>
</table>
