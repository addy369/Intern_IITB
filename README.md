
<h4>This repository is a collection of work done as part of my internship at Medal Labs IITB under Dr Amit Sethi</h5>

Problem Statement: Domain Adaptation/generalization on Histopathology images

Papers implemented:

1)[Deep domain generalization via conditional invariant adversarial networks](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf)

2)[Unsupervised Domain Adaptation by Backpropagation](http://proceedings.mlr.press/v37/ganin15.pdf)


There is also a code on binary classification in Pytorch and can be used as a base for various tasks easily

The dataset used is modified BACH data.

<H3>HOW TO USE</H3>

<H4>Deep domain generalization via conditionally invariant adversarial networks</H4>

Sample image name is

ROI__0__A05__layer__0__3663__12676__2196__1249___0_400.png where 05 is the domain label<br/>
To use it on custom images file name of images must be in image folder format with file name in form of first___second___domain_label <br/>

Image size is 400 * 400

Classification is binary class i.e between benign and insitu cancer

<H4>Unsupervised domain adaptation by backpropogation</H4>

To use it on custom images file name of images must be in image folder format(binary class only)<br/>

Test data is domain 1(label of domain is 1) and target domain is domain 0(label of domain is 0)<br/> 

Binary classification task of benign vs insitu

<H4>Binary Classification</H4>
To use it on custom images file name of images must be in image folder format <br/>
<br/>

All the codes can be run by typing in terminal python main.py






